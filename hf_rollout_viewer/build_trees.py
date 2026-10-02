"""Build the static data behind the GVR tree Space.

Input: the branched-GVR artifact root, whose `export/` holds the flattened trees, nodes and
verifications written by scripts/export_gvr_tree_dataset.py. The per-call reasoning traces and
prompts are read from each tree's `solve/runs/<run>/result.json`.

Output, all gzipped JSON except the small index files:
- index/trees.json.gz: one row per tree for the problem list (split, difficulty, spine labels).
- index/overview.json: dataset-level counts and tables for the Overview tab.
- index/prompts.json: one example prompt per call kind, with the problem, the candidate under
  review and the verifier critique replaced by ⟦problem⟧, ⟦candidate⟧, ⟦critique⟧ and ⟦category⟧.
- trees/<problem>.json.gz: one tree: problem, gold answer, every candidate with its judge label,
  every verdict with its critique, and the distinct final answers.
- reasoning/<problem>.json.gz: the private reasoning of every call in that tree, loaded on demand.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from build_data import dump_gz

ARTIFACT_ROOT = Path("/checkpoint/fort/anikaitsingh/value_as_tool/qwen35_9b_gvr_tree_arxivmath_v1")
ROUNDS = 10
LABEL = re.compile(r"^r(\d+)\.b(\d+)\.")


def bucket(rate):
    if rate == 0:
        return "zero"
    if rate == 1:
        return "one"
    return "low" if rate < 0.5 else "high"


def call_kind(label):
    """`r03.b1.recheck.t0` -> `recheck`, `r03.b1.verify.recovery` -> `verify.recovery`."""
    kind = LABEL.sub("", label)
    return kind[:-3] if kind.endswith(".t0") else kind


def with_markers(text, replacements):
    for needle, marker in replacements:
        for variant in (needle, needle.strip()):
            if variant and variant in text:
                text = text.replace(variant, marker)
                break
    return text


def prompt_examples(calls, problem, nodes, verdicts):
    """System and user messages of the first call of each kind, with shared texts as markers."""
    spine = {node["cycle"]: node for node in nodes if node["on_spine"]}
    critique = {(v["point"], v["branch"]): v.get("critique") or "" for v in verdicts}
    category = {(v["point"], v["branch"]): v.get("fault_category") or "" for v in verdicts}
    examples = {}
    for call in calls:
        kind = call_kind(call["label"])
        if kind in examples:
            continue
        replacements = [(problem, "⟦problem⟧")]
        position = LABEL.match(call["label"])
        if position:
            point, branch = int(position[1]), int(position[2])
            if point in spine:
                replacements.append((spine[point]["content"], "⟦candidate⟧"))
            replacements.append((critique.get((point, branch), ""), "⟦critique⟧"))
            replacements.append((category.get((point, branch), ""), "⟦category⟧"))
        examples[kind] = {
            "label": call["label"],
            "role": call["role"],
            "messages": [
                {
                    "role": message["role"],
                    "content": with_markers(message.get("content") or "", replacements),
                }
                for message in call["messages"]
            ],
        }
    return examples


def answer_groups(nodes):
    """Distinct extracted answers in order of first appearance.

    An answer's ``correct`` is null when its candidates were labelled differently.
    """
    groups, index = [], {}
    for node in sorted(nodes, key=lambda node: node["node_call_index"]):
        text = node["extracted_answer"]
        key = " ".join(text.split()) if text else None
        if key not in index:
            index[key] = len(groups)
            groups.append({"text": key, "count": 0, "labels": set()})
        group = groups[index[key]]
        group["count"] += 1
        group["labels"].add(node["judge_correct"])
    for group in groups:
        labels = group.pop("labels")
        group["correct"] = labels.pop() if len(labels) == 1 else None
    return groups, index


def build_tree(task):
    tree, node_lines, verification_lines, result_path, out = task
    nodes = [json.loads(line) for line in node_lines]
    verdicts = [json.loads(line) for line in verification_lines]
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    calls = result["calls"]
    pid = tree["problem_id"]
    split = tree["benchmark"].removeprefix("arxivmath_")
    groups, answer_of = answer_groups(nodes)

    record = {
        "id": pid,
        "run_id": tree["run_id"],
        "split": split,
        "arxiv_id": tree["arxiv_id"],
        "problem": tree["problem"],
        "gold": tree["gold_answer"],
        "qwen36": {
            "attempts": tree.get("qwen36_attempts"),
            "correct": tree.get("qwen36_correct"),
            "pass_rate": tree.get("qwen36_pass_rate"),
            "mean_output_tokens": tree.get("qwen36_mean_output_tokens"),
        },
        "status": tree["status"],
        "generated_tokens": tree["generated_tokens"],
        "prompt_tokens": tree["prompt_tokens"],
        "answers": groups,
        "calls": [
            [
                call["index"],
                call["label"],
                call["role"],
                (call.get("usage") or {}).get("completion_tokens"),
                (call.get("response") or {}).get("finish_reason"),
            ]
            for call in calls
        ],
        "nodes": [
            {
                "call": node["node_call_index"],
                "parent": node["parent_call_index"],
                "cycle": node["cycle"],
                "point": node["point"],
                "branch": node["branch"],
                "spine": node["on_spine"],
                "mode": node["mode"],
                "answer": answer_of[
                    " ".join(node["extracted_answer"].split()) if node["extracted_answer"] else None
                ],
                "correct": node["judge_correct"],
                "judge_status": node["judge_status"],
                "finish": node["finish_reason"],
                "recovered": node["forced_recovery"],
                "tokens": node["completion_tokens"],
                "reasoning_tokens": node["reasoning_tokens"],
                "prompt_tokens": node["prompt_tokens"],
                "verdict_call": node["routing_verdict_call_index"],
                "text": node["content"],
            }
            for node in nodes
        ],
        "verdicts": [
            {
                "call": verdict["verdict_call_index"],
                "on": verdict["candidate_call_index"],
                "point": verdict["point"],
                "branch": verdict["branch"],
                "verdict": verdict["verdict"],
                "category": verdict["fault_category"],
                "critique": verdict["critique"],
                "tokens": verdict["completion_tokens"],
                "recovered": verdict["recovered"],
            }
            for verdict in verdicts
        ],
    }
    reasoning = {
        str(call["index"]): text
        for call in calls
        if (text := ((call.get("response") or {}).get("message") or {}).get("reasoning"))
    }
    out = Path(out)
    raw = dump_gz(out / "trees" / f"{pid}.json.gz", record)
    raw += dump_gz(out / "reasoning" / f"{pid}.json.gz", reasoning)

    # Overview statistics for this tree.
    label_of = {node["node_call_index"]: node["judge_correct"] for node in nodes}
    changes = Counter()
    for node in nodes:
        if node["parent_call_index"] is not None:
            before, after = label_of[node["parent_call_index"]], node["judge_correct"]
            changes[(node["mode"], f"{'C' if before else 'I'}->{'C' if after else 'I'}")] += 1
    verdicts_by_label = Counter(
        ("correct" if verdict["candidate_judge_correct"] else "incorrect", verdict["verdict"])
        for verdict in verdicts
    )
    labels = [bool(node["judge_correct"]) for node in nodes]
    spine = [bool(value) for value in tree["spine_correct"]]
    kind = (
        "mixed"
        if 0 < sum(labels) < len(labels)
        else "all_correct" if all(labels) else "all_incorrect"
    )
    change = (
        "fixed" if spine[-1] and not spine[0]
        else "broken" if spine[0] and not spine[-1]
        else "same"
    )
    row = {
        "id": pid,
        "split": split,
        "arxiv": tree["arxiv_id"],
        "pass": tree.get("qwen36_pass_rate"),
        "bucket": bucket(tree.get("qwen36_pass_rate")),
        "spine": [int(value) for value in spine],
        "correct": sum(labels),
        "nodes": len(labels),
        "kind": kind,
        "change": change,
        "answers": len(groups),
        "tokens": tree["generated_tokens"],
        "problem": tree["problem"],
    }
    examples = prompt_examples(calls, tree["problem"], nodes, verdicts)
    statuses = Counter(node["judge_status"] for node in nodes)
    return row, changes, verdicts_by_label, statuses, examples, raw


def grouped(path):
    """Yield (run_id, lines) for a JSONL file written one tree at a time."""
    with open(path, encoding="utf-8") as handle:
        for run_id, lines in itertools.groupby(handle, key=lambda line: json.loads(line)["run_id"]):
            yield run_id, list(lines)


def tasks(root, out, limit=None):
    trees = {}
    for line in (root / "export" / "trees.jsonl").read_text(encoding="utf-8").splitlines():
        tree = json.loads(line)
        trees[tree["run_id"]] = tree
    verifications = grouped(root / "export" / "verifications.jsonl")
    seen = set()
    for run_id, node_lines in grouped(root / "export" / "nodes.jsonl"):
        verification_run, verification_lines = next(verifications)
        assert verification_run == run_id and run_id not in seen, f"export out of order at {run_id}"
        seen.add(run_id)
        result = root / "solve" / "runs" / run_id / "result.json"
        yield trees[run_id], node_lines, verification_lines, str(result), str(out)
        if limit is not None and len(seen) >= limit:
            return
    assert seen == set(trees), "every exported tree needs its nodes"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="first N trees only (smoke builds)")
    args = parser.parse_args()
    root, out = args.artifact_root.resolve(), args.out.resolve()
    export = json.loads((root / "export" / "manifest.json").read_text())

    rows, examples, raw_total = [], {}, 0
    changes, verdicts, judge_statuses = Counter(), Counter(), Counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for done, (row, tree_changes, tree_verdicts, statuses, tree_examples, raw) in enumerate(
            pool.map(build_tree, tasks(root, out, args.limit), chunksize=4), 1
        ):
            rows.append(row)
            changes.update(tree_changes)
            verdicts.update(tree_verdicts)
            judge_statuses.update(statuses)
            for kind, example in tree_examples.items():
                examples.setdefault(kind, example)
            raw_total += raw
            if done % 200 == 0:
                print(f"  {done} trees, {raw_total / 1e9:.2f} GB uncompressed", flush=True)

    rows.sort(key=lambda row: (row["split"] != "train", int(row["id"].rsplit("-", 1)[-1])))
    kinds = defaultdict(Counter)
    for row in rows:
        kinds[row["bucket"]][row["kind"]] += 1
    spine_accuracy = [
        sum(row["spine"][index] for row in rows) / len(rows) for index in range(ROUNDS)
    ]
    overview = {
        "model": "Qwen/Qwen3.5-9B",
        "judge": "openai/gpt-oss-20b (answer judge: extracted \\boxed{} answer vs gold)",
        "source_dataset": "MathArena/arxivmath-training_outputs",
        "config_fingerprint": export["config_fingerprint"],
        "rounds": ROUNDS,
        "branches": 4,
        "trees": dict(Counter(row["split"] for row in rows)),
        "nodes": sum(row["nodes"] for row in rows),
        "verifications": export["counts"]["verifications"],
        "generated_tokens": sum(row["tokens"] for row in rows),
        "judge_statuses": dict(judge_statuses),
        "node_accuracy": {
            split: sum(row["correct"] for row in rows if row["split"] == split)
            / sum(row["nodes"] for row in rows if row["split"] == split)
            for split in ("train", "eval")
            if any(row["split"] == split for row in rows)
        },
        "spine_accuracy": spine_accuracy,
        "any_correct": sum(row["correct"] > 0 for row in rows) / len(rows),
        "changes": {
            mode: {key: changes[(mode, key)] for key in ("I->C", "I->I", "C->I", "C->C")}
            for mode in ("recheck", "revise", "regenerate")
        },
        "verdicts_by_label": {
            label: {
                key: verdicts[(label, key)] for key in ("correct", "minor_fix", "critical_flaw")
            }
            for label in ("correct", "incorrect")
        },
        "kinds_by_bucket": {
            name: dict(kinds[name]) for name in ("zero", "low", "high", "one") if name in kinds
        },
        "spine_change": dict(Counter(row["change"] for row in rows)),
    }
    (out / "index").mkdir(parents=True, exist_ok=True)
    dump_gz(out / "index" / "trees.json.gz", {"trees": rows})
    (out / "index" / "overview.json").write_text(json.dumps(overview, indent=1) + "\n")
    (out / "index" / "prompts.json").write_text(
        json.dumps(dict(sorted(examples.items())), indent=1, ensure_ascii=False) + "\n"
    )
    print(
        f"{len(rows)} trees, {overview['nodes']} nodes, {raw_total / 1e9:.2f} GB uncompressed "
        f"→ {sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) / 1e9:.2f} GB written"
    )


if __name__ == "__main__":
    main()
