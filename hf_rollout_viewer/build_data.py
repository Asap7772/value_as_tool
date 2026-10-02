"""Build the static data behind the rollout-viewer Space.

Output: one gzipped JSON file per (experiment, benchmark, problem, method), holding every seed of
that method on that problem, plus per-benchmark indexes and one file per problem.

Stored verbatim, the trajectories come to ~23 GB, mostly repeats:
- every value-solver turn re-sends the whole conversation;
- verifier and reviser prompts repeat the candidate;
- attempt-conditioned verifier prompts repeat the evidence pack of eight prior attempts.
Here each text is stored once. Later prompts carry ⟦key⟧ markers that the viewer expands on demand.
Continuation calls keep only the messages that are new relative to the call they extend.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
MIN_NEEDLE = 160
BENCHMARKS = {"imo_proof": "IMO-ProofBench", "proofbench": "ProofBench"}
STATUS_CODES = {
    "accepted": "A",
    "cycle_limit": "L",
    "completed": "C",
    "protocol_error": "P",
    "budget_exhausted": "B",
    "context_exhausted": "X",
    "failed": "F",
}
VERDICT_CODES = {"correct": "c", "minor_fix": "m", "critical_flaw": "x"}

BASE_ARMS = [
    ("direct", "Direct", "One solver call, no verifier."),
    (
        "gvr",
        "GVR",
        "Generator → Verifier → Reviser/Generator, up to 3 candidates; the verifier returns "
        "a verdict and a ≤600-char critique.",
    ),
    (
        "gvr_subagents",
        "GVR + subagents",
        "GVR where the generator may spawn up to 3 fresh-context research subagents.",
    ),
    (
        "gvr_reference",
        "GVR + reference",
        "GVR whose verifier also sees the benchmark reference proof "
        "(copied reference spans are filtered from feedback).",
    ),
    (
        "gvr_rationale_score",
        "GVR · rationale+score",
        "GVR whose verifier returns a success probability and a ≤1200-char rationale.",
    ),
    (
        "gvr_reference_rationale_score",
        "GVR + reference · rationale+score",
        "Reference-assisted verifier with the rationale+score feedback.",
    ),
    (
        "value_tool",
        "Value tool",
        "The solver may call query_success_probability up to 3 times; a fresh-context "
        "verifier sees the partial trace and returns only a probability.",
    ),
    (
        "value_tool_rationale_score",
        "Value tool · rationale+score",
        "Value tool that also returns a rationale.",
    ),
    (
        "cch_plan_work_review",
        "Plan → Work → Review",
        "Claude-Code-Harness-style plan, work, fresh review and bounded retakes (blind).",
    ),
]
MODES = [
    ("solutions", "Full solutions"),
    ("solution_summary", "Solution summaries"),
    ("thinking_summary", "Solution+thinking summaries"),
]


def attempt_arms():
    arms = []
    for interaction, interaction_label in (("gvr", "GVR"), ("value_tool", "Value tool")):
        for mode, mode_label in MODES:
            for feedback in ("legacy", "rationale"):
                for gold in (False, True):
                    arm = f"attempt_{mode}_{interaction}_{feedback}_{'gold' if gold else 'no_gold'}"
                    label = f"{interaction_label} · {mode_label} · {feedback}" + (
                        " · gold" if gold else ""
                    )
                    description = (
                        f"{interaction_label} whose verifier sees {mode_label.lower()} of 8 "
                        "labeled prior Direct attempts (seeds 0–7 of the 9B harness run)"
                        + (" plus the reference proof" if gold else "")
                        + f"; {feedback} feedback."
                    )
                    arms.append((arm, label, description))
    return arms


EXPERIMENTS = {
    "q9_base": dict(
        artifact="qwen35_9b_harness_large_budget_v1",
        model="Qwen/Qwen3.5-9B",
        label="Qwen3.5-9B · harness methods",
        seeds=list(range(8)),
        arms=BASE_ARMS,
        description="Nine proof harnesses, 8 seeds, 8.4M-token trajectory cap, thinking enabled.",
    ),
    "q9_attempt": dict(
        artifact="qwen35_9b_attempt_conditioning_v1",
        model="Qwen/Qwen3.5-9B",
        label="Qwen3.5-9B · attempt-conditioned verifiers",
        seeds=list(range(8, 16)),
        arms=attempt_arms(),
        description=(
            "24 arms: prior-attempt evidence × GVR/value tool × feedback × reference access; "
            "fresh seeds 8–15."
        ),
    ),
    "q27_base": dict(
        artifact="qwen38_27b_harness_v4",
        model="Qwen/Qwen3.8-27B",
        label="Qwen3.8-27B · harness methods",
        seeds=list(range(3)),
        arms=BASE_ARMS,
        description="Nine proof harnesses, 3 seeds, 229k-token trajectory cap.",
    ),
}


def dump_gz(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    return len(payload)


class Markers:
    """Longest-first replacement of already-shown texts by ⟦key⟧ markers."""

    def __init__(self):
        self.needles = {}

    def add(self, text, key):
        if isinstance(text, str) and len(text) >= MIN_NEEDLE and text not in self.needles:
            self.needles[text] = f"⟦{key}⟧"
        stripped = text.strip() if isinstance(text, str) else ""
        if len(stripped) >= MIN_NEEDLE and stripped not in self.needles:
            self.needles[stripped] = f"⟦{key}⟧"

    def apply(self, value):
        if isinstance(value, str):
            if len(value) < MIN_NEEDLE:
                return value
            for needle in sorted(self.needles, key=len, reverse=True):
                if needle in value:
                    value = value.replace(needle, self.needles[needle])
            return value
        if isinstance(value, list):
            return [self.apply(item) for item in value]
        if isinstance(value, dict):
            return {key: self.apply(item) for key, item in value.items()}
        return value


def parse_json(text):
    if not isinstance(text, str) or not text.strip().startswith(("{", "[")):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def tool_call_view(call):
    function = call.get("function") or {}
    name = call.get("name") or function.get("name")
    arguments = call.get("arguments", function.get("arguments"))
    parsed = parse_json(arguments) if isinstance(arguments, str) else arguments
    return {"id": call.get("id"), "name": name, "args": parsed if parsed is not None else arguments}


def message_signature(message):
    tool_ids = tuple(call.get("id") for call in message.get("tool_calls") or [])
    return (
        message.get("role"),
        message.get("content") or "",
        message.get("tool_call_id"),
        message.get("name"),
        tool_ids,
    )


def response_signature(response):
    message = response.get("message") or {}
    tool_ids = tuple(call.get("id") for call in message.get("tool_calls") or [])
    return ("assistant", message.get("content") or "", None, None, tool_ids)


def common_prefix(left, right):
    size = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        size += 1
    return size


def message_view(message, markers, response_owner):
    role = message.get("role")
    if role == "assistant":
        owner = response_owner.get(message_signature(message))
        if owner is not None:
            return {"role": "assistant", "ref": owner}
        view = {"role": "assistant", "content": markers.apply(message.get("content") or "")}
        if message.get("reasoning_content"):
            view["reasoning"] = markers.apply(message["reasoning_content"])
        if message.get("tool_calls"):
            view["tool_calls"] = [tool_call_view(call) for call in message["tool_calls"]]
        return view
    view = {"role": role}
    if message.get("name"):
        view["name"] = message["name"]
    parsed = parse_json(message.get("content"))
    if parsed is not None and isinstance(parsed, (dict, list)):
        view["json"] = markers.apply(parsed)
    else:
        view["content"] = markers.apply(message.get("content") or "")
    return view


def trajectory(result, judge):
    request = result["request"]
    markers = Markers()
    markers.add(request.get("solver_prompt"), "prompt")
    markers.add(request.get("problem"), "problem")
    markers.add(request.get("reference_proof"), "reference")
    evidence = request.get("verifier_evidence")
    if evidence:
        compact = {key: str(evidence[key]) for key in ("mode", "pack_sha256", "content")}
        markers.add(json.dumps(compact, ensure_ascii=False, separators=(",", ":")), "evidence")
        markers.add(evidence["content"], "evidence")

    calls, conversations, response_owner = [], {}, {}
    for call in result.get("calls", []):
        index = call["index"]
        response = call.get("response") or {}
        message = response.get("message") or {}
        usage = call.get("usage") or {}
        signatures = [message_signature(m) for m in call.get("messages", [])]
        best, best_size = None, 0
        for previous, conversation in conversations.items():
            size = common_prefix(signatures, conversation)
            if size > best_size:
                best, best_size = previous, size
        skip = best_size if best_size >= 2 else 0
        view = {
            "i": index,
            "role": call["role"],
            "cycle": call.get("cycle"),
            "label": call.get("label"),
            "max_tokens": call.get("max_tokens"),
            "finish": response.get("finish_reason"),
            "usage": {
                "prompt": usage.get("prompt_tokens"),
                "completion": usage.get("completion_tokens"),
                "reasoning": usage.get("reasoning_tokens"),
            },
            "tools": [(tool.get("function") or {}).get("name") for tool in call.get("tools") or []],
            "messages": [
                message_view(m, markers, response_owner) for m in call.get("messages", [])[skip:]
            ],
            "reasoning": message.get("reasoning") or "",
            "content": message.get("content") or "",
            "tool_calls": [tool_call_view(tc) for tc in message.get("tool_calls") or []],
        }
        if skip:
            view["cont"] = {"call": best, "n": skip}
        if call.get("error"):
            view["error"] = str(call["error"])[:4000]
        calls.append(view)
        conversations[index] = signatures + ([response_signature(response)] if response else [])
        if response:
            response_owner[response_signature(response)] = index
        markers.add(view["reasoning"], f"r{index}")
        markers.add(view["content"], f"c{index}")
        for tool_call in view["tool_calls"]:
            if isinstance(tool_call["args"], dict):
                for name, value in tool_call["args"].items():
                    markers.add(value, f"a{index}.{name}")

    content_owner = {}
    for view in calls:
        if view["content"].strip():
            content_owner[view["content"].strip()] = view["i"]
    candidates = []
    for candidate in result.get("candidates", []):
        item = {"cycle": candidate["cycle"], "role": candidate["role"]}
        owner = content_owner.get(candidate["content"].strip())
        if owner is None:
            item["content"] = candidate["content"]
        else:
            item["call"] = owner
        candidates.append(item)
    final_output = result.get("final_output") or ""
    final_owner = content_owner.get(final_output.strip())
    final = {"call": final_owner} if final_owner is not None else {"content": final_output}

    subagents = []
    for record in result.get("subagents", []):
        subagents.append(
            {key: record.get(key) for key in ("call_index", "cycle", "task_index", "task")}
            | {"context_excerpt": markers.apply(record.get("context_excerpt") or "")}
        )

    budget = result.get("budget") or {}
    total = result.get("usage") or {}
    out = {
        "run_id": result["run_id"],
        "arm": result["harness_id"],
        "seed": request["seed"],
        "status": result["status"],
        "error": (str(result["error"])[:4000] if result.get("error") else None),
        "tokens": {
            "generated": budget.get("spent_generated_tokens"),
            "prompt": total.get("prompt_tokens"),
            "reasoning": total.get("reasoning_tokens"),
        },
        "calls": calls,
        "candidates": candidates,
        "final": final,
        "verdicts": [
            {key: value for key, value in v.items() if value not in (None, "")}
            for v in result.get("verdicts", [])
        ],
        "value_estimates": [
            {
                key: v.get(key)
                for key in (
                    "query_index",
                    "probability",
                    "rationale",
                    "solver_call_index",
                    "verifier_call_index",
                )
            }
            for v in result.get("value_estimates", [])
        ],
        "subagents": subagents,
        "transitions": [
            [t["cycle"], t["source"], t["action"], t["target"]]
            + ([t["detail"]] if t.get("detail") else [])
            for t in result.get("transitions", [])
        ],
    }
    template = None
    if judge:
        out["judge"] = {key: judge.get(key) for key in ("score", "judge_status", "correct")}
        judge_result = judge.get("judge_result") or {}
        out["judge"]["raw"] = judge_result.get("raw_response")
        out["judge"]["error"] = judge_result.get("error")
        usage = judge.get("judge_usage") or {}
        out["judge"]["usage"] = {
            "prompt": usage.get("prompt_tokens"),
            "completion": usage.get("completion_tokens"),
        }
        messages = judge_result.get("messages") or []
        prompt = messages[0]["content"] if messages else ""
        solution = final_output.strip()
        if prompt and solution and solution in prompt:
            template = prompt.replace(solution, "⟦final⟧", 1)
            out["judge"]["template"] = hashlib.sha256(template.encode()).hexdigest()[:12]
        elif prompt:
            out["judge"]["prompt"] = prompt
    return out, template


def problem_group(task):
    experiment, benchmark, problem_id, run_ids, out_root = task
    base = ARTIFACTS / EXPERIMENTS[experiment]["artifact"]
    by_arm, templates, static, packs, cells, raw_bytes = (
        defaultdict(list),
        {},
        None,
        {},
        defaultdict(list),
        0,
    )
    for run_id in run_ids:
        result = json.loads((base / "solve" / "runs" / run_id / "result.json").read_text())
        judge_path = base / "judge" / "runs" / run_id / "result.json"
        judge = json.loads(judge_path.read_text()) if judge_path.exists() else None
        view, template = trajectory(result, judge)
        if template is not None:
            templates[view["judge"]["template"]] = template
        by_arm[view["arm"]].append(view)
        request = result["request"]
        if static is None:
            static = {
                "id": problem_id,
                "benchmark": benchmark,
                "problem": request.get("problem"),
                "solver_prompt": request.get("solver_prompt"),
                "reference": request.get("reference_proof"),
            }
        evidence = request.get("verifier_evidence")
        if evidence and evidence["mode"] not in packs:
            packs[evidence["mode"]] = {
                "content": parse_json(evidence["content"]) or evidence["content"],
                "pack_sha256": evidence.get("pack_sha256"),
                "labels": evidence.get("labels"),
                "source_summary": evidence.get("source_summary"),
            }
        verdicts = "".join(VERDICT_CODES.get(v["verdict"], "?") for v in result.get("verdicts", []))
        cells[view["arm"]].append(
            [
                view["seed"],
                (judge or {}).get("score"),
                STATUS_CODES.get(result["status"], "?"),
                verdicts
                or (
                    f"q{len(result.get('value_estimates', []))}"
                    if result.get("value_estimates")
                    else ""
                ),
                view["tokens"]["generated"],
                len(view["calls"]),
            ]
        )
    for arm, views in by_arm.items():
        views.sort(key=lambda v: v["seed"])
        raw_bytes += dump_gz(
            Path(out_root) / "rollouts" / experiment / benchmark / problem_id / f"{arm}.json.gz",
            views,
        )
    for arm in cells:
        cells[arm].sort()
    return experiment, benchmark, problem_id, static, templates, packs, dict(cells), raw_bytes


def identity(run_dir):
    return run_dir.name, json.loads((run_dir / "run.json").read_text())["identity"]


def schedule(experiment):
    base = ARTIFACTS / EXPERIMENTS[experiment]["artifact"] / "solve" / "runs"
    groups = defaultdict(list)
    with ThreadPoolExecutor(32) as pool:
        for run_id, ident in pool.map(identity, sorted(base.iterdir())):
            groups[(ident["benchmark"], ident["problem_id"])].append(run_id)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS)
    )
    parser.add_argument(
        "--problems", nargs="*", default=None, help="limit to these problem ids (smoke builds)"
    )
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    out = args.out.resolve()
    tasks = []
    for experiment in args.experiments:
        for (benchmark, problem_id), run_ids in schedule(experiment).items():
            if args.problems is None or problem_id in args.problems:
                tasks.append((experiment, benchmark, problem_id, run_ids, str(out)))
    print(f"{len(tasks)} problem groups, {sum(len(t[3]) for t in tasks)} trajectories", flush=True)

    problems, index, raw_total = {}, defaultdict(dict), 0
    with ProcessPoolExecutor(args.workers) as pool:
        for done, (
            experiment,
            benchmark,
            problem_id,
            static,
            templates,
            packs,
            cells,
            raw_bytes,
        ) in enumerate(pool.map(problem_group, tasks, chunksize=1), 1):
            entry = problems.setdefault(
                (benchmark, problem_id), {**static, "judge_templates": {}, "packs": {}}
            )
            entry["judge_templates"].update(templates)
            entry["packs"].update(packs)
            index[(benchmark, problem_id)][experiment] = cells
            raw_total += raw_bytes
            if done % 50 == 0 or done == len(tasks):
                print(
                    f"  {done}/{len(tasks)} groups, {raw_total / 1e9:.2f} GB uncompressed rollouts",
                    flush=True,
                )

    for (benchmark, problem_id), entry in problems.items():
        dump_gz(out / "problems" / benchmark / f"{problem_id}.json.gz", entry)
    for benchmark in BENCHMARKS:
        rows = []
        for (bench, problem_id), entry in sorted(problems.items()):
            if bench != benchmark:
                continue
            labels = next(
                (pack["labels"] for pack in entry["packs"].values() if pack.get("labels")), None
            )
            rows.append(
                {
                    "id": problem_id,
                    "preview": " ".join((entry["problem"] or "").split())[:220],
                    "bank": [bool(label["correct"]) for label in labels] if labels else None,
                    "cells": index[(bench, problem_id)],
                }
            )
        dump_gz(out / "index" / f"{benchmark}.json.gz", {"benchmark": benchmark, "problems": rows})
    meta = {
        "benchmarks": [{"id": key, "label": label} for key, label in BENCHMARKS.items()],
        "experiments": [
            {
                "id": key,
                "label": spec["label"],
                "model": spec["model"],
                "seeds": spec["seeds"],
                "description": spec["description"],
                "artifact": spec["artifact"],
                "arms": [
                    {"id": arm, "label": label, "description": description}
                    for arm, label, description in spec["arms"]
                ],
            }
            for key, spec in EXPERIMENTS.items()
            if key in args.experiments
        ],
        "status_codes": {code: name for name, code in STATUS_CODES.items()},
        "verdict_codes": {code: name for name, code in VERDICT_CODES.items()},
        "cell_fields": ["seed", "score", "status", "verdicts", "generated_tokens", "calls"],
    }
    (out / "index").mkdir(parents=True, exist_ok=True)
    (out / "index" / "experiments.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(
        f"done: {len(problems)} problems; {raw_total / 1e9:.2f} GB uncompressed rollouts",
        flush=True,
    )


if __name__ == "__main__":
    main()
