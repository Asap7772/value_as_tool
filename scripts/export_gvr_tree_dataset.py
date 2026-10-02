"""Flatten branched GVR trees and their node judgments into a value-learning dataset.

Reads an artifact root as plain JSON (schedule, solve results, node-judge
results and the prepared benchmark rows) and writes:

- nodes.jsonl: one row per candidate, with its problem, gold answer and
  Qwen3.6 pass rate, its tree position, how it was produced, the verdict that
  routed it, its tokens and its judge label. Spine
  nodes also carry the share of their verdicts that said "correct" and the
  share of their one-step children judged correct (a Monte Carlo next-step
  value).
- verifications.jsonl: one row per verdict, joined to the judge label of the
  candidate it assessed.
- plans.jsonl: for replanning trees, one row per planned branch: the plan the
  executor received, the state it was written for and what came of it.
- trees.jsonl: one row per tree.
- manifest.json: counts and sources.

Replanning trees (harnesses gvr_replan_joint and gvr_replan_independent) have no
verdict-routed nodes, so their routing fields are null; their plans are re-parsed
from the planner calls with the harness's own parser.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from value_as_tool.judging import find_last_boxed_content

MODE = re.compile(r"\.(recheck|revise|regenerate|exec)\.t[01]$")
REPLAN_HARNESSES = ("gvr_replan_joint", "gvr_replan_independent")


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _mode(label: str) -> str:
    if label.startswith("gen."):
        return "generate"
    match = MODE.search(label)
    return match[1] if match else "unknown"


def _share(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _answer(text: str | None) -> str | None:
    """The whitespace-collapsed final answer; distinct answers are distinct keys."""

    value = find_last_boxed_content(text or "")
    return " ".join(value.split()) if value else None


def _details(solve: dict[str, Any], action: str) -> dict[int, tuple[str, dict[str, Any]]]:
    found = {}
    for transition in solve.get("transitions", []):
        if transition.get("action") == action:
            try:
                detail = json.loads(transition.get("detail") or "{}")
            except json.JSONDecodeError:
                detail = {}
            found[transition["cycle"]] = (transition["source"], detail)
    return found


def tree_plans(solve: dict[str, Any], harness_id: str) -> dict[tuple[int, int], dict[str, Any]]:
    """The plan each replanning branch received, keyed by (point, branch).

    The harness uses a planner's recovery call only when its first call had no
    valid plan, so a recorded recovery call is the one that was used.
    """

    from value_as_tool.harnesses.gvr_replan import BRANCHES, ROUNDS, parse_plan_arguments

    calls = {call["label"]: call for call in solve.get("calls", [])}
    joint = harness_id == "gvr_replan_joint"
    plans: dict[tuple[int, int], dict[str, Any]] = {}
    for point in range(1, ROUNDS + 1):
        labels = (
            [f"r{point:02d}.plan"]
            if joint
            else [f"r{point:02d}.b{branch}.plan" for branch in range(BRANCHES)]
        )
        for index, base in enumerate(labels):
            label = f"{base}.recovery" if f"{base}.recovery" in calls else base
            call = calls.get(label)
            if call is None:
                continue
            tool_calls = ((call.get("response") or {}).get("message") or {}).get("tool_calls") or []
            slots = BRANCHES if joint else 1
            parsed = None
            if len(tool_calls) == 1 and tool_calls[0].get("name") == "submit_plans":
                try:
                    arguments = tool_calls[0].get("arguments")
                    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                    if isinstance(arguments, dict):
                        parsed = parse_plan_arguments(arguments, slots)
                except json.JSONDecodeError:
                    parsed = None
            for slot in range(slots):
                plans[(point, slot if joint else index)] = {
                    "call": call,
                    "label": label,
                    "recovered": label.endswith(".recovery"),
                    "shared": joint,
                    "plan": parsed[slot] if parsed else None,
                }
    return plans


def export_tree(
    item: dict[str, Any],
    solve: dict[str, Any],
    judged: dict[str, Any] | None,
    prepared: dict[str, Any],
    *,
    include_reasoning: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    harness_id = item.get("harness_id") or "gvr_branched"
    replan = harness_id in REPLAN_HARNESSES
    calls = {call["index"]: call for call in solve.get("calls", [])}
    labels = {node["call_index"]: node for node in (judged or {}).get("nodes", [])}
    verdicts = solve.get("verdicts", [])
    candidates = solve.get("candidates", [])
    by_call = {c["call_index"]: c for c in candidates}
    by_key = {(c["cycle"], c.get("branch")): c for c in candidates}
    promotes = _details(solve, "promote")
    supports = _details(solve, "promotion_support")
    promoted = {
        (cycle, int(source.removeprefix("branch_"))): detail
        for cycle, (source, detail) in promotes.items()
    }
    spine = {c["call_index"] for c in candidates if c["cycle"] == 1 and c.get("branch") is None}
    spine |= {by_key[key]["call_index"] for key in promoted if key in by_key}
    spine_at = {by_call[index]["cycle"]: by_call[index] for index in spine}
    plans = tree_plans(solve, harness_id) if replan else {}
    shared = {
        "run_id": item["run_id"],
        "benchmark": item["benchmark"],
        "problem_id": item["problem_id"],
        "arxiv_id": prepared.get("arxiv_id"),
        "problem": item.get("problem"),
    }
    verdict_of = {(v["cycle"], v.get("branch")): v for v in verdicts}
    children: dict[int, list[bool]] = defaultdict(list)
    verdicts_on: dict[int, list[bool]] = defaultdict(list)
    probabilities_on: dict[int, list[float]] = defaultdict(list)
    for candidate in candidates:
        label = labels.get(candidate["call_index"])
        if candidate.get("parent_call_index") is not None and label is not None:
            children[candidate["parent_call_index"]].append(bool(label["correct"]))
    for verdict in verdicts:
        if verdict.get("parent_call_index") is not None:
            verdicts_on[verdict["parent_call_index"]].append(verdict["verdict"] == "correct")
            if verdict.get("success_probability") is not None:
                probabilities_on[verdict["parent_call_index"]].append(verdict["success_probability"])

    nodes = []
    for candidate in sorted(candidates, key=lambda c: c["call_index"]):
        call = calls.get(candidate["call_index"], {})
        response = call.get("response") or {}
        usage = call.get("usage") or {}
        label = labels.get(candidate["call_index"], {})
        point = candidate["cycle"] - 1
        routed = verdict_of.get((point, candidate.get("branch"))) if point and not replan else None
        parent = by_call.get(candidate.get("parent_call_index"))
        answer = _answer(candidate["content"])
        row = {
            **shared,
            "gold_answer": item.get("golden_answer"),
            "qwen36_pass_rate": prepared.get("qwen36_pass_rate"),
            "harness_id": harness_id,
            "node_call_index": candidate["call_index"],
            "parent_call_index": candidate.get("parent_call_index"),
            "cycle": candidate["cycle"],
            "point": point,
            "branch": candidate.get("branch"),
            "on_spine": candidate["call_index"] in spine,
            "role": candidate["role"],
            "mode": _mode(str(call.get("label", ""))),
            "forced_recovery": str(call.get("label", "")).endswith(".t1"),
            "routing_verdict": routed["verdict"] if routed else None,
            "routing_critique": routed.get("critique") if routed else None,
            "routing_category": routed.get("fault_category") if routed else None,
            "routing_verdict_call_index": routed["call_index"] if routed else None,
            "content": candidate["content"],
            "extracted_answer": find_last_boxed_content(candidate["content"]),
            "parent_extracted_answer": _answer(parent["content"]) if parent else None,
            "answer_changed": answer != _answer(parent["content"]) if parent else None,
            "finish_reason": response.get("finish_reason"),
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "judge_status": label.get("judge_status"),
            "judge_correct": label.get("correct"),
            "judge_prompt_sha256": label.get("judge_prompt_sha256"),
            "verdicts_correct_share": _share(verdicts_on.get(candidate["call_index"], [])),
            "verdicts_mean_probability": _mean(probabilities_on.get(candidate["call_index"], [])),
            "children_correct_share": _share(children.get(candidate["call_index"], [])),
        }
        if replan:
            key = (point, candidate.get("branch"))
            entry = plans.get(key) or {}
            plan = entry.get("plan")
            messages = call.get("messages") or []
            user = messages[1].get("content", "") if len(messages) > 1 else ""
            promotion = promoted.get((candidate["cycle"], candidate.get("branch")))
            support = supports.get(candidate["cycle"], ("", {}))[1]
            row.update(
                {
                    "plan_title": plan.title if plan else None,
                    "plan_brief": plan.brief if plan else None,
                    "plan_show_current_solution": plan.show_current_solution if plan else None,
                    "plan_success_probability": plan.success_probability if plan else None,
                    "plan_slot": plan.slot if plan else None,
                    "plan_recovered": entry.get("recovered"),
                    "plan_call_index": (entry.get("call") or {}).get("index"),
                    "executor_saw_solution": bool(parent and parent["content"] in user),
                    "promotion_rank": promotion.get("rank") if promotion else None,
                    "promotion_probability": support.get("p") if promotion else None,
                }
            )
        if include_reasoning:
            row["reasoning"] = (response.get("message") or {}).get("reasoning")
        nodes.append(row)

    verifications = []
    for verdict in sorted(verdicts, key=lambda v: (v["cycle"], v.get("branch") or 0)):
        call = calls.get(verdict["call_index"], {})
        target = labels.get(verdict.get("parent_call_index"), {})
        assessed = by_call.get(verdict.get("parent_call_index"))
        verifications.append(
            {
                **shared,
                "harness_id": harness_id,
                "point": verdict["cycle"],
                "branch": verdict.get("branch"),
                "candidate_call_index": verdict.get("parent_call_index"),
                "verdict_call_index": verdict["call_index"],
                "recovered": str(call.get("label", "")).endswith(".recovery"),
                "verdict": verdict["verdict"],
                "critique": verdict.get("critique"),
                "rationale": verdict.get("rationale"),
                "fault_category": verdict.get("fault_category"),
                "candidate_excerpt": verdict.get("candidate_excerpt"),
                "success_probability": verdict.get("success_probability"),
                "completion_tokens": (call.get("usage") or {}).get("completion_tokens"),
                "candidate_extracted_answer": _answer(assessed["content"]) if assessed else None,
                "candidate_judge_correct": target.get("correct"),
            }
        )

    plan_rows = []
    for (point, branch), entry in sorted(plans.items()):
        plan, call = entry["plan"], entry["call"]
        parent = spine_at.get(point)
        node = by_key.get((point + 1, branch))
        parent_answer = _answer(parent["content"]) if parent else None
        node_answer = _answer(node["content"]) if node else None
        plan_rows.append(
            {
                "run_id": item["run_id"],
                "benchmark": item["benchmark"],
                "problem_id": item["problem_id"],
                "harness_id": harness_id,
                "point": point,
                "branch": branch,
                "slot": plan.slot if plan else None,
                "planner_call_index": call["index"],
                "planner_label": entry["label"],
                "planner_recovered": entry["recovered"],
                "planner_call_shared": entry["shared"],
                "planner_tokens": (call.get("usage") or {}).get("completion_tokens"),
                "planner_finish_reason": (call.get("response") or {}).get("finish_reason"),
                "valid": bool(plan and plan.valid),
                "error": plan.error if plan else "no usable submit_plans call",
                "title": plan.title if plan else None,
                "title_derived": plan.title_derived if plan else None,
                "brief": plan.brief if plan else None,
                "brief_truncated": plan.brief_truncated if plan else None,
                "show_current_solution": plan.show_current_solution if plan else None,
                "success_probability": plan.success_probability if plan else None,
                "brief_contains_parent_answer": bool(
                    plan and plan.brief and parent_answer and parent_answer in plan.brief
                ),
                "parent_call_index": parent["call_index"] if parent else None,
                "parent_extracted_answer": parent_answer,
                "parent_judge_correct": labels.get(parent["call_index"], {}).get("correct")
                if parent
                else None,
                "parent_verdicts_mean_probability": _mean(
                    probabilities_on.get(parent["call_index"], []) if parent else []
                ),
                "executed": node is not None,
                "node_call_index": node["call_index"] if node else None,
                "node_extracted_answer": node_answer,
                "node_judge_correct": labels.get(node["call_index"], {}).get("correct")
                if node
                else None,
                "answer_changed": node_answer != parent_answer if node else None,
                "promoted": (point + 1, branch) in promoted,
            }
        )

    usage = solve.get("usage") or {}
    node_labels = [row["judge_correct"] for row in nodes if row["judge_correct"] is not None]
    answers = {_answer(c["content"]) for c in candidates}
    spine_correct = [labels.get(index, {}).get("correct") for index in sorted(spine)]
    tokens_by_role: Counter[str] = Counter()
    for call in solve.get("calls", []):
        tokens_by_role[call["role"]] += (call.get("usage") or {}).get("completion_tokens") or 0
    tree = {
        **shared,
        "gold_answer": item.get("golden_answer"),
        "harness_id": harness_id,
        "status": solve.get("status"),
        "candidates": len(candidates),
        "verdicts": len(verdicts),
        "spine_call_indices": sorted(spine),
        "generated_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "tokens_by_role": dict(sorted(tokens_by_role.items())),
        "node_judge_status": (judged or {}).get("judge_status"),
        "spine_correct": spine_correct,
        "distinct_answers": len(answers),
        "single_answer": len(answers) == 1,
        "label_mix": (
            None
            if not node_labels
            else "mixed"
            if 0 < sum(node_labels) < len(node_labels)
            else "all_correct"
            if all(node_labels)
            else "all_incorrect"
        ),
        "any_correct": any(node_labels) if node_labels else None,
        "final_correct": spine_correct[-1] if spine_correct else None,
        "failures": dict(
            Counter(t["source"] for t in solve.get("transitions", []) if t["action"] == "failed")
        ),
        **{key: value for key, value in prepared.items() if key.startswith("qwen36_")},
    }
    return tree, nodes, verifications, plan_rows


def _check_harness_sources(items: list[dict[str, Any]], allow_drift: bool) -> None:
    """Plans are re-parsed with the live harness module, which must be the one that ran."""

    from value_as_tool.harnesses import resolve_harness

    for entrypoint, scheduled in {
        (item.get("harness_entrypoint"), item.get("harness_source_sha256"))
        for item in items
        if item.get("harness_id") in REPLAN_HARNESSES
    }:
        live = resolve_harness(entrypoint).source_sha256
        if live != scheduled and not allow_drift:
            raise SystemExit(
                f"{entrypoint} source changed since the collection ran ({scheduled[:12]} → "
                f"{live[:12]}); export from the launch snapshot or pass --allow-harness-drift"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-reasoning", action="store_true")
    parser.add_argument(
        "--allow-harness-drift",
        action="store_true",
        help="re-parse plans even if the harness source differs from the scheduled one",
    )
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    schedule = json.loads((root / "schedule.json").read_text(encoding="utf-8"))
    _check_harness_sources(schedule["items"], args.allow_harness_drift)
    prepared: dict[tuple[str, str], dict[str, Any]] = {}
    for path in (root / "prepared" / "benchmarks").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            prepared[(path.stem, str(row.get("item_id")))] = row
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with (
        (output / "trees.jsonl").open("w", encoding="utf-8") as trees,
        (output / "nodes.jsonl").open("w", encoding="utf-8") as nodes,
        (output / "verifications.jsonl").open("w", encoding="utf-8") as verifications,
        (output / "plans.jsonl").open("w", encoding="utf-8") as plans,
    ):
        for item in schedule["items"]:
            solve = _load(root / "solve" / "runs" / item["run_id"] / "result.json")
            if solve is None:
                counts["unsolved"] += 1
                continue
            judged = _load(root / "node_judge" / "runs" / item["run_id"] / "result.json")
            counts["trees"] += 1
            counts["unjudged_trees"] += judged is None
            tree, node_rows, verdict_rows, plan_rows = export_tree(
                item,
                solve,
                judged,
                prepared.get((item["benchmark"], item["problem_id"]), {}),
                include_reasoning=args.include_reasoning,
            )
            trees.write(json.dumps(tree, ensure_ascii=False) + "\n")
            for row in node_rows:
                nodes.write(json.dumps(row, ensure_ascii=False) + "\n")
            for row in verdict_rows:
                verifications.write(json.dumps(row, ensure_ascii=False) + "\n")
            for row in plan_rows:
                plans.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["nodes"] += len(node_rows)
            counts["verifications"] += len(verdict_rows)
            counts["plans"] += len(plan_rows)
    manifest = {
        "artifact_root": str(root),
        "config_fingerprint": schedule.get("config_fingerprint"),
        "include_reasoning": args.include_reasoning,
        "counts": dict(sorted(counts.items())),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    sys.exit(main())
