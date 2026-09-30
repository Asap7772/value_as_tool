"""Audit generated-token limits and completion coverage for a saved experiment.

The optional context check retokenizes recorded requests with the pinned solver
template. It distinguishes a smaller harness allowance from the native context
limit, including on calls that finished normally.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from value_as_tool.harnesses.cch_plan_work_review import SUBMIT_PLAN_TOOL, SUBMIT_REVIEW_TOOL
from value_as_tool.orchestrator import (
    QUERY_SUCCESS_PROBABILITY_TOOL,
    SPAWN_SUBAGENTS_TOOL,
    SUBMIT_PROBABILITY_TOOL,
    SUBMIT_RATIONALE_SCORE_TOOL,
    SUBMIT_RATIONALE_SCORE_VERDICT_TOOL,
    SUBMIT_VERDICT_TOOL,
)
from value_as_tool.pipeline import _load_schedule, _model_entry, load_context
from value_as_tool.schemas import AssistantMessage
from value_as_tool.storage import atomic_write_json, canonical_json
from value_as_tool.tokenization import HuggingFaceTokenCounter

_RUNTIME_TOOL_SCHEMAS = {
    canonical_json(tool): tool
    for tool in (
        QUERY_SUCCESS_PROBABILITY_TOOL,
        SPAWN_SUBAGENTS_TOOL,
        SUBMIT_PROBABILITY_TOOL,
        SUBMIT_RATIONALE_SCORE_TOOL,
        SUBMIT_RATIONALE_SCORE_VERDICT_TOOL,
        SUBMIT_VERDICT_TOOL,
        SUBMIT_PLAN_TOOL,
        SUBMIT_REVIEW_TOOL,
    )
}


def _restore_runtime_tools(
    tools: Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]] | None:
    """Recover schema key order lost when results are saved with sort_keys=True.

    Chat templates can serialize nested schema objects in insertion order, so
    counting the sorted artifact can differ from the original request. Use the
    runtime source's order only for a full schema match, preserving array order
    and refusing unknown schemas whose original key order cannot be recovered.
    """
    if tools is None:
        return None
    restored = []
    for index, tool in enumerate(tools):
        original = _RUNTIME_TOOL_SCHEMAS.get(canonical_json(tool))
        if original is None:
            raise ValueError(f"recorded tool {index} does not match a known runtime schema")
        restored.append(deepcopy(original))
    return restored


def _restore_runtime_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Restore the text runtime's message constructors without changing values.

    Qwen's template rejects string tool arguments, which makes the runtime token
    counter use its JSON fallback. That serialization also depends on message
    and nested tool-call key order. AssistantMessage supplies the exact runtime
    constructor, retaining argument strings verbatim; other text messages have
    fixed field order in the harness builders. Reject shapes we cannot recover.
    """
    restored = []
    for index, message in enumerate(messages):
        try:
            role = message.get("role")
            if role == "assistant":
                original = AssistantMessage.from_api(message).to_api_dict()
            else:
                fields = {
                    "system": ("role", "content"),
                    "user": ("role", "content"),
                    "tool": ("role", "tool_call_id", "name", "content"),
                }[role]
                original = {field: message[field] for field in fields}
                if not all(isinstance(value, str) for value in original.values()):
                    raise ValueError("expected text message fields")
            if canonical_json(message) != canonical_json(original):
                raise ValueError("message constructor would change recorded values")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"recorded message {index} does not match a known runtime shape"
            ) from exc
        restored.append(original)
    return restored


def audit(
    config: str,
    *,
    shards: tuple[int, ...] = (),
    shard_count: int = 1,
    check_context: bool = False,
) -> dict[str, Any]:
    context = load_context(config)
    schedule = _load_schedule(context)
    if shard_count <= 0 or any(not 0 <= shard < shard_count for shard in shards):
        raise ValueError("invalid shard selection")
    selected = [
        item for item in schedule if not shards or item.ordinal % shard_count in shards
    ]
    counter = (
        HuggingFaceTokenCounter(
            _model_entry(context, "solver")["path"],
            enable_thinking=context.config.sampling.enable_thinking,
        )
        if check_context
        else None
    )
    summary: dict[str, Counter[str]] = defaultdict(Counter)
    solve_statuses: Counter[str] = Counter()
    judge_statuses: Counter[str] = Counter()
    pending: list[dict[str, str]] = []
    issues: list[dict[str, Any]] = []
    limited: list[dict[str, Any]] = []
    for item in selected:
        method = item.harness_id or str(item.condition)
        counts = summary[f"{item.benchmark}/{method}"]
        counts["scheduled"] += 1
        result_path = context.artifact_root / "solve" / "runs" / item.run_id / "result.json"
        judge_path = context.artifact_root / "judge" / "runs" / item.run_id / "result.json"
        if not result_path.is_file():
            pending.append({"run_id": item.run_id, "stage": "solve"})
            continue
        result = json.loads(result_path.read_text())
        counts["solved"] += 1
        status = result["status"]
        solve_statuses[status] += 1
        if status not in {
            "completed", "accepted", "cycle_limit", "protocol_error",
            "context_exhausted", "budget_exhausted",
        }:
            issues.append({"run_id": item.run_id, "issue": "solve_failure", "status": status})
        usage = result.get("usage", {})
        generated = usage.get("completion_tokens", 0)
        counts["generated_tokens"] += generated
        counts["reasoning_tokens"] += usage.get("reasoning_tokens") or 0
        counts["max_trajectory_tokens"] = max(counts["max_trajectory_tokens"], generated)
        if generated >= context.config.budget.generated_tokens:
            counts["shared_budget_reached"] += 1
            issues.append({"run_id": item.run_id, "issue": "shared_budget_reached"})
        length_calls = []
        for call in result.get("calls", []):
            counts["calls"] += 1
            response = call.get("response") or {}
            finish = response.get("finish_reason", "missing")
            counts[f"finish_{finish}"] += 1
            detail = {
                "run_id": item.run_id,
                "index": call["index"],
                "role": call["role"],
                "label": call["label"],
                "max_tokens": call["max_tokens"],
            }
            if call.get("error") or not response:
                issues.append({**detail, "issue": "call_error", "error": call.get("error")})
            if counter is not None:
                try:
                    tools = _restore_runtime_tools(call.get("tools"))
                except ValueError as exc:
                    counts["unrecognized_tool_schema"] += 1
                    issues.append(
                        {**detail, "issue": "unrecognized_tool_schema", "error": str(exc)}
                    )
                else:
                    try:
                        messages = _restore_runtime_messages(call["messages"])
                    except ValueError as exc:
                        counts["unrecognized_message_schema"] += 1
                        issues.append(
                            {**detail, "issue": "unrecognized_message_schema", "error": str(exc)}
                        )
                    else:
                        prompt_tokens = counter.count_messages(messages, tools)
                        context_cap = max(
                            0,
                            context.config.budget.context_tokens
                            - context.config.budget.context_headroom_tokens
                            - prompt_tokens,
                        )
                        detail.update(prompt_tokens=prompt_tokens, context_cap=context_cap)
                        if call["max_tokens"] != context_cap:
                            issue = (
                                "harness_cap_below_context"
                                if call["max_tokens"] < context_cap
                                else "request_exceeds_context"
                            )
                            counts[issue] += 1
                            issues.append({**detail, "issue": issue})
            if finish == "length":
                length_calls.append(detail)
        if length_calls:
            counts["trajectories_with_length"] += 1
            limited.append(
                {
                    "run_id": item.run_id,
                    "method": method,
                    "benchmark": item.benchmark,
                    "status": status,
                    "calls": length_calls,
                }
            )
        if status == "context_exhausted":
            counts["context_exhausted"] += 1
        if status == "budget_exhausted":
            counts["budget_exhausted"] += 1
            calls = result.get("calls", [])
            final_length = bool(
                calls
                and length_calls
                and length_calls[-1]["index"] == calls[-1]["index"]
                and result.get("error") in {
                    "direct completion reached its generation limit",
                    "value solver completion reached its generation limit",
                }
            )
            if not final_length:
                issues.append(
                    {
                        "run_id": item.run_id,
                        "issue": "harness_budget_exhausted",
                        "error": result.get("error"),
                    }
                )
        if not judge_path.is_file():
            pending.append({"run_id": item.run_id, "stage": "judge"})
            continue
        judged = json.loads(judge_path.read_text())
        judge_status = judged.get("judge_status", "missing")
        judge_statuses[judge_status] += 1
        counts["judged"] += 1
        if judge_status not in {"completed", "parse_error", "solve_failed"}:
            issues.append(
                {"run_id": item.run_id, "issue": "judge_failure", "status": judge_status}
            )
        if (judged.get("judge_result") or {}).get("context_recovery"):
            counts["judge_context_recovery"] += 1
    return {
        "config_fingerprint": context.config_fingerprint,
        "schedule_fingerprint": schedule.fingerprint,
        "model": context.config.models.solver.name,
        "seeds": list(context.config.evaluation.seeds),
        "selected": len(selected),
        "shards": list(shards),
        "shard_count": shard_count,
        "context_checked": check_context,
        "thinking_enabled": context.config.sampling.enable_thinking,
        "forced_thinking_cutoff": context.config.sampling.thinking_content_reserve_tokens != 0,
        "ready": bool(selected) and not pending and not issues,
        "solve_statuses": dict(solve_statuses),
        "judge_statuses": dict(judge_statuses),
        "summary": dict(sorted(summary.items())),
        "pending": pending,
        "issues": issues,
        "length_affected_runs": limited,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-index", action="append", type=int, default=[])
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--check-context", action="store_true")
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.config,
        shards=tuple(args.shard_index),
        shard_count=args.shard_count,
        check_context=args.check_context,
    )
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {key: result[key] for key in ("selected", "ready", "solve_statuses", "judge_statuses")}
        )
    )
    if args.require_ready and not result["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
