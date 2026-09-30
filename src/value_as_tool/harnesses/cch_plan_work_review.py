"""Plan/work/review proof harness inspired by Claude Code Harness v5.15.0.

The reference project is a software-delivery workflow, not a model runtime.
This adaptation keeps its central separation of planning, implementation,
fresh-context review, and bounded retakes while using only the trusted proof
runtime and the configured child model.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from value_as_tool.harnesses.base import HarnessRuntime, HarnessSpec
from value_as_tool.schemas import (
    RATIONALE_MAX_CHARS,
    ChatCompletion,
    Role,
    TrajectoryStatus,
    Verdict,
    VerdictRecord,
)

CCH_REFERENCE_COMMIT = "2b2b74805321089bd9b660a1064fa97556299703"

PLAN_CAP = 16_384
INITIAL_PROOF_CAP = 65_536
REVIEW_CAP = 24_576
FIRST_REPAIR_CAP = 40_960
SECOND_REPAIR_CAP = 32_768

SUBMIT_PLAN_TOOL: Mapping[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": "Submit the proof plan and its verification obligations.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "outline": {"type": "string", "maxLength": 8_000},
                "obligations": {
                    "type": "string",
                    "maxLength": 6_000,
                    "description": "A concise newline-separated list of proof obligations.",
                },
                "risks": {
                    "type": "string",
                    "maxLength": 4_000,
                    "description": "A concise newline-separated list of logical risks.",
                },
            },
            "required": ["outline", "obligations", "risks"],
        },
    },
}

SUBMIT_REVIEW_TOOL: Mapping[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_review",
        "description": "Submit an independent proof-review gate decision.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["approve", "request_changes"],
                },
                "severity": {
                    "type": "string",
                    "enum": ["none", "minor", "major"],
                },
                "success_probability": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "summary": {"type": "string", "maxLength": RATIONALE_MAX_CHARS},
                "findings": {
                    "type": "string",
                    "maxLength": 1_800,
                    "description": (
                        "At most three newline-separated actionable findings, or an "
                        "empty string when the proof is approved."
                    ),
                },
            },
            "required": [
                "decision",
                "severity",
                "success_probability",
                "summary",
                "findings",
            ],
        },
    },
}

PLANNER_SYSTEM_PROMPT = """You are the planning role in a rigorous mathematical
proof workflow. Analyze the problem before anyone writes the final proof. Use
exactly one submit_plan tool call. Give a viable outline, list every result that
must be justified, and identify the most likely logical failure points. Do not
claim the problem is solved and do not mention this orchestration protocol.
The obligations and risks fields are plain newline-separated strings, never
arrays, JSON strings, or XML tags. Keep the outline concise and submit the tool
before the response limit; obligations and risks may be empty if space is low."""

WORKER_SYSTEM_PROMPT = """You are the proof-writing role in a rigorous
Plan-Work-Review workflow. Produce a complete self-contained proof of the
original problem. Use the supplied plan as guidance, but independently correct
it if needed. Check all domains, equality cases, quantifiers, and imported
lemmas. Return only the polished proof, with no discussion of the workflow."""

REVIEWER_SYSTEM_PROMPT = """You are a fresh-context independent reviewer of a
mathematical proof. Verify every substantive claim against the original
problem. Use exactly one submit_review tool call. Approve only a complete,
rigorous proof. Otherwise report at most three concrete, actionable findings;
major means the central approach needs reconstruction, while minor means a
local repair suffices. The probability is an uncalibrated estimate of receiving
full credit. Do not solve the problem or introduce facts unavailable to the
candidate. The findings field is one plain newline-separated string, never an
array, JSON string, or XML tags; use an empty string when approving."""

REPAIR_SYSTEM_PROMPT = """You are the proof worker responding to an independent
review. Return a complete, self-contained replacement proof, not a patch or an
account of edits. Address every review finding while preserving correct parts,
and independently verify the repaired argument. Return only the polished
proof."""


def _forced_tool(name: str) -> Mapping[str, Any]:
    return {"type": "function", "function": {"name": name}}


def _tool_arguments(completion: ChatCompletion, name: str) -> Mapping[str, Any]:
    calls = completion.message.tool_calls
    if len(calls) != 1 or calls[0].name != name:
        raise ValueError(f"expected exactly one {name} tool call")
    try:
        arguments = calls[0].parsed_arguments()
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name} arguments: {exc}") from exc
    return arguments


def _bounded_text(
    value: Any,
    *,
    length: int,
    field: str,
    allow_empty: bool = False,
) -> str:
    """Normalize bounded text while accepting older array-shaped checkpoints."""

    if isinstance(value, str) and value.lstrip().startswith("["):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list):
            value = decoded
    if isinstance(value, list):
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError(f"{field} entries must be nonempty strings")
        text = "\n".join(f"- {item.strip()}" for item in value)
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise ValueError(f"{field} must be text")
    if not text and not allow_empty:
        raise ValueError(f"{field} must be nonempty")
    return text[:length]


def _parse_plan(completion: ChatCompletion) -> str:
    arguments = dict(_tool_arguments(completion, "submit_plan"))
    # SGLang's Qwen parser has also been observed to fold a missing string
    # field into the preceding value using an XML-like marker. Recover only
    # this unambiguous, lossless form.
    raw_obligations = arguments.get("obligations")
    if "risks" not in arguments and isinstance(raw_obligations, str):
        before, marker, after = raw_obligations.partition("<risks>")
        if marker:
            arguments["obligations"] = before.rstrip().removesuffix(",").rstrip()
            arguments["risks"] = after.strip()
    outline = arguments.get("outline")
    if not isinstance(outline, str) or not outline.strip():
        raise ValueError("plan outline must be nonempty")
    obligations = _bounded_text(
        arguments.get("obligations", ""),
        length=6_000,
        field="obligations",
        allow_empty=True,
    )
    risks = _bounded_text(
        arguments.get("risks", ""),
        length=4_000,
        field="risks",
        allow_empty=True,
    )
    sections = [f"Outline:\n{outline.strip()[:8000]}"]
    if obligations:
        sections.append("Proof obligations:\n" + obligations)
    if risks:
        sections.append("Risks to check:\n" + risks)
    return "\n\n".join(sections)


def _parse_review(completion: ChatCompletion, *, cycle: int, call_index: int) -> VerdictRecord:
    arguments = dict(_tool_arguments(completion, "submit_review"))
    decision = arguments.get("decision")
    severity = arguments.get("severity")
    if isinstance(decision, str):
        decision = decision.strip().casefold()
    if isinstance(severity, str):
        severity = severity.strip().casefold()
    if decision not in {"approve", "request_changes"}:
        raise ValueError("invalid review decision")
    if severity not in {"none", "minor", "major"}:
        raise ValueError("invalid review severity")
    if (decision == "approve") != (severity == "none"):
        raise ValueError("approve requires severity=none and changes require minor/major")
    raw_probability = arguments.get("success_probability")
    if isinstance(raw_probability, bool):
        raise ValueError("review probability must be numeric")
    try:
        probability = float(raw_probability)
    except (TypeError, ValueError) as exc:
        raise ValueError("review probability must be numeric") from exc
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("review probability must be finite and between zero and one")
    summary = arguments.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("review summary must be nonempty")
    summary = summary.strip()[:RATIONALE_MAX_CHARS]
    findings = _bounded_text(
        arguments.get("findings", ""),
        length=1_800,
        field="findings",
        allow_empty=True,
    )
    if findings.casefold().strip(" .") in {
        "none",
        "no findings",
        "no issues",
        "n/a",
    }:
        findings = ""
    if decision == "request_changes" and not findings:
        findings = summary
    if decision == "approve":
        findings = ""
    verdict = (
        Verdict.CORRECT
        if decision == "approve"
        else Verdict.MINOR_FIX
        if severity == "minor"
        else Verdict.CRITICAL_FLAW
    )
    return VerdictRecord(
        cycle=cycle,
        verdict=verdict,
        critique=findings[:600],
        fault_category=f"cch_review_{severity}",
        candidate_excerpt="",
        call_index=call_index,
        success_probability=probability,
        rationale=summary,
    )


def _proof_content(completion: ChatCompletion, label: str) -> str:
    content = completion.message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{label} returned no visible proof")
    if completion.message.tool_calls:
        raise ValueError(f"{label} returned an unexpected tool call")
    return content.strip()


def _review_text(review: VerdictRecord) -> str:
    pieces = [
        f"Decision: {review.verdict.value}",
        f"Estimated full-credit probability: {review.success_probability:.6f}",
        f"Summary: {review.rationale}",
    ]
    if review.critique:
        pieces.append(f"Required changes:\n{review.critique}")
    return "\n".join(pieces)


class AgentHarness:
    spec = HarnessSpec(
        harness_id="cch_plan_work_review",
        display_name="CCH Plan/Work/Review",
    )

    async def run(self, runtime: HarnessRuntime) -> None:
        problem = runtime.request.solver_prompt or runtime.request.problem
        stage_tokens = runtime.cch_stage_tokens
        stage_caps = (
            (stage_tokens,) * 7
            if stage_tokens is not None
            else (
                PLAN_CAP,
                INITIAL_PROOF_CAP,
                REVIEW_CAP,
                FIRST_REPAIR_CAP,
                REVIEW_CAP,
                SECOND_REPAIR_CAP,
                REVIEW_CAP,
            )
        )
        stage_keeps = tuple(sum(stage_caps[index + 1 :]) for index in range(7))
        try:
            plan_completion, _ = await runtime.call(
                role=Role.PLANNER,
                cycle=0,
                label="cch.plan",
                messages=(
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": problem},
                ),
                cap=stage_caps[0],
                keep=stage_keeps[0],
                seed=runtime.stable_seed("cch.plan"),
                tools=(SUBMIT_PLAN_TOOL,),
                tool_choice=_forced_tool("submit_plan"),
                parallel_tool_calls=False,
            )
            plan = _parse_plan(plan_completion)
        except ValueError as exc:
            runtime.abort(TrajectoryStatus.PROTOCOL_ERROR, str(exc))

        draft_completion, draft_index = await runtime.call(
            role=Role.WORKER,
            cycle=0,
            label="cch.work.0",
            messages=(
                {"role": "system", "content": WORKER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Problem:\n{problem}\n\nApproved plan:\n{plan}",
                },
            ),
            cap=stage_caps[1],
            keep=stage_keeps[1],
            seed=runtime.stable_seed("cch.work.0"),
        )
        try:
            proof = _proof_content(draft_completion, "initial proof worker")
        except ValueError as exc:
            runtime.abort(TrajectoryStatus.PROTOCOL_ERROR, str(exc))
        await runtime.add_candidate(
            cycle=0,
            role=Role.WORKER,
            content=proof,
            reasoning=draft_completion.message.reasoning,
            call_index=draft_index,
        )
        await runtime.transition(0, "plan", "work", "review")

        for cycle in range(3):
            review_stage = 2 + 2 * cycle
            review_completion, review_index = await runtime.call(
                role=Role.REVIEWER,
                cycle=cycle,
                label=f"cch.review.{cycle}",
                messages=(
                    {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Problem:\n{problem}\n\nCandidate proof:\n{proof}",
                    },
                ),
                cap=stage_caps[review_stage],
                keep=stage_keeps[review_stage],
                seed=runtime.stable_seed(f"cch.review.{cycle}"),
                tools=(SUBMIT_REVIEW_TOOL,),
                tool_choice=_forced_tool("submit_review"),
                parallel_tool_calls=False,
            )
            try:
                review = _parse_review(
                    review_completion,
                    cycle=cycle,
                    call_index=review_index,
                )
            except ValueError as exc:
                runtime.abort(TrajectoryStatus.PROTOCOL_ERROR, str(exc))
            await runtime.add_verdict(review)
            if review.verdict is Verdict.CORRECT:
                await runtime.transition(cycle, "review", "approve", "final_output")
                await runtime.finish(proof, status=TrajectoryStatus.ACCEPTED)
                return
            if cycle == 2:
                await runtime.transition(
                    cycle,
                    "review",
                    "retake_limit",
                    "final_output",
                    detail=review.fault_category,
                )
                await runtime.finish(proof, status=TrajectoryStatus.CYCLE_LIMIT)
                return

            repair_cycle = cycle + 1
            repair_completion, repair_index = await runtime.call(
                role=Role.WORKER,
                cycle=repair_cycle,
                label=f"cch.work.{repair_cycle}",
                messages=(
                    {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Problem:\n{problem}\n\nPlan:\n{plan}\n\n"
                            f"Current proof:\n{proof}\n\nIndependent review:\n"
                            f"{_review_text(review)}"
                        ),
                    },
                ),
                cap=stage_caps[review_stage + 1],
                keep=stage_keeps[review_stage + 1],
                seed=runtime.stable_seed(f"cch.work.{repair_cycle}"),
            )
            try:
                proof = _proof_content(repair_completion, f"repair worker {repair_cycle}")
            except ValueError as exc:
                runtime.abort(TrajectoryStatus.PROTOCOL_ERROR, str(exc))
            await runtime.add_candidate(
                cycle=repair_cycle,
                role=Role.WORKER,
                content=proof,
                reasoning=repair_completion.message.reasoning,
                call_index=repair_index,
            )
            await runtime.transition(
                repair_cycle,
                "review",
                "retake",
                "review",
                detail=review.fault_category,
            )


__all__ = ["AgentHarness", "CCH_REFERENCE_COMMIT"]
