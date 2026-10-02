"""Branched GVR data collection: ten rounds that never stop at "correct".

A main chain ("spine") of Generator → Verifier → Reviser runs ``ROUNDS``
verification points. At every point, ``BRANCHES`` siblings each sample their
own verdict on the spine candidate and then their own next candidate. The
lowest-index sibling that produced a candidate continues the spine; the others
are kept one step deep. The verifier is overconfident, so a "correct" verdict
routes to an independent re-check instead of ending the trajectory. Every
candidate is recorded, keyed by ``(cycle, branch)``, for external judging.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from value_as_tool.budget import BudgetExhausted
from value_as_tool.harnesses.base import (
    HarnessAbort,
    HarnessReplayError,
    HarnessRuntime,
    HarnessSpec,
)
from value_as_tool.orchestrator import (
    GENERATOR_SYSTEM_PROMPT,
    GVR_FORCED_CANDIDATE_REMINDER,
    REVISER_SYSTEM_PROMPT,
)
from value_as_tool.schemas import (
    CandidateRecord,
    Role,
    TrajectoryStatus,
    Verdict,
    VerdictRecord,
)

ROUNDS = 10
BRANCHES = 4
CANDIDATE_CAP = 98_304
VERIFIER_CAP = 49_152
# Held back from each node's first attempt for its in-budget recovery.
CANDIDATE_RECOVERY_RESERVE = 8_192
VERIFIER_RECOVERY_RESERVE = 4_096
WORST_CASE_TOKENS = CANDIDATE_CAP + ROUNDS * BRANCHES * (CANDIDATE_CAP + VERIFIER_CAP)

RECHECK_SYSTEM_PROMPT = """You are the Reviser in a mathematical solution system.
A verifier judged the candidate correct, but verifiers are often overconfident.
Independently re-check every claim, computation, case, and the final answer.
Keep what survives scrutiny and fix anything wrong or unjustified. Return a
complete, self-contained replacement that obeys the original requested answer
format, not a patch or a discussion of edits."""

NO_GAP_ASSESSMENT = "The verifier found no substantive gap."


@dataclass(frozen=True)
class _Outcome:
    verdict: VerdictRecord | None
    candidate: CandidateRecord | None


def _contained(exc: HarnessAbort) -> bool:
    """Node failures with known usage end one branch, never the tree."""

    return not isinstance(exc, HarnessReplayError) and exc.status in {
        TrajectoryStatus.PROTOCOL_ERROR,
        TrajectoryStatus.CONTEXT_EXHAUSTED,
    }


def _first_fatal(group: BaseExceptionGroup) -> BaseException:
    """The error the orchestrator should see for a failed sibling group."""

    leaves: list[BaseException] = []
    pending: list[BaseException] = [group]
    while pending:
        item = pending.pop(0)
        if isinstance(item, BaseExceptionGroup):
            pending.extend(item.exceptions)
        else:
            leaves.append(item)
    for kind in (HarnessAbort, BudgetExhausted):
        for leaf in leaves:
            if isinstance(leaf, kind):
                return leaf
    return leaves[0]


def _revision_messages(
    base_task: str,
    candidate: str,
    verdict: VerdictRecord,
    feedback: str,
) -> tuple[str, Role, list[Mapping[str, str]]]:
    """The next step for one verdict; revise and regenerate match built-in GVR."""

    if verdict.verdict is Verdict.CORRECT:
        return (
            "recheck",
            Role.REVISER,
            [
                {"role": "system", "content": RECHECK_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Original task and required output format:\n{base_task}\n\n"
                        f"Candidate judged correct:\n{candidate}\n\n"
                        f"Verifier assessment:\n{feedback or NO_GAP_ASSESSMENT}\n\n"
                        "Independently re-check it and return a complete replacement "
                        "response that follows the original task."
                    ),
                },
            ],
        )
    if verdict.verdict is Verdict.MINOR_FIX:
        return (
            "revise",
            Role.REVISER,
            [
                {"role": "system", "content": REVISER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Original task and required output format:\n{base_task}\n\n"
                        f"Candidate to replace:\n{candidate}\n\n"
                        f"Verifier feedback:\n{feedback}\n\n"
                        "Return a complete replacement response that follows the original task."
                    ),
                },
            ],
        )
    return (
        "regenerate",
        Role.GENERATOR,
        [
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original task and required output format:\n{base_task}\n\n"
                    f"Rejected candidate:\n{candidate}\n\n"
                    f"Verifier feedback:\n{feedback}\n\n"
                    "Develop a substantially corrected response that follows the original task."
                ),
            },
        ],
    )


class AgentHarness:
    spec = HarnessSpec(
        harness_id="gvr_branched",
        display_name="GVR tree (10 rounds × 4 branches)",
    )

    async def run(self, runtime: HarnessRuntime) -> None:
        if WORST_CASE_TOKENS > runtime.total_generated_tokens:
            runtime.abort(
                TrajectoryStatus.PROTOCOL_ERROR,
                f"branched GVR needs {WORST_CASE_TOKENS} generated tokens in the worst case; "
                f"the trajectory budget is {runtime.total_generated_tokens}",
            )
        request = runtime.request
        base_task = request.solver_prompt or f"Problem:\n{request.problem}"
        try:
            spine = await self._candidate(
                runtime,
                label="gen",
                cycle=1,
                role=Role.GENERATOR,
                messages=[
                    {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": base_task},
                ],
                branch=None,
                parent_call_index=None,
                first_seed=request.seed,
            )
        except HarnessAbort as exc:
            if not _contained(exc):
                raise
            runtime.abort(exc.status, str(exc))
        await runtime.transition(1, "generator", "candidate", "verifier")

        dead_at: int | None = None
        try:
            async with asyncio.TaskGroup() as group:
                for point in range(1, ROUNDS + 1):
                    siblings = [
                        group.create_task(self._sibling(runtime, base_task, point, branch, spine))
                        for branch in range(BRANCHES)
                    ]
                    if point == ROUNDS:
                        break
                    promoted: CandidateRecord | None = None
                    # Index order, not completion order, keeps promotion deterministic.
                    for task in siblings:
                        promoted = (await task).candidate
                        if promoted is not None:
                            break
                    if promoted is None:
                        dead_at = point
                        break
                    await runtime.transition(
                        point + 1, f"branch_{promoted.branch}", "promote", "spine"
                    )
                    spine = promoted
        except BaseExceptionGroup as group_error:
            raise _first_fatal(group_error) from group_error
        if dead_at is not None:
            await runtime.transition(dead_at, "spine", "no_candidate", "final_output")
            await runtime.finish(spine.content, status=TrajectoryStatus.PROTOCOL_ERROR)
            return
        await runtime.transition(ROUNDS, "spine", "round_limit", "final_output")
        await runtime.finish(spine.content, status=TrajectoryStatus.CYCLE_LIMIT)

    async def _sibling(
        self,
        runtime: HarnessRuntime,
        base_task: str,
        point: int,
        branch: int,
        spine: CandidateRecord,
    ) -> _Outcome:
        """One branch: a fresh verdict on the spine candidate, then one revision."""

        prefix = f"r{point:02d}.b{branch}"
        try:
            verdict = await runtime.verify(
                cycle=point,
                label=f"{prefix}.verify",
                candidate=spine.content,
                cap=VERIFIER_CAP,
                recovery_reserve=VERIFIER_RECOVERY_RESERVE,
                branch=branch,
                parent_call_index=spine.call_index,
            )
        except HarnessAbort as exc:
            if not _contained(exc):
                raise
            await runtime.transition(
                point, "verifier", "failed", "leaf", detail=f"branch_{branch}: {exc}"[:240]
            )
            return _Outcome(None, None)
        await runtime.add_verdict(verdict)
        mode, role, messages = _revision_messages(
            base_task, spine.content, verdict, runtime.feedback_for(verdict)
        )
        await runtime.transition(
            point, "verifier", verdict.verdict.value, mode, detail=f"branch_{branch}"
        )
        try:
            candidate = await self._candidate(
                runtime,
                label=f"{prefix}.{mode}",
                cycle=point + 1,
                role=role,
                messages=messages,
                branch=branch,
                parent_call_index=spine.call_index,
            )
        except HarnessAbort as exc:
            if not _contained(exc):
                raise
            await runtime.transition(
                point + 1, mode, "failed", "leaf", detail=f"branch_{branch}: {exc}"[:240]
            )
            return _Outcome(verdict, None)
        return _Outcome(verdict, candidate)

    async def _candidate(
        self,
        runtime: HarnessRuntime,
        *,
        label: str,
        cycle: int,
        role: Role,
        messages: Sequence[Mapping[str, str]],
        branch: int | None,
        parent_call_index: int | None,
        first_seed: int | None = None,
    ) -> CandidateRecord:
        """A candidate, with the built-in protocol's forced final-synthesis recovery."""

        first_label = f"{label}.t0"
        completion, call_index = await runtime.call(
            role=role,
            cycle=cycle,
            label=first_label,
            messages=messages,
            cap=CANDIDATE_CAP - CANDIDATE_RECOVERY_RESERVE,
            keep=0,
            seed=runtime.stable_seed(first_label) if first_seed is None else first_seed,
        )
        content = (completion.message.content or "").strip()
        if not content or completion.message.tool_calls:
            remaining = CANDIDATE_CAP - completion.usage.completion_tokens
            if remaining < runtime.minimum_call_tokens:
                raise HarnessAbort(
                    TrajectoryStatus.PROTOCOL_ERROR,
                    f"{label} returned no candidate and left no budget for recovery",
                )
            recovery_label = f"{label}.t1"
            completion, call_index = await runtime.call(
                role=role,
                cycle=cycle,
                label=recovery_label,
                messages=[
                    *messages,
                    completion.message.to_api_dict(),
                    {"role": "user", "content": GVR_FORCED_CANDIDATE_REMINDER},
                ],
                cap=remaining,
                keep=0,
                seed=runtime.stable_seed(recovery_label),
                force_no_thinking=True,
                use_sampling=False,
            )
            content = (completion.message.content or "").strip()
            if not content or completion.message.tool_calls:
                raise HarnessAbort(
                    TrajectoryStatus.PROTOCOL_ERROR,
                    f"{label} forced final synthesis returned no candidate",
                )
        # The reasoning stays in the call record; omitting it here keeps the
        # checkpoint that is rewritten after every call small.
        return await runtime.add_candidate(
            cycle=cycle,
            role=role,
            content=content,
            reasoning=None,
            call_index=call_index,
            branch=branch,
            parent_call_index=parent_call_index,
        )


__all__ = [
    "BRANCHES",
    "CANDIDATE_CAP",
    "RECHECK_SYSTEM_PROMPT",
    "ROUNDS",
    "VERIFIER_CAP",
    "WORST_CASE_TOKENS",
    "AgentHarness",
]
