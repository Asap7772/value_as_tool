"""Branched GVR data collection in which the model replans at every branching point.

The tree has the shape of ``gvr_branched``: a first candidate, then ``ROUNDS``
verification points with ``BRANCHES`` branches each, every candidate recorded
for external judging. No verdict is mapped to a fixed revision. At each point
four rationale+score verifier calls assess the spine candidate; a planner then
writes, for each branch, a free-form brief for an executor and decides whether
the executor sees the current candidate. ``JointPlanHarness`` asks one planner
call for four materially different plans; ``IndependentPlanHarness`` asks four
planner calls, given the same input, for one plan each. The executor sees only
the original task, its brief, and the candidate if the plan shows it. The spine
continues from a uniformly random branch that produced a candidate.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from value_as_tool.harnesses.base import HarnessAbort, HarnessRuntime, HarnessSpec
from value_as_tool.harnesses.gvr_branched import _contained, _first_fatal
from value_as_tool.judging import find_last_boxed_content
from value_as_tool.orchestrator import GENERATOR_SYSTEM_PROMPT, GVR_FORCED_CANDIDATE_REMINDER
from value_as_tool.schemas import (
    CandidateRecord,
    ChatCompletion,
    Role,
    TrajectoryStatus,
    VerdictRecord,
)

ROUNDS = 10
BRANCHES = 4
CANDIDATE_CAP = 98_304
VERIFIER_CAP = 49_152
PLANNER_CAP = 49_152
# Held back from each first attempt for its in-budget recovery.
CANDIDATE_RECOVERY_RESERVE = 8_192
VERIFIER_RECOVERY_RESERVE = 4_096
PLANNER_RECOVERY_RESERVE = 8_192
JOINT_WORST_CASE_TOKENS = CANDIDATE_CAP + ROUNDS * (
    BRANCHES * (VERIFIER_CAP + CANDIDATE_CAP) + PLANNER_CAP
)
INDEPENDENT_WORST_CASE_TOKENS = CANDIDATE_CAP + ROUNDS * BRANCHES * (
    VERIFIER_CAP + PLANNER_CAP + CANDIDATE_CAP
)
TITLE_MAX_CHARS = 120
BRIEF_MAX_CHARS = 8_000
ANSWER_MAX_CHARS = 300
PLAN_FIELDS = ("title", "brief", "show_current_solution", "success_probability")

PLANNER_SYSTEM_PROMPT = """You are the Planner in a mathematical solution system. You receive
the task, the current candidate solution, independent verifier assessments of it, and
the final answers of the earlier candidates on the main line. Verifiers are often
overconfident and frequently judge wrong solutions correct. Decide what the next attempt
should do and write a brief for the executor who will carry it out.
The executor sees only the original task, your brief, and, if you set
show_current_solution to true, the current candidate solution. It sees nothing else (not
the assessments, the earlier answers, or this message) unless you copy it into the brief.
The executor's response becomes the next candidate solution, so the brief must lead to a
complete, self-contained solution of the original task in its requested answer format.
A plan may be anything from a targeted repair of the current candidate to an entirely
independent attempt. For each plan, estimate the probability that the executor's final
answer will be correct. Do not mention this protocol in a brief."""

JOINT_PLANNER_SUFFIX = """Propose exactly four plans, one for each of four executors working in
parallel. Make them materially different from one another, not rewordings of one idea.
Judge each probability on its own; the four need not sum to one. Use exactly one
submit_plans call. Every field is a plain string, boolean, or number; never arrays, JSON
strings, or XML tags."""

INDEPENDENT_PLANNER_SUFFIX = """Propose one plan. Use exactly one submit_plans call. Every field
is a plain string, boolean, or number; never arrays, JSON strings, or XML tags."""

PLANNER_RECOVERY_SUFFIX = """Your response must be a native submit_plans call with every
required field. show_current_solution fields are true or false; success_probability
fields are numbers between 0 and 1. Keep each brief concise and do not solve the task
here."""

EXECUTOR_SYSTEM_PROMPT = """You are solving a mathematical task by carrying out a brief written
by a planner. Follow the brief. If it relies on a mathematical mistake, do not reproduce
the mistake. Return a complete, self-contained response to the original task that obeys
its requested answer format, not a patch, a report on the brief, or a discussion of the
planner."""

UNAVAILABLE_ASSESSMENT = "Unavailable: the verifier returned no usable assessment."


class PlanParseError(HarnessAbort):
    """No usable plan after the in-budget recovery attempt; usage is known."""

    def __init__(self, message: str) -> None:
        super().__init__(TrajectoryStatus.PROTOCOL_ERROR, message)


@dataclass(frozen=True)
class PlanSlot:
    """One plan from a submit_plans call; ``error`` is set when it cannot be executed."""

    slot: int
    title: str | None
    brief: str | None
    show_current_solution: bool | None
    success_probability: float | None
    brief_truncated: bool = False
    title_derived: bool = False
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class _Outcome:
    plan: PlanSlot | None
    candidate: CandidateRecord | None


def plan_tool(slots: int) -> dict[str, Any]:
    """The forced submit_plans tool: flat scalar fields, which qwen3_coder parses reliably."""

    properties: dict[str, Any] = {}
    for slot in range(1, slots + 1):
        properties[f"plan_{slot}_title"] = {"type": "string", "maxLength": TITLE_MAX_CHARS}
        properties[f"plan_{slot}_brief"] = {"type": "string", "maxLength": BRIEF_MAX_CHARS}
        properties[f"plan_{slot}_show_current_solution"] = {"type": "boolean"}
        properties[f"plan_{slot}_success_probability"] = {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        }
    return {
        "type": "function",
        "function": {
            "name": "submit_plans",
            "description": "Submit the executor briefs for the next attempt.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": properties,
                "required": list(properties),
            },
        },
    }


def _bounded_text(value: Any, *, length: int, field: str) -> tuple[str, bool]:
    """Nonempty text, accepting the array shape SGLang's Qwen parser sometimes emits."""

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
    if not text:
        raise ValueError(f"{field} must be nonempty")
    return text[:length], len(text) > length


def _boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str) and value.strip().casefold() in {"true", "false"}:
        return value.strip().casefold() == "true"
    return None


def _probability(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if not isinstance(value, (int, float)):
        return None
    probability = float(value)
    return probability if math.isfinite(probability) and 0 <= probability <= 1 else None


def _unfold(arguments: dict[str, Any], slots: int) -> dict[str, Any]:
    """Recover fields the qwen3_coder parser folded into a neighbouring string value.

    Only a field that is missing is recovered, and only from an explicit
    ``<parameter=name>`` or ``<name>`` marker, so the repair is lossless.
    """

    for name in (f"plan_{slot}_{field}" for slot in range(1, slots + 1) for field in PLAN_FIELDS):
        if name in arguments:
            continue
        for key, value in list(arguments.items()):
            if not isinstance(value, str):
                continue
            for marker in (f"<parameter={name}>", f"<{name}>"):
                before, found, after = value.partition(marker)
                if found:
                    arguments[key] = before.rstrip().removesuffix("</parameter>").rstrip()
                    arguments[name] = (
                        after.strip().removesuffix("</parameter>").removesuffix(f"</{name}>")
                    ).strip()
                    break
            if name in arguments:
                break
    return arguments


def parse_plan_arguments(arguments: Mapping[str, Any], slots: int) -> tuple[PlanSlot, ...]:
    """Every slot of a submit_plans call; invalid slots carry the reason."""

    arguments = _unfold(dict(arguments), slots)
    plans = []
    for slot in range(1, slots + 1):
        prefix = f"plan_{slot}_"
        errors = []
        brief, truncated = None, False
        try:
            brief, truncated = _bounded_text(
                arguments.get(prefix + "brief"), length=BRIEF_MAX_CHARS, field="brief"
            )
        except ValueError as exc:
            errors.append(str(exc))
        raw_title = arguments.get(prefix + "title")
        title = " ".join(raw_title.split())[:TITLE_MAX_CHARS] if isinstance(raw_title, str) else ""
        derived = not title and brief is not None
        if derived:
            title = " ".join(brief.splitlines()[0].split())[:TITLE_MAX_CHARS]
        show = _boolean(arguments.get(prefix + "show_current_solution"))
        if show is None:
            errors.append("show_current_solution must be true or false")
        probability = _probability(arguments.get(prefix + "success_probability"))
        if probability is None:
            errors.append("success_probability must be a number between 0 and 1")
        plans.append(
            PlanSlot(
                slot=slot,
                title=title or None,
                brief=brief,
                show_current_solution=show,
                success_probability=probability,
                brief_truncated=truncated,
                title_derived=derived,
                error="; ".join(errors) or None,
            )
        )
    return tuple(plans)


def parse_plan_completion(completion: ChatCompletion, slots: int) -> tuple[PlanSlot, ...] | None:
    """The plans of exactly one submit_plans call, or None if there is no usable call."""

    calls = completion.message.tool_calls
    if len(calls) != 1 or calls[0].name != "submit_plans":
        return None
    try:
        arguments = calls[0].parsed_arguments()
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(arguments, Mapping):
        return None
    return parse_plan_arguments(arguments, slots)


def final_answer(text: str) -> str:
    answer = find_last_boxed_content(text)
    if not answer:
        return "no \\boxed{} answer"
    return " ".join(answer.split())[:ANSWER_MAX_CHARS]


def history_line(
    attempt: int, title: str, candidate: str, verdicts: Sequence[VerdictRecord | None]
) -> str:
    """One earlier spine candidate as the planner sees it: answer and verifier mean."""

    probabilities = [
        verdict.success_probability
        for verdict in verdicts
        if verdict is not None and verdict.success_probability is not None
    ]
    mean = (
        f"mean verifier probability {sum(probabilities) / len(probabilities):.2f} "
        f"({len(probabilities)} of {len(verdicts)} assessments)"
        if probabilities
        else "no verifier assessment"
    )
    return f"Attempt {attempt} ({title}): {final_answer(candidate)} | {mean}"


def planner_message(
    base_task: str,
    history: Sequence[str],
    attempt: int,
    candidate: str,
    assessments: Sequence[str],
) -> str:
    """The planner input, identical in both variants."""

    lines = [
        "Original task (the executor sees this verbatim):",
        base_task,
        "",
        "Final answers of the candidates so far, oldest first:",
        *history,
        "",
        f"Current candidate solution (attempt {attempt}):",
        candidate,
        "",
        "Independent verifier assessments of the current candidate:",
    ]
    for index, assessment in enumerate(assessments, 1):
        lines += [f"Assessment {index}", assessment]
    return "\n".join(lines)


def assessment_text(runtime: HarnessRuntime, verdict: VerdictRecord | None) -> str:
    """One verifier assessment as the planner sees it."""

    if verdict is None:
        return UNAVAILABLE_ASSESSMENT
    return f"Outcome: {verdict.verdict.value}\n{runtime.feedback_for(verdict)}"


def executor_messages(base_task: str, plan: PlanSlot, candidate: str) -> list[dict[str, str]]:
    """The executor's whole context: the task, the brief, and the candidate if shown."""

    user = (
        f"Original task and required output format:\n{base_task}\n\n"
        f"Brief from the planner:\n{plan.brief}"
    )
    if plan.show_current_solution:
        user += f"\n\nCurrent candidate solution referred to by the brief:\n{candidate}"
    user += "\n\nCarry out the brief and return a complete response to the original task."
    return [
        {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def promotion_order(runtime: HarnessRuntime, point: int) -> tuple[int, ...]:
    """A seeded uniform permutation of the branches.

    The spine continues from the first branch in this order that produced a
    candidate, which is uniform over the successful branches, and promotion
    waits only for the branches ranked ahead of the winner.
    """

    keys = sorted(
        (runtime.stable_seed(f"r{point:02d}.promote.b{branch}"), branch)
        for branch in range(BRANCHES)
    )
    return tuple(branch for _, branch in keys)


def _detail(**values: Any) -> str:
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


async def _candidate(
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


class _ReplanHarness:
    """The shared tree; subclasses choose joint or independent planning."""

    joint: bool
    worst_case_tokens: int

    async def run(self, runtime: HarnessRuntime) -> None:
        if self.worst_case_tokens > runtime.total_generated_tokens:
            runtime.abort(
                TrajectoryStatus.PROTOCOL_ERROR,
                f"{self.spec.harness_id} needs {self.worst_case_tokens} generated tokens in "
                f"the worst case; the trajectory budget is {runtime.total_generated_tokens}",
            )
        request = runtime.request
        base_task = request.solver_prompt or f"Problem:\n{request.problem}"
        try:
            # The same first request as gvr_branched, so c1 is comparable across harnesses.
            spine = await _candidate(
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

        history: list[str] = []
        title = "initial attempt"
        dead_at: int | None = None
        branches_by_point: list[list[asyncio.Task[_Outcome]]] = []
        try:
            async with asyncio.TaskGroup() as group:
                for point in range(1, ROUNDS + 1):
                    panel = [
                        group.create_task(self._assess(runtime, point, branch, spine))
                        for branch in range(BRANCHES)
                    ]
                    verdicts = [await task for task in panel]
                    history.append(history_line(point, title, spine.content, verdicts))
                    message = planner_message(
                        base_task,
                        history,
                        point,
                        spine.content,
                        [assessment_text(runtime, verdict) for verdict in verdicts],
                    )
                    branches = self._branches(group, runtime, base_task, point, spine, message)
                    branches_by_point.append(branches)
                    if point == ROUNDS:
                        break
                    order = promotion_order(runtime, point)
                    winner: tuple[int, int, CandidateRecord, PlanSlot] | None = None
                    for rank, branch in enumerate(order):
                        outcome = await branches[branch]
                        if outcome.candidate is not None and outcome.plan is not None:
                            winner = (rank, branch, outcome.candidate, outcome.plan)
                            break
                    if winner is None:
                        dead_at = point
                        break
                    rank, branch, spine, plan = winner
                    await runtime.transition(
                        point + 1,
                        f"branch_{branch}",
                        "promote",
                        "spine",
                        detail=_detail(order=list(order), rank=rank),
                    )
                    title = f'plan "{plan.title}"'
        except BaseExceptionGroup as group_error:
            raise _first_fatal(group_error) from group_error
        # Every branch has finished, so each point's set of successful branches,
        # and with it the promotion probability, is known exactly.
        for point, branches in enumerate(branches_by_point[: ROUNDS - 1], 1):
            successful = [index for index, task in enumerate(branches) if task.result().candidate]
            if successful:
                await runtime.transition(
                    point + 1,
                    "spine",
                    "promotion_support",
                    "spine",
                    detail=_detail(p=round(1 / len(successful), 6), successful=successful),
                )
        if dead_at is not None:
            await runtime.transition(dead_at, "spine", "no_candidate", "final_output")
            await runtime.finish(spine.content, status=TrajectoryStatus.PROTOCOL_ERROR)
            return
        await runtime.transition(ROUNDS, "spine", "round_limit", "final_output")
        await runtime.finish(spine.content, status=TrajectoryStatus.CYCLE_LIMIT)

    def _branches(
        self,
        group: asyncio.TaskGroup,
        runtime: HarnessRuntime,
        base_task: str,
        point: int,
        spine: CandidateRecord,
        message: str,
    ) -> list[asyncio.Task[_Outcome]]:
        if self.joint:
            plans = group.create_task(
                self._plan(
                    runtime,
                    label=f"r{point:02d}.plan",
                    point=point,
                    message=message,
                    slots=BRANCHES,
                )
            )
            return [
                group.create_task(
                    self._joint_branch(runtime, plans, base_task, point, branch, spine)
                )
                for branch in range(BRANCHES)
            ]
        return [
            group.create_task(
                self._independent_branch(runtime, base_task, point, branch, spine, message)
            )
            for branch in range(BRANCHES)
        ]

    async def _assess(
        self,
        runtime: HarnessRuntime,
        point: int,
        branch: int,
        spine: CandidateRecord,
    ) -> VerdictRecord | None:
        """One rationale+score assessment; a failure only marks it unavailable."""

        try:
            verdict = await runtime.verify(
                cycle=point,
                label=f"r{point:02d}.b{branch}.verify",
                candidate=spine.content,
                cap=VERIFIER_CAP,
                recovery_reserve=VERIFIER_RECOVERY_RESERVE,
                rationale_score=True,
                branch=branch,
                parent_call_index=spine.call_index,
            )
        except HarnessAbort as exc:
            if not _contained(exc):
                raise
            await runtime.transition(
                point, "verifier", "failed", "planner", detail=f"branch_{branch}: {exc}"[:240]
            )
            return None
        await runtime.add_verdict(verdict)
        await runtime.transition(
            point, "verifier", verdict.verdict.value, "planner", detail=f"branch_{branch}"
        )
        return verdict

    async def _plan(
        self,
        runtime: HarnessRuntime,
        *,
        label: str,
        point: int,
        message: str,
        slots: int,
    ) -> tuple[tuple[PlanSlot, ...], bool] | None:
        """Plans from at most two labelled calls, or None when neither is usable.

        The first call samples with thinking. If it yields no valid plan, one
        call with thinking disabled may use what the first left of the cap; it
        still samples, so independent planners that both recover differ.
        """

        system = f"{PLANNER_SYSTEM_PROMPT}\n\n" + (
            JOINT_PLANNER_SUFFIX if slots > 1 else INDEPENDENT_PLANNER_SUFFIX
        )
        tool = plan_tool(slots)
        errors: list[str] = []
        spent = 0
        try:
            for recovery in (False, True):
                cap = PLANNER_CAP - spent if recovery else PLANNER_CAP - PLANNER_RECOVERY_RESERVE
                if cap < runtime.minimum_call_tokens:
                    break
                attempt = f"{label}.recovery" if recovery else label
                completion, _ = await runtime.call(
                    role=Role.PLANNER,
                    cycle=point,
                    label=attempt,
                    messages=[
                        {
                            "role": "system",
                            "content": f"{system}\n\n{PLANNER_RECOVERY_SUFFIX}"
                            if recovery
                            else system,
                        },
                        {"role": "user", "content": message},
                    ],
                    cap=cap,
                    keep=0,
                    seed=runtime.stable_seed(attempt),
                    tools=(tool,),
                    tool_choice={"type": "function", "function": {"name": "submit_plans"}},
                    parallel_tool_calls=False,
                    force_no_thinking=recovery,
                )
                plans = parse_plan_completion(completion, slots)
                if plans is not None and any(plan.valid for plan in plans):
                    return plans, recovery
                errors.append(
                    "planner response reached its generation limit"
                    if completion.finish_reason == "length"
                    else "; ".join(plan.error or "" for plan in plans or ())
                    or "no submit_plans call"
                )
                spent += completion.usage.completion_tokens
            raise PlanParseError(f"{label}: " + ("; ".join(errors) or "no plan"))
        except HarnessAbort as exc:
            if not _contained(exc):
                raise
            await runtime.transition(
                point,
                "planner",
                "failed",
                "leaf" if slots == 1 else "spine",
                detail=f"{label}: {exc}"[:240],
            )
            return None

    async def _joint_branch(
        self,
        runtime: HarnessRuntime,
        plans: asyncio.Task[tuple[tuple[PlanSlot, ...], bool] | None],
        base_task: str,
        point: int,
        branch: int,
        spine: CandidateRecord,
    ) -> _Outcome:
        planned = await plans
        if planned is None:
            return _Outcome(None, None)
        slots, recovered = planned
        plan = slots[branch]
        if not plan.valid:
            await runtime.transition(
                point, "planner", "failed", "leaf", detail=f"branch_{branch}: {plan.error}"[:240]
            )
            return _Outcome(plan, None)
        return await self._execute(runtime, base_task, point, branch, spine, plan, recovered)

    async def _independent_branch(
        self,
        runtime: HarnessRuntime,
        base_task: str,
        point: int,
        branch: int,
        spine: CandidateRecord,
        message: str,
    ) -> _Outcome:
        planned = await self._plan(
            runtime, label=f"r{point:02d}.b{branch}.plan", point=point, message=message, slots=1
        )
        if planned is None:
            return _Outcome(None, None)
        slots, recovered = planned
        return await self._execute(runtime, base_task, point, branch, spine, slots[0], recovered)

    async def _execute(
        self,
        runtime: HarnessRuntime,
        base_task: str,
        point: int,
        branch: int,
        spine: CandidateRecord,
        plan: PlanSlot,
        recovered: bool,
    ) -> _Outcome:
        assert plan.success_probability is not None
        await runtime.transition(
            point,
            "planner",
            "plan",
            f"branch_{branch}",
            detail=_detail(
                p=round(plan.success_probability, 6),
                recovered=recovered,
                show=plan.show_current_solution,
                slot=plan.slot,
            ),
        )
        try:
            candidate = await _candidate(
                runtime,
                label=f"r{point:02d}.b{branch}.exec",
                cycle=point + 1,
                role=Role.WORKER,
                messages=executor_messages(base_task, plan, spine.content),
                branch=branch,
                parent_call_index=spine.call_index,
            )
        except HarnessAbort as exc:
            if not _contained(exc):
                raise
            await runtime.transition(
                point + 1, "executor", "failed", "leaf", detail=f"branch_{branch}: {exc}"[:240]
            )
            return _Outcome(plan, None)
        return _Outcome(plan, candidate)


class JointPlanHarness(_ReplanHarness):
    spec = HarnessSpec(
        harness_id="gvr_replan_joint",
        display_name="GVR tree, one planner proposing 4 plans per point",
    )
    joint = True
    worst_case_tokens = JOINT_WORST_CASE_TOKENS


class IndependentPlanHarness(_ReplanHarness):
    spec = HarnessSpec(
        harness_id="gvr_replan_independent",
        display_name="GVR tree, 4 independent planners per point",
    )
    joint = False
    worst_case_tokens = INDEPENDENT_WORST_CASE_TOKENS


__all__ = [
    "BRANCHES",
    "CANDIDATE_CAP",
    "EXECUTOR_SYSTEM_PROMPT",
    "INDEPENDENT_PLANNER_SUFFIX",
    "INDEPENDENT_WORST_CASE_TOKENS",
    "JOINT_PLANNER_SUFFIX",
    "JOINT_WORST_CASE_TOKENS",
    "PLANNER_CAP",
    "PLANNER_RECOVERY_SUFFIX",
    "PLANNER_SYSTEM_PROMPT",
    "ROUNDS",
    "VERIFIER_CAP",
    "IndependentPlanHarness",
    "JointPlanHarness",
    "PlanSlot",
    "executor_messages",
    "parse_plan_arguments",
    "parse_plan_completion",
    "plan_tool",
    "planner_message",
    "promotion_order",
]
