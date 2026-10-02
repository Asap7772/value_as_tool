"""Replanning GVR trees: shape, planner and executor isolation, promotion, recovery, resume."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from value_as_tool.client import ChatClientError, MissingUsageError
from value_as_tool.harnesses import DATA_COLLECTION_HARNESSES, load_harness, resolve_harness
from value_as_tool.harnesses.gvr_replan import (
    BRANCHES,
    BRIEF_MAX_CHARS,
    CANDIDATE_CAP,
    CANDIDATE_RECOVERY_RESERVE,
    EXECUTOR_SYSTEM_PROMPT,
    INDEPENDENT_WORST_CASE_TOKENS,
    JOINT_WORST_CASE_TOKENS,
    PLANNER_CAP,
    PLANNER_RECOVERY_RESERVE,
    PLANNER_RECOVERY_SUFFIX,
    ROUNDS,
    UNAVAILABLE_ASSESSMENT,
    VERIFIER_CAP,
    VERIFIER_RECOVERY_RESERVE,
    parse_plan_arguments,
    parse_plan_completion,
    promotion_order,
)
from value_as_tool.orchestrator import (
    GENERATOR_SYSTEM_PROMPT,
    SUBMIT_RATIONALE_SCORE_VERDICT_TOOL,
    AletheiaOrchestrator,
    OrchestratorConfig,
)
from value_as_tool.schemas import (
    AssistantMessage,
    ChatCompletion,
    Role,
    TokenUsage,
    ToolCall,
    TrajectoryRequest,
    TrajectoryResult,
    TrajectoryStatus,
)

JOINT = "value_as_tool.harnesses.gvr_replan:JointPlanHarness"
INDEPENDENT = "value_as_tool.harnesses.gvr_replan:IndependentPlanHarness"
HARNESS_IDS = {JOINT: "gvr_replan_joint", INDEPENDENT: "gvr_replan_independent"}
BRANCHED_SHA256 = "bf02f28347915ab6679f20ea217ed3b76e6a369433fdceff5b719aeeffa37066"
GOLD = "GOLD-ANSWER-SENTINEL"
OUTCOMES = ("correct", "minor_fix", "critical_flaw", "correct")
both = pytest.mark.parametrize("entrypoint", [JOINT, INDEPENDENT], ids=["joint", "independent"])


def _usage(completion: int) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=5,
        completion_tokens=completion,
        total_tokens=5 + completion,
        reasoning_tokens=completion,
    )


def _completion(
    content: str | None = None,
    *,
    tool_calls: Sequence[ToolCall] = (),
    completion_tokens: int = 7,
    finish_reason: str | None = None,
) -> ChatCompletion:
    return ChatCompletion(
        id="offline-completion",
        model="Qwen/Qwen3.5-9B",
        message=AssistantMessage(content=content, reasoning=None, tool_calls=tuple(tool_calls)),
        finish_reason=finish_reason or ("tool_calls" if tool_calls else "stop"),
        usage=_usage(completion_tokens),
    )


def _tool(name: str, arguments: Mapping[str, Any]) -> ChatCompletion:
    return _completion(
        tool_calls=[ToolCall(id=f"{name}-call", name=name, arguments=json.dumps(arguments))]
    )


def _verdict(outcome: str, probability: float, rationale: str) -> ChatCompletion:
    return _tool(
        "submit_verdict",
        {
            "outcome": outcome,
            "success_probability": probability,
            "rationale": rationale,
            "fault_category": "logic",
            "candidate_excerpt": "",
        },
    )


def _plan_fields(slot: int, point: int, branch: int, *, show: bool | None = None) -> dict[str, Any]:
    return {
        f"plan_{slot}_title": f"plan p{point} b{branch}",
        f"plan_{slot}_brief": f"brief p{point} b{branch}",
        f"plan_{slot}_show_current_solution": branch % 2 == 0 if show is None else show,
        f"plan_{slot}_success_probability": round(0.2 * (branch + 1), 2),
    }


def _config(**overrides: Any) -> OrchestratorConfig:
    values: dict[str, Any] = {
        "model": "offline-qwen",
        "total_generated_tokens": 8_388_608,
        "context_tokens": 262_144,
        "context_headroom_tokens": 1_024,
        "initial_generator_cap": 2_621_440,
        "verifier_cap": 524_288,
        "correction_pool": 5_242_880,
        "minimum_call_tokens": 1_024,
        "subagent_cap": 262_144,
        "final_candidate_reserve_tokens": 262_144,
        "value_verifier_cap": 262_144,
        "value_final_response_reserve_tokens": 262_144,
    }
    values.update(overrides)
    return OrchestratorConfig(**values)


def _request(entrypoint: str, **overrides: Any) -> TrajectoryRequest:
    values: dict[str, Any] = {
        "benchmark": "imo_answer",
        "problem_id": "offline-replan-problem",
        "problem": "Compute the offline sentinel quantity.",
        "condition": None,
        "seed": 23,
        "harness_id": HARNESS_IDS[entrypoint],
        "solver_prompt": "Solve the offline task. Put the final answer within \\boxed{}.",
        "metadata": {"answer": GOLD},
    }
    values.update(overrides)
    return TrajectoryRequest(**values)


def _labels() -> list[str]:
    labels = ["gen.t0", "gen.t1"]
    for point in range(1, ROUNDS + 1):
        labels += [f"r{point:02d}.plan", f"r{point:02d}.plan.recovery"]
        for branch in range(BRANCHES):
            prefix = f"r{point:02d}.b{branch}"
            labels += [
                f"{prefix}.verify",
                f"{prefix}.verify.recovery",
                f"{prefix}.plan",
                f"{prefix}.plan.recovery",
                f"{prefix}.exec.t0",
                f"{prefix}.exec.t1",
            ]
    return labels


def _point(label: str) -> int:
    return int(label.split(".")[0][1:])


def _branch(label: str) -> int:
    return int(label.split(".")[1][1:])


def default_policy(label: str, messages: Sequence[Mapping[str, Any]]) -> ChatCompletion:
    if ".verify" in label:
        point, branch = _point(label), _branch(label)
        return _verdict(OUTCOMES[(point + branch) % 4], 0.5 + 0.1 * branch, f"rationale {label}")
    if label.endswith(".plan") and label.count(".") == 1:
        point = _point(label)
        fields: dict[str, Any] = {}
        for branch in range(BRANCHES):
            fields.update(_plan_fields(branch + 1, point, branch))
        return _tool("submit_plans", fields)
    if label.endswith(".plan"):
        return _tool("submit_plans", _plan_fields(1, _point(label), _branch(label)))
    return _completion(f"candidate from {label} \\boxed{{{label}}}")


Policy = Callable[[str, Sequence[Mapping[str, Any]]], ChatCompletion | BaseException]


class TreeClient:
    """Answers by call label, recovered from the label-derived seed."""

    def __init__(
        self,
        request: TrajectoryRequest,
        policy: Policy = default_policy,
        *,
        gates: Mapping[str, asyncio.Event] | None = None,
    ) -> None:
        self.labels = {request.seed: "gen.t0"}
        for label in _labels()[1:]:
            seed = AletheiaOrchestrator.stable_seed_for_harness(
                request.seed, request.problem_id, label
            )
            assert seed not in self.labels
            self.labels[seed] = label
        self.policy = policy
        self.gates = dict(gates or {})
        self.calls: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = []
        self.started: dict[str, asyncio.Event] = {}

    async def complete(
        self, messages: Sequence[Mapping[str, Any]], **kwargs: Any
    ) -> ChatCompletion:
        label = self.labels[kwargs["seed"]]
        self.calls.append(
            (label, [copy.deepcopy(dict(m)) for m in messages], copy.deepcopy(kwargs))
        )
        self.started.setdefault(label, asyncio.Event()).set()
        await asyncio.sleep(0)
        if label in self.gates:
            await self.gates[label].wait()
        response = self.policy(label, messages)
        if isinstance(response, BaseException):
            raise response
        return response


async def _run(
    entrypoint: str,
    client: Any,
    request: TrajectoryRequest,
    *,
    config: OrchestratorConfig | None = None,
    resume: Mapping[str, Any] | None = None,
    checkpoint: Callable[[TrajectoryResult], None] | None = None,
) -> TrajectoryResult:
    orchestrator = AletheiaOrchestrator(client, config or _config(), checkpoint=checkpoint)
    return await orchestrator.run(request, resume=resume, harness=load_harness(entrypoint))


def _tree(result: TrajectoryResult) -> dict[tuple[int, int | None], Any]:
    return {(candidate.cycle, candidate.branch): candidate for candidate in result.candidates}


def _sent(client: TreeClient) -> dict[str, tuple[list[dict[str, Any]], dict[str, Any]]]:
    return {label: (messages, kwargs) for label, messages, kwargs in client.calls}


def _planner_messages(client: TreeClient, point: int) -> list[str]:
    return [
        messages[1]["content"]
        for label, messages, _ in client.calls
        if label.startswith(f"r{point:02d}.") and label.endswith(".plan")
    ]


@both
@pytest.mark.asyncio
async def test_tree_runs_every_point_with_planned_branches(entrypoint: str) -> None:
    request = _request(entrypoint)
    client = TreeClient(request)
    result = await _run(entrypoint, client, request)

    assert result.status is TrajectoryStatus.CYCLE_LIMIT
    labels = [call.label for call in result.calls]
    planners = 1 if entrypoint == JOINT else BRANCHES
    assert len(labels) == len(set(labels)) == 1 + ROUNDS * (2 * BRANCHES + planners)
    tree = _tree(result)
    points = range(1, ROUNDS + 1)
    assert set(tree) == {(1, None)} | {(p + 1, b) for p in points for b in range(BRANCHES)}
    assert {(v.cycle, v.branch) for v in result.verdicts} == {
        (p, b) for p in points for b in range(BRANCHES)
    }
    assert all(v.success_probability is not None and v.rationale for v in result.verdicts)

    calls = {call.label: call for call in result.calls}
    for label, call in calls.items():
        if ".verify" in label:
            assert (
                call.role is Role.VERIFIER
                and call.max_tokens == VERIFIER_CAP - VERIFIER_RECOVERY_RESERVE
            )
            assert call.tools[0]["function"] == SUBMIT_RATIONALE_SCORE_VERDICT_TOOL["function"]
        elif label.endswith(".plan"):
            assert (
                call.role is Role.PLANNER
                and call.max_tokens == PLANNER_CAP - PLANNER_RECOVERY_RESERVE
            )
        elif ".exec." in label:
            assert (
                call.role is Role.WORKER
                and call.max_tokens == CANDIDATE_CAP - CANDIDATE_RECOVERY_RESERVE
            )
    assert len({call.seed for call in result.calls}) == len(result.calls)
    assert not any(GOLD in json.dumps(messages) for _, messages, _ in client.calls)

    # c1 is requested exactly as gvr_branched requests it.
    first = calls["gen.t0"]
    assert first.seed == request.seed
    assert list(first.messages) == [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": request.solver_prompt},
    ]
    assert first.max_tokens == CANDIDATE_CAP - CANDIDATE_RECOVERY_RESERVE

    # The spine follows the seeded promotion order and every promotion is recorded.
    promotes = {t.cycle: t for t in result.transitions if t.action == "promote"}
    supports = {
        t.cycle: json.loads(t.detail) for t in result.transitions if t.action == "promotion_support"
    }
    assert set(promotes) == set(supports) == set(range(2, ROUNDS + 1))
    spine = tree[(1, None)]
    for point in range(1, ROUNDS):
        detail = json.loads(promotes[point + 1].detail)
        branch = int(promotes[point + 1].source.removeprefix("branch_"))
        assert detail["rank"] == 0 and branch == detail["order"][0]
        assert supports[point + 1] == {"p": 0.25, "successful": [0, 1, 2, 3]}
        assert tree[(point + 1, branch)].parent_call_index == spine.call_index
        spine = tree[(point + 1, branch)]
    assert result.final_output == spine.content


@pytest.mark.asyncio
async def test_both_variants_give_their_planners_byte_identical_inputs() -> None:
    joint, independent = TreeClient(_request(JOINT)), TreeClient(_request(INDEPENDENT))
    await _run(JOINT, joint, _request(JOINT))
    await _run(INDEPENDENT, independent, _request(INDEPENDENT))
    for point in range(1, ROUNDS + 1):
        (joint_message,) = _planner_messages(joint, point)
        independent_messages = _planner_messages(independent, point)
        assert len(independent_messages) == BRANCHES
        assert set(independent_messages) == {joint_message}
    joint_system = _sent(joint)["r01.plan"][0][0]["content"]
    independent_system = _sent(independent)["r01.b0.plan"][0][0]["content"]
    assert joint_system.split("\n\n")[0] == independent_system.split("\n\n")[0]
    joint_tool = _sent(joint)["r01.plan"][1]["tools"][0]["function"]["parameters"]["properties"]
    independent_tool = _sent(independent)["r01.b0.plan"][1]["tools"][0]["function"]["parameters"]
    assert len(joint_tool) == 4 * BRANCHES and len(independent_tool["properties"]) == 4


@both
@pytest.mark.asyncio
async def test_the_planner_sees_the_spine_history_and_never_leaves_or_gold(entrypoint: str) -> None:
    request = _request(entrypoint)
    client = TreeClient(request)
    result = await _run(entrypoint, client, request)
    tree = _tree(result)
    message = _planner_messages(client, 3)[0]

    promoted = [
        int(t.source.removeprefix("branch_"))
        for t in sorted(result.transitions, key=lambda t: t.cycle)
        if t.action == "promote"
    ]
    spine = [tree[(1, None)], tree[(2, promoted[0])], tree[(3, promoted[1])]]
    for attempt, node in enumerate(spine, 1):
        assert node.content in message or attempt < 3
        answer = node.content.split("\\boxed{")[1].rstrip("}")
        assert f"Attempt {attempt} " in message and answer in message
    assert "Attempt 1 (initial attempt)" in message
    assert f'Attempt 2 (plan "plan p1 b{promoted[0]}")' in message
    assert "mean verifier probability" in message
    leaves = [tree[(3, b)] for b in range(BRANCHES) if b != promoted[1]]
    assert not any(leaf.content in message for leaf in leaves)
    assert message.count("Assessment ") == BRANCHES
    assert "rationale r03.b2.verify" in message and GOLD not in message


@both
@pytest.mark.asyncio
async def test_the_executor_sees_only_the_task_its_brief_and_the_solution_if_shown(
    entrypoint: str,
) -> None:
    pasted = "Verifier said: rationale r01.b1.verify"

    def policy(label: str, messages: Sequence[Mapping[str, Any]]) -> Any:
        response = default_policy(label, messages)
        if label.startswith("r01.") and label.endswith(".plan"):
            arguments = json.loads(response.message.tool_calls[0].arguments)
            key = "plan_2_brief" if label == "r01.plan" else "plan_1_brief"
            if label in {"r01.plan", "r01.b1.plan"}:
                arguments[key] = f"brief p1 b1\n{pasted}"
            return _tool("submit_plans", arguments)
        return response

    request = _request(entrypoint)
    client = TreeClient(request, policy)
    result = await _run(entrypoint, client, request)
    c1 = _tree(result)[(1, None)].content
    sent = _sent(client)
    for branch in range(BRANCHES):
        system, user = sent[f"r01.b{branch}.exec.t0"][0]
        assert system == {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT}
        content = user["content"]
        assert content.startswith(
            f"Original task and required output format:\n{request.solver_prompt}"
        )
        assert f"Brief from the planner:\nbrief p1 b{branch}" in content
        assert (c1 in content) is (branch % 2 == 0)
        assert ("rationale r01.b0.verify" in content) is False
        assert (pasted in content) is (branch == 1)


def test_promotion_order_is_a_uniform_permutation() -> None:
    class Stub:
        def __init__(self, problem: str) -> None:
            self.problem = problem

        def stable_seed(self, label: str) -> int:
            return AletheiaOrchestrator.stable_seed_for_harness(0, self.problem, label)

    firsts = Counter()
    for index in range(4_000):
        order = promotion_order(Stub(f"problem-{index}"), 3)  # type: ignore[arg-type]
        assert sorted(order) == list(range(BRANCHES))
        firsts[order[0]] += 1
    assert all(abs(firsts[branch] / 4_000 - 0.25) < 0.03 for branch in range(BRANCHES))


@both
@pytest.mark.asyncio
async def test_a_failed_first_ranked_branch_hands_the_spine_to_the_next(entrypoint: str) -> None:
    request = _request(entrypoint)
    order = promotion_order(_StubRuntime(request), 1)
    failing = f"r01.b{order[0]}.exec"

    def policy(label: str, messages: Sequence[Mapping[str, Any]]) -> Any:
        if label.startswith(failing):
            return _completion(None)
        return default_policy(label, messages)

    result = await _run(entrypoint, TreeClient(request, policy), request)
    assert result.status is TrajectoryStatus.CYCLE_LIMIT
    promote = next(t for t in result.transitions if t.action == "promote" and t.cycle == 2)
    assert promote.source == f"branch_{order[1]}" and json.loads(promote.detail)["rank"] == 1
    support = next(
        t for t in result.transitions if t.action == "promotion_support" and t.cycle == 2
    )
    assert json.loads(support.detail) == {
        "p": round(1 / 3, 6),
        "successful": sorted(set(range(BRANCHES)) - {order[0]}),
    }
    assert (2, order[0]) not in _tree(result)


class _StubRuntime:
    def __init__(self, request: TrajectoryRequest) -> None:
        self.request = request

    def stable_seed(self, label: str) -> int:
        return AletheiaOrchestrator.stable_seed_for_harness(
            self.request.seed, self.request.problem_id, label
        )


@both
@pytest.mark.asyncio
async def test_a_straggling_branch_does_not_block_the_next_point(entrypoint: str) -> None:
    request = _request(entrypoint)
    order = promotion_order(_StubRuntime(request), 1)
    gate = asyncio.Event()
    straggler = f"r01.b{order[-1]}.exec.t0"
    client = TreeClient(request, gates={straggler: gate})
    run = asyncio.create_task(_run(entrypoint, client, request))
    await client.started.setdefault("r02.b0.verify", asyncio.Event()).wait()
    assert not run.done()
    gate.set()
    result = await run
    assert result.status is TrajectoryStatus.CYCLE_LIMIT


@both
@pytest.mark.asyncio
async def test_a_malformed_plan_gets_one_sampled_recovery_without_thinking(entrypoint: str) -> None:
    first = "r02.plan" if entrypoint == JOINT else "r02.b1.plan"

    def policy(label: str, messages: Sequence[Mapping[str, Any]]) -> Any:
        if label == first:
            return _completion(
                "I would rather explain the plan in prose.", completion_tokens=30_000
            )
        if label == f"{first}.recovery":
            return default_policy(first, messages)
        return default_policy(label, messages)

    request = _request(entrypoint)
    client = TreeClient(request, policy)
    result = await _run(entrypoint, client, request)
    assert result.status is TrajectoryStatus.CYCLE_LIMIT
    messages, kwargs = _sent(client)[f"{first}.recovery"]
    assert kwargs["use_sampling"] is True
    assert kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert messages[0]["content"].endswith(PLANNER_RECOVERY_SUFFIX)
    calls = {call.label: call for call in result.calls}
    assert calls[f"{first}.recovery"].max_tokens == PLANNER_CAP - 30_000
    plans = [
        json.loads(t.detail) for t in result.transitions if t.action == "plan" and t.cycle == 2
    ]
    assert any(plan["recovered"] for plan in plans)
    assert len(result.candidates) == 1 + ROUNDS * BRANCHES


@pytest.mark.asyncio
async def test_joint_partial_plans_run_and_a_dead_planner_ends_the_tree() -> None:
    def partial(label: str, messages: Sequence[Mapping[str, Any]]) -> Any:
        response = default_policy(label, messages)
        if label == "r02.plan":
            arguments = json.loads(response.message.tool_calls[0].arguments)
            arguments["plan_2_success_probability"] = 1.5
            arguments["plan_4_show_current_solution"] = "maybe"
            return _tool("submit_plans", arguments)
        return response

    request = _request(JOINT)
    client = TreeClient(request, partial)
    result = await _run(JOINT, client, request)
    assert result.status is TrajectoryStatus.CYCLE_LIMIT
    assert {(3, 0), (3, 2)} <= set(_tree(result)) and not {(3, 1), (3, 3)} & set(_tree(result))
    assert "r02.plan.recovery" not in _sent(client)
    failed = [t for t in result.transitions if t.action == "failed" and t.source == "planner"]
    assert {t.detail.split(":")[0] for t in failed} == {"branch_1", "branch_3"}

    def dead(label: str, messages: Sequence[Mapping[str, Any]]) -> Any:
        if label in {"r03.plan", "r03.plan.recovery"}:
            return _completion("no tool call")
        return default_policy(label, messages)

    client = TreeClient(request, dead)
    result = await _run(JOINT, client, request)
    assert result.status is TrajectoryStatus.PROTOCOL_ERROR
    tree = _tree(result)
    promoted = [
        t for t in sorted(result.transitions, key=lambda t: t.cycle) if t.action == "promote"
    ]
    c3 = tree[(3, int(promoted[-1].source.removeprefix("branch_")))]
    assert result.final_output == c3.content
    assert not any(label.startswith("r04.") for label, _, _ in client.calls)
    assert any(t.action == "no_candidate" and t.cycle == 3 for t in result.transitions)


@pytest.mark.asyncio
async def test_an_independent_planner_failure_ends_only_its_branch() -> None:
    def policy(label: str, messages: Sequence[Mapping[str, Any]]) -> Any:
        if label in {"r02.b3.plan", "r02.b3.plan.recovery"}:
            return _completion("no tool call")
        return default_policy(label, messages)

    request = _request(INDEPENDENT)
    result = await _run(INDEPENDENT, TreeClient(request, policy), request)
    assert result.status is TrajectoryStatus.CYCLE_LIMIT
    assert (3, 3) not in _tree(result) and len(result.candidates) == ROUNDS * BRANCHES


@both
@pytest.mark.asyncio
async def test_unavailable_assessments_still_reach_the_planner(entrypoint: str) -> None:
    def policy(label: str, messages: Sequence[Mapping[str, Any]]) -> Any:
        if label.startswith("r01.") and ".verify" in label:
            return _completion("not a verdict")
        return default_policy(label, messages)

    request = _request(entrypoint)
    client = TreeClient(request, policy)
    result = await _run(entrypoint, client, request)
    assert result.status is TrajectoryStatus.CYCLE_LIMIT
    message = _planner_messages(client, 1)[0]
    assert message.count(UNAVAILABLE_ASSESSMENT) == BRANCHES
    assert "no verifier assessment" in message
    assert not any(v.cycle == 1 for v in result.verdicts)


@both
@pytest.mark.asyncio
async def test_known_usage_errors_are_contained_and_missing_usage_is_fatal(entrypoint: str) -> None:
    def contained(label: str, messages: Sequence[Mapping[str, Any]]) -> Any:
        if label == "r02.b2.exec.t0":
            return ChatClientError("malformed response", usage=_usage(11))
        return default_policy(label, messages)

    request = _request(entrypoint)
    result = await _run(entrypoint, TreeClient(request, contained), request)
    assert result.status is TrajectoryStatus.CYCLE_LIMIT
    assert (3, 2) not in _tree(result)

    def fatal(label: str, messages: Sequence[Mapping[str, Any]]) -> Any:
        if label.endswith(".plan") and label.startswith("r02."):
            return MissingUsageError("connection reset")
        return default_policy(label, messages)

    result = await _run(entrypoint, TreeClient(request, fatal), request)
    assert result.status is TrajectoryStatus.INVALID_USAGE


@both
@pytest.mark.asyncio
async def test_a_budget_below_the_worst_case_is_rejected_before_any_call(entrypoint: str) -> None:
    assert JOINT_WORST_CASE_TOKENS == 6_488_064
    assert INDEPENDENT_WORST_CASE_TOKENS == 7_962_624
    request = _request(entrypoint)
    client = TreeClient(request)
    config = _config(
        total_generated_tokens=6_000_000,
        initial_generator_cap=1_000_000,
        verifier_cap=500_000,
        correction_pool=4_000_000,
    )
    result = await _run(entrypoint, client, request, config=config)
    assert result.status is TrajectoryStatus.PROTOCOL_ERROR
    assert client.calls == []


@both
@pytest.mark.asyncio
async def test_resume_from_a_mid_tree_checkpoint_dispatches_only_missing_calls(
    entrypoint: str,
) -> None:
    request = _request(entrypoint)
    snapshots: list[TrajectoryResult] = []
    complete = await _run(
        entrypoint,
        TreeClient(request),
        request,
        checkpoint=lambda value: snapshots.append(copy.deepcopy(value)),
    )
    boundary = next(
        snapshot
        for snapshot in snapshots
        if snapshot.status is TrajectoryStatus.RUNNING
        and len(snapshot.calls) >= 30
        and [call.index for call in snapshot.calls] == list(range(len(snapshot.calls)))
    )
    done = {call.label for call in boundary.calls}
    resumed_client = TreeClient(request)
    resumed = await _run(
        entrypoint, resumed_client, request, resume=json.loads(json.dumps(boundary.to_dict()))
    )
    assert {label for label, _, _ in resumed_client.calls} == {
        c.label for c in complete.calls
    } - done
    assert resumed.status is complete.status and resumed.final_output == complete.final_output
    assert {k: n.content for k, n in _tree(resumed).items()} == {
        k: n.content for k, n in _tree(complete).items()
    }
    assert sorted(t.to_dict().items() for t in resumed.transitions) == sorted(
        t.to_dict().items() for t in complete.transitions
    )
    finished = TreeClient(request)
    await _run(entrypoint, finished, request, resume=json.loads(json.dumps(complete.to_dict())))
    assert finished.calls == []


def test_plan_parser_tolerates_qwen_quirks_and_rejects_bad_fields() -> None:
    (plan,) = parse_plan_arguments(
        {
            "plan_1_title": "",
            "plan_1_brief": "Restart\nwith induction",
            "plan_1_show_current_solution": "false",
            "plan_1_success_probability": "0.35",
        },
        1,
    )
    assert plan.valid and plan.show_current_solution is False and plan.success_probability == 0.35
    assert plan.title == "Restart" and plan.title_derived

    (folded,) = parse_plan_arguments(
        {
            "plan_1_title": "t",
            "plan_1_brief": "do it</parameter>\n<parameter=plan_1_show_current_solution>true",
            "plan_1_success_probability": 0.5,
        },
        1,
    )
    assert folded.valid and folded.brief == "do it" and folded.show_current_solution is True

    (listed,) = parse_plan_arguments(
        {
            "plan_1_title": "t",
            "plan_1_brief": ["check case n=1", "check the bound"],
            "plan_1_show_current_solution": 1,
            "plan_1_success_probability": 1,
        },
        1,
    )
    assert listed.valid and listed.brief == "- check case n=1\n- check the bound"

    for bad in (1.5, True, "70%", None):
        (plan,) = parse_plan_arguments(
            {
                "plan_1_title": "t",
                "plan_1_brief": "b",
                "plan_1_show_current_solution": True,
                "plan_1_success_probability": bad,
            },
            1,
        )
        assert not plan.valid and "success_probability" in (plan.error or "")

    (long,) = parse_plan_arguments(
        {
            "plan_1_title": "t",
            "plan_1_brief": "x" * (BRIEF_MAX_CHARS + 5),
            "plan_1_show_current_solution": True,
            "plan_1_success_probability": 0.1,
        },
        1,
    )
    assert long.valid and long.brief_truncated and len(long.brief or "") == BRIEF_MAX_CHARS

    assert parse_plan_completion(_tool("submit_verdict", {"plan_1_brief": "b"}), 1) is None
    two = _completion(
        tool_calls=[
            ToolCall(id="a", name="submit_plans", arguments="{}"),
            ToolCall(id="b", name="submit_plans", arguments="{}"),
        ]
    )
    assert parse_plan_completion(two, 1) is None


def test_both_harnesses_are_registered_and_the_baseline_harness_is_unchanged() -> None:
    joint, independent = resolve_harness(JOINT), resolve_harness(INDEPENDENT)
    assert {joint.harness_id, independent.harness_id} == set(HARNESS_IDS.values())
    assert joint.source_sha256 == independent.source_sha256
    assert {JOINT, INDEPENDENT} <= set(DATA_COLLECTION_HARNESSES)
    branched = Path(__file__).resolve().parents[1] / "src/value_as_tool/harnesses/gvr_branched.py"
    assert hashlib.sha256(branched.read_bytes()).hexdigest() == BRANCHED_SHA256
