"""Branched GVR harness: tree shape, routing, concurrency, recovery and resume."""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import pytest

from value_as_tool.client import ChatClientError, MissingUsageError
from value_as_tool.harnesses import load_harness
from value_as_tool.harnesses.gvr_branched import (
    BRANCHES,
    CANDIDATE_CAP,
    CANDIDATE_RECOVERY_RESERVE,
    RECHECK_SYSTEM_PROMPT,
    ROUNDS,
    VERIFIER_CAP,
    VERIFIER_RECOVERY_RESERVE,
)
from value_as_tool.orchestrator import (
    GENERATOR_SYSTEM_PROMPT,
    GVR_FORCED_CANDIDATE_REMINDER,
    REVISER_SYSTEM_PROMPT,
    AletheiaOrchestrator,
    OrchestratorConfig,
)
from value_as_tool.schemas import (
    AssistantMessage,
    ChatCompletion,
    Condition,
    Role,
    TokenUsage,
    ToolCall,
    TrajectoryRequest,
    TrajectoryResult,
    TrajectoryStatus,
    Verdict,
)

ENTRYPOINT = "value_as_tool.harnesses.gvr_branched:AgentHarness"
GOLD = "GOLD-ANSWER-SENTINEL"
OUTCOMES = ("correct", "minor_fix", "critical_flaw", "correct")


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
    reasoning: str | None = None,
    tool_calls: Sequence[ToolCall] = (),
    completion_tokens: int = 7,
) -> ChatCompletion:
    return ChatCompletion(
        id="offline-completion",
        model="Qwen/Qwen3.5-9B",
        message=AssistantMessage(
            content=content, reasoning=reasoning, tool_calls=tuple(tool_calls)
        ),
        finish_reason="tool_calls" if tool_calls else "stop",
        usage=_usage(completion_tokens),
    )


def _verdict(outcome: str, *, critique: str = "The cited step needs attention.") -> ChatCompletion:
    arguments = {
        "outcome": outcome,
        "critique": critique,
        "fault_category": "logic",
        "candidate_excerpt": "",
    }
    return _completion(
        tool_calls=[
            ToolCall(id="verdict-call", name="submit_verdict", arguments=json.dumps(arguments))
        ]
    )


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


def _request(**overrides: Any) -> TrajectoryRequest:
    values: dict[str, Any] = {
        "benchmark": "imo_answer",
        "problem_id": "offline-tree-problem",
        "problem": "Compute the offline sentinel quantity.",
        "condition": None,
        "seed": 23,
        "harness_id": "gvr_branched",
        "solver_prompt": "Solve the offline task. Put the final answer within \\boxed{}.",
        "metadata": {"answer": GOLD},
    }
    values.update(overrides)
    return TrajectoryRequest(**values)


def _labels() -> list[str]:
    labels = ["gen.t0", "gen.t1"]
    for point in range(1, ROUNDS + 1):
        for branch in range(BRANCHES):
            prefix = f"r{point:02d}.b{branch}"
            labels += [f"{prefix}.verify", f"{prefix}.verify.recovery"]
            for mode in ("recheck", "revise", "regenerate"):
                labels += [f"{prefix}.{mode}.t0", f"{prefix}.{mode}.t1"]
    return labels


def _point_branch(label: str) -> tuple[int, int]:
    point, branch = label.split(".")[:2]
    return int(point[1:]), int(branch[1:])


def _default_policy(label: str, messages: Sequence[Mapping[str, Any]]) -> ChatCompletion:
    if ".verify" in label:
        point, branch = _point_branch(label)
        return _verdict(OUTCOMES[(point + branch) % 4], critique=f"critique for {label}")
    return _completion(f"candidate from {label} \\boxed{{{label}}}")


Policy = Callable[[str, Sequence[Mapping[str, Any]]], ChatCompletion | BaseException]


class TreeClient:
    """Answers by call label, recovered from the label-derived seed."""

    def __init__(
        self,
        request: TrajectoryRequest,
        policy: Policy = _default_policy,
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
        self.in_flight = 0
        self.max_in_flight = 0

    async def dispatched(self, label: str) -> None:
        await self.started.setdefault(label, asyncio.Event()).wait()

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        label = self.labels[kwargs["seed"]]
        self.calls.append(
            (label, [copy.deepcopy(dict(message)) for message in messages], copy.deepcopy(kwargs))
        )
        self.started.setdefault(label, asyncio.Event()).set()
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0)
            if label in self.gates:
                await self.gates[label].wait()
            response = self.policy(label, messages)
        finally:
            self.in_flight -= 1
        if isinstance(response, BaseException):
            raise response
        return response


class ScriptedClient:
    def __init__(self, *responses: ChatCompletion) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

    async def complete(
        self, messages: Sequence[Mapping[str, Any]], **kwargs: Any
    ) -> ChatCompletion:
        self.calls.append(([copy.deepcopy(dict(message)) for message in messages], kwargs))
        return self.responses.pop(0)


async def _run(
    client: Any,
    request: TrajectoryRequest,
    *,
    config: OrchestratorConfig | None = None,
    resume: Mapping[str, Any] | None = None,
    checkpoint: Callable[[TrajectoryResult], None] | None = None,
) -> TrajectoryResult:
    orchestrator = AletheiaOrchestrator(client, config or _config(), checkpoint=checkpoint)
    return await orchestrator.run(request, resume=resume, harness=load_harness(ENTRYPOINT))


def _tree(result: TrajectoryResult) -> dict[tuple[int, int | None], Any]:
    return {(candidate.cycle, candidate.branch): candidate for candidate in result.candidates}


@pytest.mark.asyncio
async def test_tree_runs_every_point_and_branch_without_stopping_at_correct() -> None:
    request = _request()
    client = TreeClient(request)
    result = await _run(client, request)

    assert result.status is TrajectoryStatus.CYCLE_LIMIT
    labels = [call.label for call in result.calls]
    assert len(labels) == len(set(labels)) == 1 + 2 * ROUNDS * BRANCHES
    tree = _tree(result)
    points = range(1, ROUNDS + 1)
    assert set(tree) == {(1, None)} | {(p + 1, b) for p in points for b in range(BRANCHES)}
    assert {(v.cycle, v.branch) for v in result.verdicts} == {
        (p, b) for p in points for b in range(BRANCHES)
    }
    assert sum(v.verdict is Verdict.CORRECT for v in result.verdicts) >= ROUNDS

    # Sibling 0 continues the spine; every node points at the state it branched from.
    spine = tree[(1, None)]
    for point in points:
        assert {v.parent_call_index for v in result.verdicts if v.cycle == point} == {
            spine.call_index
        }
        assert {tree[(point + 1, b)].parent_call_index for b in range(BRANCHES)} == {
            spine.call_index
        }
        if point < ROUNDS:
            spine = tree[(point + 1, 0)]
    assert result.final_output == tree[(ROUNDS, 0)].content

    calls = {call.label: call for call in result.calls}
    for verdict in result.verdicts:
        mode, role, system = {
            Verdict.CORRECT: ("recheck", Role.REVISER, RECHECK_SYSTEM_PROMPT),
            Verdict.MINOR_FIX: ("revise", Role.REVISER, REVISER_SYSTEM_PROMPT),
            Verdict.CRITICAL_FLAW: ("regenerate", Role.GENERATOR, GENERATOR_SYSTEM_PROMPT),
        }[verdict.verdict]
        call = calls[f"r{verdict.cycle:02d}.b{verdict.branch}.{mode}.t0"]
        assert call.role is role
        assert call.cycle == verdict.cycle + 1
        assert call.messages[0]["content"] == system
    assert len({call.seed for call in result.calls}) == len(result.calls)
    for call in result.calls:
        expected = (
            VERIFIER_CAP - VERIFIER_RECOVERY_RESERVE
            if call.role is Role.VERIFIER
            else CANDIDATE_CAP - CANDIDATE_RECOVERY_RESERVE
        )
        assert call.max_tokens == expected
    assert GOLD not in json.dumps([messages for _, messages, _ in client.calls])
    assert result.budget["reserved_generated_tokens"] == 0
    assert result.budget["unknown_usage"] == []


@pytest.mark.asyncio
async def test_verifier_and_revision_messages_match_builtin_gvr() -> None:
    critique = "Step two divides by zero."
    builtin_client = ScriptedClient(
        _completion("Candidate A \\boxed{1}"),
        _verdict("minor_fix", critique=critique),
        _completion("Candidate B \\boxed{2}"),
        _verdict("correct"),
    )
    await AletheiaOrchestrator(builtin_client, _config()).run(
        replace(_request(), condition=Condition.GVR, harness_id=None)
    )

    def policy(label: str, messages: Sequence[Mapping[str, Any]]) -> ChatCompletion:
        if label == "gen.t0":
            return _completion("Candidate A \\boxed{1}")
        if label == "r01.b1.verify":
            return _verdict("minor_fix", critique=critique)
        return _default_policy(label, messages)

    request = _request()
    client = TreeClient(request, policy)
    await _run(client, request)
    sent = {label: messages for label, messages, _ in client.calls}
    assert sent["gen.t0"] == builtin_client.calls[0][0]
    assert sent["r01.b0.verify"] == builtin_client.calls[1][0]
    assert sent["r01.b1.revise.t0"] == builtin_client.calls[2][0]


@pytest.mark.asyncio
async def test_siblings_run_concurrently_and_a_straggler_does_not_block_the_spine() -> None:
    request = _request()
    gate = asyncio.Event()
    client = TreeClient(request, gates={"r01.b3.verify": gate})
    task = asyncio.create_task(_run(client, request))

    await asyncio.wait_for(client.dispatched("r02.b0.verify"), timeout=10)
    started = [label for label, _, _ in client.calls]
    assert not [label for label in started if label.startswith("r01.b3.") and "verify" not in label]
    gate.set()
    result = await asyncio.wait_for(task, timeout=30)

    assert result.status is TrajectoryStatus.CYCLE_LIMIT
    assert client.max_in_flight >= BRANCHES
    assert (2, 3) in _tree(result)


@pytest.mark.asyncio
async def test_a_failed_sibling_zero_hands_the_spine_to_the_next_sibling() -> None:
    def policy(label: str, messages: Sequence[Mapping[str, Any]]) -> ChatCompletion:
        if label in {"r01.b0.verify", "r01.b0.verify.recovery"}:
            return _completion("A verdict in prose, without the tool.")
        return _default_policy(label, messages)

    request = _request()
    result = await _run(TreeClient(request, policy), request)

    assert result.status is TrajectoryStatus.CYCLE_LIMIT
    tree = _tree(result)
    assert (2, 0) not in tree and (2, 1) in tree
    promotions = [t for t in result.transitions if t.action == "promote" and t.cycle == 2]
    assert [t.source for t in promotions] == ["branch_1"]
    assert {v.parent_call_index for v in result.verdicts if v.cycle == 2} == {
        tree[(2, 1)].call_index
    }


@pytest.mark.asyncio
async def test_a_point_without_any_surviving_sibling_ends_the_tree_cleanly() -> None:
    def policy(label: str, messages: Sequence[Mapping[str, Any]]) -> ChatCompletion:
        if label.startswith("r03.") and ".verify" in label:
            return _completion("No verdict tool call.")
        return _default_policy(label, messages)

    request = _request()
    result = await _run(TreeClient(request, policy), request)

    assert result.status is TrajectoryStatus.PROTOCOL_ERROR
    assert result.final_output == _tree(result)[(3, 0)].content
    assert not [call for call in result.calls if call.label.startswith("r04.")]
    assert result.budget["reserved_generated_tokens"] == 0
    assert result.budget["unknown_usage"] == []


@pytest.mark.asyncio
async def test_verifier_and_candidate_recoveries_stay_within_their_node_caps() -> None:
    def policy(label: str, messages: Sequence[Mapping[str, Any]]) -> ChatCompletion:
        if label == "gen.t0":
            return _completion(None, reasoning="Only private reasoning.", completion_tokens=90_000)
        if label == "gen.t1":
            return _completion("Recovered candidate \\boxed{2}")
        if label == "r01.b0.verify":
            return _completion("Thinking out loud.", completion_tokens=40_000)
        if label == "r01.b0.verify.recovery":
            return _verdict("minor_fix")
        return _default_policy(label, messages)

    request = _request()
    client = TreeClient(request, policy)
    result = await _run(client, request)

    calls = {call.label: call for call in result.calls}
    assert calls["gen.t0"].max_tokens == CANDIDATE_CAP - CANDIDATE_RECOVERY_RESERVE
    assert calls["gen.t1"].max_tokens == CANDIDATE_CAP - 90_000
    assert calls["r01.b0.verify.recovery"].max_tokens == VERIFIER_CAP - 40_000
    sent = {label: (messages, kwargs) for label, messages, kwargs in client.calls}
    for label in ("gen.t1", "r01.b0.verify.recovery"):
        assert sent[label][1]["use_sampling"] is False
        assert sent[label][1]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent["gen.t1"][0][-1] == {"role": "user", "content": GVR_FORCED_CANDIDATE_REMINDER}
    assert _tree(result)[(1, None)].content == "Recovered candidate \\boxed{2}"
    assert next(v for v in result.verdicts if (v.cycle, v.branch) == (1, 0)).verdict is (
        Verdict.MINOR_FIX
    )


@pytest.mark.asyncio
async def test_known_usage_errors_are_contained_and_missing_usage_is_fatal() -> None:
    failing = "r02.b2.recheck.t0"

    def contained(label: str, messages: Sequence[Mapping[str, Any]]) -> Any:
        if label == failing:
            return ChatClientError("malformed response", usage=_usage(11))
        return _default_policy(label, messages)

    request = _request()
    result = await _run(TreeClient(request, contained), request)
    assert result.status is TrajectoryStatus.CYCLE_LIMIT
    assert (3, 2) not in _tree(result)
    assert len(result.candidates) == ROUNDS * BRANCHES
    assert next(call for call in result.calls if call.label == failing).error

    def fatal(label: str, messages: Sequence[Mapping[str, Any]]) -> Any:
        if label == "r02.b1.verify":
            return MissingUsageError("connection reset")
        return _default_policy(label, messages)

    result = await _run(TreeClient(request, fatal), request)
    assert result.status is TrajectoryStatus.INVALID_USAGE


@pytest.mark.asyncio
async def test_a_budget_below_the_worst_case_is_rejected_before_any_call() -> None:
    request = _request()
    client = TreeClient(request)
    config = _config(
        total_generated_tokens=4_000_000,
        initial_generator_cap=1_000_000,
        verifier_cap=500_000,
        correction_pool=2_000_000,
    )
    result = await _run(client, request, config=config)
    assert result.status is TrajectoryStatus.PROTOCOL_ERROR
    assert client.calls == []


@pytest.mark.asyncio
async def test_resume_from_a_mid_tree_checkpoint_dispatches_only_missing_calls() -> None:
    request = _request()
    snapshots: list[TrajectoryResult] = []
    complete = await _run(
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
    resumed = await _run(resumed_client, request, resume=json.loads(json.dumps(boundary.to_dict())))

    assert {label for label, _, _ in resumed_client.calls} == {
        call.label for call in complete.calls
    } - done
    assert resumed.status is complete.status
    assert resumed.final_output == complete.final_output
    assert {key: node.content for key, node in _tree(resumed).items()} == {
        key: node.content for key, node in _tree(complete).items()
    }
    assert {(v.cycle, v.branch): v.verdict for v in resumed.verdicts} == {
        (v.cycle, v.branch): v.verdict for v in complete.verdicts
    }

    finished_client = TreeClient(request)
    await _run(finished_client, request, resume=json.loads(json.dumps(complete.to_dict())))
    assert finished_client.calls == []
