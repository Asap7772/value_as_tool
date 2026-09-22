from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from value_as_tool.budget import BudgetAccountingError, BudgetExhausted, TokenBudget
from value_as_tool.client import MissingUsageError, OpenAIChatClient, parse_chat_completion
from value_as_tool.config import load_config
from value_as_tool.orchestrator import (
    QWEN3_THINKING_BUDGET_PROCESSOR,
    QWEN35_THINK_TOKEN_IDS,
    SUBAGENT_SYSTEM_PROMPT,
    AletheiaOrchestrator,
    NonResumableTrajectoryError,
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


def _usage(completion: int = 1, prompt: int = 2) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        reasoning_tokens=completion if completion else 0,
    )


def _completion(
    content: str | None = None,
    *,
    reasoning: str | None = None,
    tool_calls: Sequence[ToolCall] = (),
    completion_tokens: int = 1,
    finish_reason: str | None = None,
) -> ChatCompletion:
    usage = _usage(completion_tokens)
    return ChatCompletion(
        id="offline-completion",
        model="Qwen/Qwen3.5-9B",
        message=AssistantMessage(
            content=content,
            reasoning=reasoning,
            tool_calls=tuple(tool_calls),
        ),
        finish_reason=finish_reason or ("tool_calls" if tool_calls else "stop"),
        usage=usage,
    )


def _tool_call(name: str, arguments: Mapping[str, Any], call_id: str = "tool-1") -> ToolCall:
    return ToolCall(
        id=call_id,
        name=name,
        arguments=json.dumps(arguments, ensure_ascii=False),
    )


def _verdict(
    outcome: str,
    *,
    critique: str = "The cited step needs attention.",
    category: str = "logic",
    excerpt: str = "",
) -> ChatCompletion:
    return _completion(
        tool_calls=[
            _tool_call(
                "submit_verdict",
                {
                    "outcome": outcome,
                    "critique": critique,
                    "fault_category": category,
                    "candidate_excerpt": excerpt,
                },
                "verdict-call",
            )
        ]
    )


def _config(**overrides: Any) -> OrchestratorConfig:
    values: dict[str, Any] = {
        "model": "offline-qwen",
        "total_generated_tokens": 120,
        "context_tokens": 160,
        "context_headroom_tokens": 8,
        "initial_generator_cap": 60,
        "verifier_cap": 10,
        "correction_pool": 50,
        "minimum_call_tokens": 1,
        "max_cycles": 3,
        "subagent_cap": 8,
        "max_subagents": 3,
        "final_candidate_reserve_tokens": 15,
        "max_tool_rounds_per_candidate": 4,
        "subagent_context_max_chars": 12,
    }
    values.update(overrides)
    return OrchestratorConfig(**values)


def _request(
    condition: Condition,
    *,
    reference: str | None = None,
) -> TrajectoryRequest:
    return TrajectoryRequest(
        benchmark="imo_proof",
        problem_id="offline-problem-1",
        problem="Prove the offline sentinel statement.",
        condition=condition,
        seed=17,
        reference_proof=reference,
        solver_prompt="Use the pinned direct prompt for the offline sentinel.",
    )


class ScriptedClient:
    def __init__(self, *responses: ChatCompletion | BaseException) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        self.calls.append(
            ([copy.deepcopy(dict(message)) for message in messages], copy.deepcopy(kwargs))
        )
        if not self.responses:
            raise AssertionError("unexpected model call")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_qwen_reasoning_fields_normalize_and_replay(field: str) -> None:
    payload = {
        "id": "completion-1",
        "model": "qwen",
        "created": 1,
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    field: "private reasoning",
                    "tool_calls": [
                        {
                            "id": "call-a",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": {"query": "x"},
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "completion_tokens_details": {"reasoning_tokens": 6},
        },
    }

    parsed = parse_chat_completion(payload)
    replay = parsed.message.to_api_dict()

    assert parsed.message.reasoning == "private reasoning"
    assert parsed.message.tool_calls[0].arguments == '{"query":"x"}'
    assert parsed.usage.total_tokens == 18
    assert parsed.usage.reasoning_tokens == 6
    assert replay == {
        "role": "assistant",
        "content": None,
        "reasoning_content": "private reasoning",
        "tool_calls": [
            {
                "id": "call-a",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"query":"x"}'},
            }
        ],
    }


def test_provider_total_is_canonicalized_from_exact_components() -> None:
    payload = {
        "id": "completion-inconsistent-total",
        "model": "qwen",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "answer"},
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 999,
        },
    }

    parsed = parse_chat_completion(payload)

    assert parsed.usage.total_tokens == 18
    assert parsed.usage.raw["total_tokens"] == 999


def test_sglang_top_level_reasoning_usage_is_normalized() -> None:
    payload = {
        "id": "completion-top-level-reasoning",
        "model": "qwen",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "answer"},
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "reasoning_tokens": 6,
        },
    }

    parsed = parse_chat_completion(payload)

    assert parsed.usage.reasoning_tokens == 6


def test_completion_without_exact_usage_is_rejected() -> None:
    with pytest.raises(MissingUsageError, match="exact usage"):
        parse_chat_completion(
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "unmetered"}}
                ]
            }
        )

    with pytest.raises(MissingUsageError, match="completion_tokens"):
        parse_chat_completion(
            {
                "choices": [{"finish_reason": "stop", "message": {"content": "partial"}}],
                "usage": {"prompt_tokens": 1},
            }
        )


@pytest.mark.asyncio
async def test_client_replays_tool_messages_and_requests_parallel_tool_calls() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "completion-2",
                "model": "qwen",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "synthesized", "reasoning": "checked"},
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 3},
            },
        )

    prior = AssistantMessage(
        content=None,
        reasoning="plan two calls",
        tool_calls=(
            _tool_call("research", {"task": "A"}, "call-a"),
            _tool_call("research", {"task": "B"}, "call-b"),
        ),
    )
    messages = [
        {"role": "user", "content": "solve"},
        prior.to_api_dict(),
        {"role": "tool", "tool_call_id": "call-a", "name": "research", "content": "A"},
        {"role": "tool", "tool_call_id": "call-b", "name": "research", "content": "B"},
    ]
    tools = [
        {
            "type": "function",
            "function": {"name": "research", "parameters": {"type": "object"}},
        }
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport_client:
        client = OpenAIChatClient("http://offline.invalid/v1", client=transport_client)
        completion = await client.complete(messages, model="qwen", max_tokens=9, tools=tools)

    assert completion.message.reasoning == "checked"
    assert len(captured) == 1
    request = captured[0]
    assert request["parallel_tool_calls"] is True
    assert request["tool_choice"] == "auto"
    assert request["messages"] == messages
    assert request["messages"][1]["reasoning_content"] == "plan two calls"
    assert [message["tool_call_id"] for message in request["messages"][2:]] == [
        "call-a",
        "call-b",
    ]


@pytest.mark.asyncio
async def test_parallel_budget_reservations_cannot_oversubscribe_global_limit() -> None:
    budget = TokenBudget(10)
    gate = asyncio.Event()

    async def reserve(label: str):  # type: ignore[no-untyped-def]
        await gate.wait()
        return budget.reserve(7, label=label)

    tasks = [asyncio.create_task(reserve("left")), asyncio.create_task(reserve("right"))]
    gate.set()
    permits = await asyncio.gather(*tasks)

    assert sorted(permit.max_tokens for permit in permits) == [3, 7]
    assert budget.reserved_generated_tokens == 10
    assert budget.remaining_generated_tokens == 0
    with pytest.raises(BudgetExhausted):
        budget.reserve(1, label="oversubscribed")

    for permit in permits:
        budget.settle(permit, _usage(permit.max_tokens, prompt=0))
    assert budget.spent_generated_tokens == 10
    assert budget.reserved_generated_tokens == 0


def test_budget_keep_is_exact_and_provider_overshoot_is_still_charged() -> None:
    budget = TokenBudget(20)
    first = budget.reserve(20, keep=7, label="candidate")
    assert first.max_tokens == 13
    with pytest.raises(BudgetExhausted):
        budget.reserve(1, keep=7, label="must-wait")
    budget.settle(first, _usage(4))

    second = budget.reserve(20, keep=7, label="correction")
    assert second.max_tokens == 9
    with pytest.raises(BudgetAccountingError, match="reported 10"):
        budget.settle(second, _usage(10))
    assert budget.spent_generated_tokens == 14
    assert budget.snapshot()["charges"][-1]["usage"]["completion_tokens"] == 10


@pytest.mark.asyncio
async def test_direct_run_preserves_reasoning_usage_and_exact_allowance() -> None:
    client = ScriptedClient(
        _completion("A direct complete proof.", reasoning="hidden derivation", completion_tokens=7)
    )
    result = await AletheiaOrchestrator(client, _config()).run(_request(Condition.DIRECT))

    assert result.status is TrajectoryStatus.COMPLETED
    assert result.final_output == "A direct complete proof."
    assert result.candidates[0].reasoning == "hidden derivation"
    assert result.usage.completion_tokens == 7
    assert result.budget["remaining_generated_tokens"] == 113
    assert len(client.calls) == 1
    messages, kwargs = client.calls[0]
    assert messages == [
        {"role": "user", "content": "Use the pinned direct prompt for the offline sentinel."}
    ]
    assert kwargs["max_tokens"] == 120
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


@pytest.mark.asyncio
async def test_direct_length_finish_is_exhausted_and_resumes_without_repeating() -> None:
    snapshots: list[TrajectoryResult] = []
    client = ScriptedClient(
        _completion("A truncated proof.", completion_tokens=120, finish_reason="length")
    )
    request = _request(Condition.DIRECT)
    result = await AletheiaOrchestrator(
        client,
        _config(),
        checkpoint=lambda value: snapshots.append(copy.deepcopy(value)),
    ).run(request)

    assert result.status is TrajectoryStatus.BUDGET_EXHAUSTED
    assert result.final_output == "A truncated proof."
    assert result.candidates[0].content == "A truncated proof."
    assert result.usage.completion_tokens == 120
    boundary = next(
        value
        for value in snapshots
        if value.status is TrajectoryStatus.RUNNING and len(value.calls) == 1
    )
    resumed_client = ScriptedClient()
    resumed = await AletheiaOrchestrator(resumed_client, _config()).run(
        request,
        resume=json.loads(json.dumps(boundary.to_dict())),
    )
    assert resumed.status is TrajectoryStatus.BUDGET_EXHAUSTED
    assert resumed.final_output == "A truncated proof."
    assert resumed.candidates[0].content == "A truncated proof."
    assert resumed_client.calls == []


@pytest.mark.asyncio
async def test_gvr_accepts_correct_candidate() -> None:
    client = ScriptedClient(
        _completion("Candidate one.", completion_tokens=5),
        _verdict("correct"),
    )
    result = await AletheiaOrchestrator(client, _config()).run(_request(Condition.GVR))

    assert result.status is TrajectoryStatus.ACCEPTED
    assert result.final_output == "Candidate one."
    assert [call.role for call in result.calls] == [Role.GENERATOR, Role.VERIFIER]
    assert [call.max_tokens for call in result.calls] == [60, 10]
    assert result.verdicts[0].verdict is Verdict.CORRECT
    assert result.transitions[-1].action == "correct"


@pytest.mark.parametrize(
    ("outcome", "next_role"),
    [("minor_fix", Role.REVISER), ("critical_flaw", Role.GENERATOR)],
)
@pytest.mark.asyncio
async def test_gvr_routes_repairs_and_major_regeneration(
    outcome: str,
    next_role: Role,
) -> None:
    client = ScriptedClient(
        _completion("Candidate one."),
        _verdict(outcome, excerpt="Candidate one."),
        _completion("Candidate two."),
        _verdict("correct"),
    )
    result = await AletheiaOrchestrator(client, _config()).run(_request(Condition.GVR))

    assert result.status is TrajectoryStatus.ACCEPTED
    assert result.final_output == "Candidate two."
    assert [candidate.role for candidate in result.candidates] == [Role.GENERATOR, next_role]
    assert [candidate.cycle for candidate in result.candidates] == [1, 2]
    assert [verdict.verdict for verdict in result.verdicts] == [
        Verdict(outcome),
        Verdict.CORRECT,
    ]
    assert result.transitions[2].target == next_role.value
    correction_prompt = result.calls[2].messages[1]["content"]
    assert "Use the pinned direct prompt for the offline sentinel." in correction_prompt
    assert "The cited step needs attention." in correction_prompt
    assert "Candidate one." in correction_prompt


@pytest.mark.asyncio
async def test_corrections_share_a_hard_pool_and_reserve_every_cycle() -> None:
    client = ScriptedClient(
        _completion("Candidate one."),
        _verdict("critical_flaw", excerpt="Candidate one."),
        _completion("Candidate two."),
        _verdict("critical_flaw", excerpt="Candidate two."),
        _completion("Candidate three."),
        _verdict("correct"),
    )

    result = await AletheiaOrchestrator(client, _config()).run(_request(Condition.GVR))

    assert result.status is TrajectoryStatus.ACCEPTED
    assert [call.max_tokens for call in result.calls] == [60, 10, 29, 10, 38, 10]
    correction_spend = sum(
        call.usage.completion_tokens
        for call in result.calls
        if call.cycle >= 2 and call.usage is not None
    )
    assert correction_spend <= 50


@pytest.mark.parametrize("outcome", ["minor_fix", "critical_flaw"])
@pytest.mark.asyncio
async def test_answer_corrections_preserve_qed_boxed_output_contract(outcome: str) -> None:
    solver_prompt = (
        "Solve the following task. Please reason step by step, and put your final "
        "answer within \\boxed{}.\nProblem: Find x."
    )
    request = TrajectoryRequest(
        benchmark="imo_answer",
        problem_id="answer-1",
        problem="Find x.",
        condition=Condition.GVR,
        seed=17,
        solver_prompt=solver_prompt,
        metadata={"golden_answer": "SECRET"},
    )
    client = ScriptedClient(
        _completion(r"Work. \boxed{1}"),
        _verdict(outcome, excerpt=r"\boxed{1}"),
        _completion(r"Corrected work. \boxed{2}"),
        _verdict("correct"),
    )

    result = await AletheiaOrchestrator(client, _config()).run(request)

    assert result.status is TrajectoryStatus.ACCEPTED
    correction_prompt = result.calls[2].messages[1]["content"]
    assert solver_prompt in correction_prompt
    assert r"\boxed{}" in correction_prompt
    assert "SECRET" not in correction_prompt


@pytest.mark.asyncio
async def test_gvr_cycle_limit_returns_latest_nonempty_candidate() -> None:
    client = ScriptedClient(_completion("Only candidate."), _verdict("critical_flaw"))
    result = await AletheiaOrchestrator(client, _config(max_cycles=1)).run(
        _request(Condition.GVR)
    )

    assert result.status is TrajectoryStatus.CYCLE_LIMIT
    assert result.final_output == "Only candidate."
    assert len(result.calls) == 2
    assert result.transitions[-1].target == "cycle_limit"


class ParallelSubagentClient:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
        self.parent_turn = 0
        self.child_active = 0
        self.maximum_child_active = 0
        self.all_children_started = asyncio.Event()

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        frozen = [copy.deepcopy(dict(message)) for message in messages]
        self.calls.append((frozen, copy.deepcopy(kwargs)))
        if messages[0]["content"] == SUBAGENT_SYSTEM_PROMPT:
            self.child_active += 1
            self.maximum_child_active = max(self.maximum_child_active, self.child_active)
            if self.child_active == 3:
                self.all_children_started.set()
            await asyncio.wait_for(self.all_children_started.wait(), timeout=1)
            focused = str(messages[1]["content"]).split("Focused task:\n", maxsplit=1)[1]
            self.child_active -= 1
            return _completion(f"finding for {focused}", completion_tokens=2)

        tools = kwargs.get("tools") or []
        tool_names = {tool["function"]["name"] for tool in tools}
        if "submit_verdict" in tool_names:
            return _verdict("correct")
        self.parent_turn += 1
        if self.parent_turn == 1:
            return _completion(
                reasoning="parent-only private plan",
                tool_calls=[
                    _tool_call(
                        "spawn_subagents",
                        {
                            "tasks": [
                                {"task": "derive lemma", "context_excerpt": "abcdefghijklmnop"},
                                {"task": "seek counterexample"},
                                {"task": "find alternate proof"},
                                {"task": "must be rejected by fanout cap"},
                            ]
                        },
                        "wave-one",
                    )
                ],
                completion_tokens=2,
            )
        if self.parent_turn == 2:
            return _completion(
                tool_calls=[
                    _tool_call(
                        "spawn_subagents",
                        {"tasks": [{"task": "forbidden second wave"}]},
                        "wave-two",
                    )
                ]
            )
        return _completion("Synthesized parent candidate.", completion_tokens=3)


@pytest.mark.asyncio
async def test_subagents_have_fresh_context_parallel_budget_fanout_and_no_depth() -> None:
    client = ParallelSubagentClient()
    config = _config()
    result = await AletheiaOrchestrator(client, config).run(
        _request(Condition.GVR_SUBAGENTS)
    )

    assert result.status is TrajectoryStatus.ACCEPTED
    assert result.final_output == "Synthesized parent candidate."
    assert len(result.subagents) == config.max_subagents == 3
    assert client.maximum_child_active == 3
    assert [len(record.context_excerpt) for record in result.subagents] == [12, 0, 0]

    child_calls = [call for call in result.calls if call.role is Role.SUBAGENT]
    assert len(child_calls) == 3
    assert all(len(call.messages) == 2 for call in child_calls)
    assert all(call.messages[0]["content"] == SUBAGENT_SYSTEM_PROMPT for call in child_calls)
    assert all(not call.tools for call in child_calls)
    assert all("parent-only private plan" not in str(call.messages) for call in child_calls)
    assert all(call.max_tokens <= config.subagent_cap for call in child_calls)
    assert sum(call.max_tokens for call in child_calls) <= (
        config.initial_generator_cap - 2 - config.final_candidate_reserve_tokens
    )

    parent_calls = [call for call in result.calls if call.role is Role.GENERATOR]
    assert len(parent_calls) == 3
    final_parent_messages = parent_calls[-1].messages
    assert any(
        message.get("reasoning_content") == "parent-only private plan"
        for message in final_parent_messages
    )
    tool_messages = [message for message in final_parent_messages if message["role"] == "tool"]
    assert len(tool_messages) == 2
    assert "only one subagent wave" in tool_messages[-1]["content"]
    assert result.budget["spent_generated_tokens"] == sum(
        call.usage.completion_tokens for call in result.calls if call.usage is not None
    )


@pytest.mark.asyncio
async def test_reference_verifier_feedback_cannot_leak_reference_only_spans() -> None:
    forbidden = "alpha beta gamma delta epsilon zeta eta theta"
    reference = f"Reference begins. {forbidden}. Reference ends."
    candidate = "Candidate local passage. This argument has a gap."
    client = ScriptedClient(
        _completion(candidate),
        _verdict(
            "minor_fix",
            critique=f"Safe diagnosis. {forbidden}. Retained warning.",
            category="logic",
            excerpt=forbidden,
        ),
        _completion("Repaired proof."),
        _verdict("correct"),
    )
    result = await AletheiaOrchestrator(client, _config()).run(
        _request(Condition.GVR_REFERENCE, reference=reference)
    )

    assert result.status is TrajectoryStatus.ACCEPTED
    assert forbidden in result.calls[1].messages[1]["content"]
    first_verdict = result.verdicts[0]
    assert forbidden not in first_verdict.critique
    assert first_verdict.critique == "Safe diagnosis. Retained warning."
    assert first_verdict.candidate_excerpt == ""
    reviser_prompt = result.calls[2].messages[1]["content"]
    assert forbidden not in reviser_prompt
    assert "Safe diagnosis" in reviser_prompt


@pytest.mark.asyncio
async def test_context_counter_clamps_completion_and_can_stop_before_dispatch() -> None:
    counts: list[tuple[int, bool]] = []

    def count_messages(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
    ) -> int:
        counts.append((len(messages), bool(tools)))
        return 100

    client = ScriptedClient(_completion("Fits the remaining context."))
    result = await AletheiaOrchestrator(
        client,
        _config(),
        count_messages=count_messages,
    ).run(_request(Condition.DIRECT))
    assert result.status is TrajectoryStatus.COMPLETED
    assert client.calls[0][1]["max_tokens"] == 52
    assert counts == [(1, False)]

    blocked_client = ScriptedClient()
    blocked = await AletheiaOrchestrator(
        blocked_client,
        _config(),
        count_messages=lambda messages, tools: 152,
    ).run(_request(Condition.DIRECT))
    assert blocked.status is TrajectoryStatus.CONTEXT_EXHAUSTED
    assert blocked_client.calls == []
    assert blocked.budget["spent_generated_tokens"] == 0


class RecordingLifecycle:
    def __init__(self, *, fail_begin: bool = False, fail_complete: bool = False) -> None:
        self.fail_begin = fail_begin
        self.fail_complete = fail_complete
        self.begun: list[tuple[str, str, Mapping[str, Any] | None]] = []
        self.completed: list[tuple[str, str, Mapping[str, Any], Mapping[str, Any]]] = []

    async def begin_request(
        self,
        request_id: str,
        *,
        role: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.begun.append((request_id, role, metadata))
        if self.fail_begin:
            raise OSError("begin unavailable")

    async def complete_request(
        self,
        request_id: str,
        *,
        state: str,
        payload: Mapping[str, Any],
        usage: Mapping[str, Any],
    ) -> None:
        self.completed.append((request_id, state, payload, usage))
        if self.fail_complete:
            raise OSError("completion unavailable")


@pytest.mark.asyncio
async def test_request_lifecycle_wraps_dispatch_and_records_exact_usage() -> None:
    lifecycle = RecordingLifecycle()
    client = ScriptedClient(_completion("Lifecycle proof.", completion_tokens=4))
    result = await AletheiaOrchestrator(
        client,
        _config(),
        request_lifecycle=lifecycle,
    ).run(_request(Condition.DIRECT))

    assert result.status is TrajectoryStatus.COMPLETED
    assert len(lifecycle.begun) == len(lifecycle.completed) == 1
    request_id, role, metadata = lifecycle.begun[0]
    assert role == "direct"
    assert metadata is not None and metadata["max_tokens"] == 120
    completed_id, state, payload, usage = lifecycle.completed[0]
    assert completed_id == request_id
    assert state == "direct.complete"
    assert payload["calls"][0]["request_id"] == request_id
    assert usage["completion_tokens"] == 4


@pytest.mark.asyncio
async def test_lifecycle_begin_failure_cancels_reservation_without_dispatch() -> None:
    lifecycle = RecordingLifecycle(fail_begin=True)
    client = ScriptedClient(_completion("must not be used"))
    result = await AletheiaOrchestrator(
        client,
        _config(),
        request_lifecycle=lifecycle,
    ).run(_request(Condition.DIRECT))

    assert result.status is TrajectoryStatus.FAILED
    assert client.calls == []
    assert result.budget["reserved_generated_tokens"] == 0
    assert result.budget["unknown_usage"] == []
    assert result.calls[0].usage is None
    assert "lifecycle begin failed" in (result.calls[0].error or "")


@pytest.mark.asyncio
async def test_missing_usage_invalidates_the_entire_trajectory() -> None:
    client = ScriptedClient(MissingUsageError("server omitted usage"))
    result = await AletheiaOrchestrator(client, _config()).run(_request(Condition.DIRECT))

    assert result.status is TrajectoryStatus.INVALID_USAGE
    assert result.final_output is None
    assert result.budget["unknown_usage_upper_bound"] == 120
    assert result.budget["remaining_generated_tokens"] == 120
    assert result.calls[0].usage is None


async def _capture_gvr_boundaries() -> tuple[
    TrajectoryRequest,
    OrchestratorConfig,
    TrajectoryResult,
    TrajectoryResult,
]:
    snapshots: list[TrajectoryResult] = []

    def checkpoint(result: TrajectoryResult) -> None:
        snapshots.append(copy.deepcopy(result))

    request = _request(Condition.GVR)
    config = _config()
    client = ScriptedClient(
        _completion("Candidate before checkpoint.", completion_tokens=3),
        _verdict("minor_fix", excerpt="Candidate before checkpoint."),
        _completion("Candidate after checkpoint.", completion_tokens=2),
        _verdict("correct"),
    )
    completed = await AletheiaOrchestrator(client, config, checkpoint=checkpoint).run(request)
    assert completed.status is TrajectoryStatus.ACCEPTED
    candidate_boundary = next(
        snapshot
        for snapshot in snapshots
        if snapshot.status is TrajectoryStatus.RUNNING
        and len(snapshot.candidates) == 1
        and len(snapshot.verdicts) == 0
        and snapshot.transitions[-1].action == "candidate"
    )
    verdict_boundary = next(
        snapshot
        for snapshot in snapshots
        if snapshot.status is TrajectoryStatus.RUNNING
        and len(snapshot.candidates) == 1
        and len(snapshot.verdicts) == 1
        and snapshot.transitions[-1].action == "minor_fix"
    )
    return request, config, candidate_boundary, verdict_boundary


@pytest.mark.asyncio
async def test_serialized_candidate_and_verdict_boundaries_resume_without_duplicate_calls() -> None:
    request, config, candidate_boundary, verdict_boundary = await _capture_gvr_boundaries()

    candidate_payload = json.loads(json.dumps(candidate_boundary.to_dict()))
    candidate_client = ScriptedClient(_verdict("correct"))
    from_candidate = await AletheiaOrchestrator(candidate_client, config).run(
        request,
        resume={"state": "candidate", "payload": candidate_payload},
    )
    assert from_candidate.status is TrajectoryStatus.ACCEPTED
    assert [call.role for call in from_candidate.calls] == [Role.GENERATOR, Role.VERIFIER]
    assert len(candidate_client.calls) == 1
    assert from_candidate.usage.completion_tokens == 4

    verdict_payload = json.loads(json.dumps(verdict_boundary.to_dict()))
    verdict_client = ScriptedClient(_completion("Resumed revision."), _verdict("correct"))
    from_verdict = await AletheiaOrchestrator(verdict_client, config).run(
        request,
        resume={"state": "verdict", "payload": verdict_payload},
    )
    assert from_verdict.status is TrajectoryStatus.ACCEPTED
    assert [call.role for call in from_verdict.calls] == [
        Role.GENERATOR,
        Role.VERIFIER,
        Role.REVISER,
        Role.VERIFIER,
    ]
    assert len(verdict_client.calls) == 2
    assert "Candidate before checkpoint." in verdict_client.calls[0][0][1]["content"]


@pytest.mark.asyncio
async def test_resume_rejects_unknown_usage_and_inflight_requests() -> None:
    request, config, candidate_boundary, _ = await _capture_gvr_boundaries()
    payload = json.loads(json.dumps(candidate_boundary.to_dict()))
    never_called = ScriptedClient()
    orchestrator = AletheiaOrchestrator(never_called, config)

    unknown = copy.deepcopy(payload)
    unknown["budget"]["unknown_usage"] = [
        {"permit_id": 9, "label": "lost", "upper_bound_tokens": 8, "reason": "crash"}
    ]
    unknown["budget"]["unknown_usage_upper_bound"] = 8
    with pytest.raises(NonResumableTrajectoryError, match="unknown provider usage"):
        await orchestrator.run(request, resume=unknown)

    with pytest.raises(NonResumableTrajectoryError, match="in-flight request"):
        await orchestrator.run(
            request,
            resume={
                "state": "dispatching",
                "payload": payload,
                "in_flight_requests": {"call-lost": {"max_tokens": 10}},
            },
        )

    assert never_called.calls == []


async def _capture_subagent_recovery_boundaries() -> tuple[
    TrajectoryRequest,
    OrchestratorConfig,
    TrajectoryResult,
    TrajectoryResult,
]:
    snapshots: list[TrajectoryResult] = []

    def checkpoint(result: TrajectoryResult) -> None:
        snapshots.append(copy.deepcopy(result))

    request = _request(Condition.GVR_SUBAGENTS)
    config = _config()
    completed = await AletheiaOrchestrator(
        ParallelSubagentClient(),
        config,
        checkpoint=checkpoint,
    ).run(request)
    assert completed.status is TrajectoryStatus.ACCEPTED
    parent_boundary = next(
        snapshot
        for snapshot in snapshots
        if not snapshot.candidates
        and len(snapshot.calls) == 1
        and snapshot.calls[0].response is not None
        and bool(snapshot.calls[0].response.message.tool_calls)
    )
    child_boundary = next(
        snapshot
        for snapshot in snapshots
        if not snapshot.candidates
        and len(snapshot.calls) == 4
        and sum(call.role is Role.SUBAGENT for call in snapshot.calls) == 3
        and not snapshot.subagents
    )
    return request, config, parent_boundary, child_boundary


@pytest.mark.asyncio
async def test_resume_replays_completed_parent_tool_call_without_repeating_it() -> None:
    request, config, parent_boundary, _ = await _capture_subagent_recovery_boundaries()
    client = ScriptedClient(
        _completion("recovered child one", completion_tokens=2),
        _completion("recovered child two", completion_tokens=2),
        _completion("recovered child three", completion_tokens=2),
        _completion("Fresh final synthesis.", completion_tokens=3),
        _verdict("correct"),
    )

    result = await AletheiaOrchestrator(client, config).run(
        request,
        resume=json.loads(json.dumps(parent_boundary.to_dict())),
    )

    assert result.status is TrajectoryStatus.ACCEPTED
    assert result.final_output == "Fresh final synthesis."
    assert len(client.calls) == 5
    assert [call.role for call in result.calls].count(Role.GENERATOR) == 2
    assert len(result.subagents) == 3


@pytest.mark.asyncio
async def test_resume_reuses_completed_child_wave_without_repeating_children() -> None:
    request, config, _, child_boundary = await _capture_subagent_recovery_boundaries()
    client = ScriptedClient(
        _completion("Synthesis from recovered children.", completion_tokens=3),
        _verdict("correct"),
    )

    result = await AletheiaOrchestrator(client, config).run(
        request,
        resume=json.loads(json.dumps(child_boundary.to_dict())),
    )

    assert result.status is TrajectoryStatus.ACCEPTED
    assert result.final_output == "Synthesis from recovered children."
    assert len(client.calls) == 2
    assert len(result.subagents) == 3
    assert len({record.call_index for record in result.subagents}) == 3
    synthesis_messages = result.calls[4].messages
    assert sum(message["role"] == "tool" for message in synthesis_messages) == 1


@pytest.mark.asyncio
async def test_candidate_requires_nonempty_post_tool_synthesis_and_uses_its_call_index() -> None:
    spawn = _tool_call(
        "spawn_subagents",
        {"tasks": [{"task": "check the provisional argument"}]},
    )
    empty_client = ScriptedClient(
        _completion("Provisional text is not final.", tool_calls=[spawn], completion_tokens=2),
        _completion("child finding", completion_tokens=2),
        _completion(None, completion_tokens=1),
    )
    failed = await AletheiaOrchestrator(empty_client, _config()).run(
        _request(Condition.GVR_SUBAGENTS)
    )

    assert failed.status is TrajectoryStatus.PROTOCOL_ERROR
    assert failed.final_output is None
    assert failed.candidates == []

    successful_client = ScriptedClient(
        _completion("Still provisional.", tool_calls=[spawn], completion_tokens=2),
        _completion("child finding", completion_tokens=2),
        _completion("Actual synthesized candidate.", completion_tokens=3),
        _verdict("correct"),
    )
    successful = await AletheiaOrchestrator(successful_client, _config()).run(
        _request(Condition.GVR_SUBAGENTS)
    )

    assert successful.status is TrajectoryStatus.ACCEPTED
    assert successful.candidates[0].content == "Actual synthesized candidate."
    assert successful.candidates[0].call_index == 2
    assert successful.calls[2].response is not None
    assert successful.calls[2].response.message.content == successful.candidates[0].content


@pytest.mark.asyncio
async def test_provisional_tool_content_is_not_returned_when_phase_budget_ends() -> None:
    client = ScriptedClient(
        _completion(
            "Provisional text must not escape.",
            tool_calls=[_tool_call("spawn_subagents", {"tasks": [{"task": "check it"}]})],
            completion_tokens=60,
        )
    )
    result = await AletheiaOrchestrator(client, _config()).run(_request(Condition.GVR_SUBAGENTS))

    assert result.status is TrajectoryStatus.BUDGET_EXHAUSTED
    assert result.final_output is None
    assert result.candidates == []
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_minimum_call_floor_prevents_tiny_budget_and_context_dispatches() -> None:
    spawn = _tool_call("spawn_subagents", {"tasks": [{"task": "check it"}]})
    budget_client = ScriptedClient(
        _completion("Provisional.", tool_calls=[spawn], completion_tokens=58)
    )
    budget_result = await AletheiaOrchestrator(
        budget_client,
        _config(minimum_call_tokens=3),
    ).run(_request(Condition.GVR_SUBAGENTS))
    assert budget_result.status is TrajectoryStatus.BUDGET_EXHAUSTED
    assert len(budget_client.calls) == 1

    context_client = ScriptedClient(_completion("must not dispatch"))
    context_result = await AletheiaOrchestrator(
        context_client,
        _config(minimum_call_tokens=5),
        count_messages=lambda messages, tools: 148,
    ).run(_request(Condition.DIRECT))
    assert context_result.status is TrajectoryStatus.CONTEXT_EXHAUSTED
    assert context_client.calls == []


def test_thinking_budget_tracks_the_granted_cap_and_reaches_the_request() -> None:
    disabled = _config()
    assert disabled.thinking_budget_for(60) is None

    enabled = _config(
        thinking_content_reserve_tokens=4,
        thinking_budget_processor=QWEN3_THINKING_BUDGET_PROCESSOR,
    )
    # The correction pool hands later cycles a smaller cap than the initial
    # generator call, so the budget has to follow the cap rather than be fixed.
    assert enabled.thinking_budget_for(60) == 56
    assert enabled.thinking_budget_for(10) == 6
    # Never below the floor that keeps a call dispatchable.
    assert enabled.thinking_budget_for(2) == enabled.minimum_call_tokens

    orchestrator = AletheiaOrchestrator(ScriptedClient(), enabled)
    extra_body = orchestrator._extra_body_for(60)
    assert extra_body["custom_params"] == {"thinking_budget": 56}
    assert extra_body["custom_logit_processor"] == QWEN3_THINKING_BUDGET_PROCESSOR
    assert extra_body["chat_template_kwargs"] == {"enable_thinking": True}
    assert "custom_params" not in AletheiaOrchestrator(
        ScriptedClient(), disabled
    )._extra_body_for(60)


@pytest.mark.asyncio
async def test_thinking_budget_is_sent_on_every_solver_call() -> None:
    client = ScriptedClient(_completion("Direct answer."))
    await AletheiaOrchestrator(
        client,
        _config(
            thinking_content_reserve_tokens=4,
            thinking_budget_processor=QWEN3_THINKING_BUDGET_PROCESSOR,
        ),
    ).run(_request(Condition.DIRECT))

    assert client.calls
    for _messages, kwargs in client.calls:
        budget = kwargs["extra_body"]["custom_params"]["thinking_budget"]
        assert budget == max(1, kwargs["max_tokens"] - 4)


def test_thinking_budget_requires_a_processor_and_leaves_room_to_think() -> None:
    with pytest.raises(ValueError, match="requires thinking_budget_processor"):
        _config(thinking_content_reserve_tokens=4)
    with pytest.raises(ValueError, match="non-negative"):
        _config(
            thinking_content_reserve_tokens=-1,
            thinking_budget_processor=QWEN3_THINKING_BUDGET_PROCESSOR,
        )
    # subagent_cap is 8 and minimum_call_tokens is 1, so a reserve of 7 would
    # leave the smallest role no thinking block at all.
    with pytest.raises(ValueError, match="no thinking room in subagent_cap"):
        _config(
            thinking_content_reserve_tokens=7,
            thinking_budget_processor=QWEN3_THINKING_BUDGET_PROCESSOR,
        )


def test_thinking_budget_processor_deserializes_and_actually_clamps() -> None:
    """The pinned payload must load the way the server loads it, and clamp.

    SGLang's own ``Qwen3ThinkingBudgetLogitProcessor`` hardcodes the Qwen3 think
    token ids, which Qwen3.5 renumbered; with the wrong ids it degrades to a
    silent no-op instead of erroring.  Exercising the logits is the only check
    that distinguishes "installed" from "working".
    """

    sglang_processor = pytest.importorskip(
        "sglang.srt.sampling.custom_logit_processor",
        reason="sglang is installed in a separate environment",
    )
    torch = pytest.importorskip("torch")

    processor = sglang_processor._cache_from_str(QWEN3_THINKING_BUDGET_PROCESSOR)()
    start, end = QWEN35_THINK_TOKEN_IDS
    assert processor.THINKING_START_TOKEN_ID == start
    assert processor.THINKING_END_TOKEN_ID == end

    class _Req:
        def __init__(self, origin: list[int], output: list[int]) -> None:
            self.origin_input_ids = origin
            self.output_ids = output

    width = end + 1
    logits = torch.zeros(1, width)
    newline = processor.NEW_LINE_TOKEN_ID

    # Thinking is open and over budget, so the next token is steered towards the
    # closing tag: first a newline, then the end-of-thinking token itself.
    over = [{"thinking_budget": 4, "__req__": _Req([start, *([42] * 32)], [42])}]
    assert int(processor(logits.clone(), over)[0].argmax()) == newline
    closing = [{"thinking_budget": 4, "__req__": _Req([start, *([42] * 32)], [newline])}]
    assert int(processor(logits.clone(), closing)[0].argmax()) == end

    # Still under budget: the processor must not touch the distribution.
    under = [{"thinking_budget": 4096, "__req__": _Req([start, 42], [42])}]
    assert torch.equal(processor(logits.clone(), under), logits)


def test_pinned_think_token_ids_match_the_solver_tokenizer() -> None:
    """Guard the exact mismatch that silently disabled the budget once already."""

    transformers = pytest.importorskip(
        "transformers", reason="tokenizer assets are only present on the cluster"
    )
    config = load_config(Path("experiment.yaml"))
    solver = config.models.solver
    model_dir = (
        Path(config.paths.asset_root)
        / "models"
        / solver.name.replace("/", "--")
        / solver.revision
    )
    if not model_dir.exists():
        pytest.skip("solver weights not downloaded in this environment")

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_dir))
    encoded = tuple(
        tokenizer.encode(tag, add_special_tokens=False)[0] for tag in ("<think>", "</think>")
    )
    assert encoded == QWEN35_THINK_TOKEN_IDS
