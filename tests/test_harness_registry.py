from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import dill
import numpy as np
import pytest

from value_as_tool.config import EvaluationConfig, ExperimentConfig
from value_as_tool.harnesses import (
    BUILTIN_HARNESSES,
    HarnessRuntime,
    HarnessSpec,
    ProofHarness,
    builtin_entrypoint_for_condition,
    load_harness,
    resolve_harness,
)
from value_as_tool.harnesses.cch_plan_work_review import (
    FIRST_REPAIR_CAP,
    INITIAL_PROOF_CAP,
    PLAN_CAP,
    REVIEW_CAP,
    SECOND_REPAIR_CAP,
    SUBMIT_PLAN_TOOL,
    SUBMIT_REVIEW_TOOL,
)
from value_as_tool.orchestrator import AletheiaOrchestrator, OrchestratorConfig
from value_as_tool.schedule import build_schedule
from value_as_tool.schemas import (
    AssistantMessage,
    ChatCompletion,
    Condition,
    Role,
    TokenUsage,
    ToolCall,
    TrajectoryRequest,
    TrajectoryStatus,
    Verdict,
)
from value_as_tool.thinking import (
    ThinkingTokenProfile,
    build_thinking_budget_processor,
    derive_thinking_token_profile,
)

CCH_ENTRYPOINT = "value_as_tool.harnesses.cch_plan_work_review:AgentHarness"


def _completion(
    content: str | None = None,
    *,
    tool_name: str | None = None,
    arguments: Mapping[str, Any] | None = None,
    reasoning: str | None = None,
) -> ChatCompletion:
    calls: tuple[ToolCall, ...] = ()
    if tool_name is not None:
        calls = (
            ToolCall(
                id=f"{tool_name}-call",
                name=tool_name,
                arguments=json.dumps(arguments, sort_keys=True),
            ),
        )
    return ChatCompletion(
        id="offline-completion",
        model="Qwen/Qwen3.8-27B",
        message=AssistantMessage(
            content=content,
            reasoning=reasoning,
            tool_calls=calls,
        ),
        finish_reason="tool_calls" if calls else "stop",
        usage=TokenUsage(
            prompt_tokens=2,
            completion_tokens=1,
            total_tokens=3,
            reasoning_tokens=1,
        ),
    )


def _plan() -> ChatCompletion:
    return _completion(
        tool_name="submit_plan",
        arguments={
            "outline": "Establish the invariant and conclude by induction.",
            "obligations": ["Prove the base case", "Verify the induction step"],
            "risks": ["Do not assume the induction hypothesis too broadly"],
        },
    )


def _review(
    decision: str,
    *,
    severity: str,
    probability: float,
    findings: Sequence[str] = (),
) -> ChatCompletion:
    return _completion(
        tool_name="submit_review",
        arguments={
            "decision": decision,
            "severity": severity,
            "success_probability": probability,
            "summary": f"Review result: {decision}.",
            "findings": list(findings),
        },
    )


class FakeChatClient:
    def __init__(self, *responses: ChatCompletion) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        self.calls.append(
            (
                [copy.deepcopy(dict(message)) for message in messages],
                copy.deepcopy(kwargs),
            )
        )
        if not self.responses:
            raise AssertionError("unexpected model call")
        return self.responses.pop(0)


def _cch_request(**overrides: Any) -> TrajectoryRequest:
    values: dict[str, Any] = {
        "benchmark": "imo_proof",
        "problem_id": "offline-cch-problem",
        "problem": "Prove the offline sentinel statement.",
        "condition": None,
        "seed": 17,
        "harness_id": "cch_plan_work_review",
    }
    values.update(overrides)
    return TrajectoryRequest(**values)


def _cch_orchestrator(client: FakeChatClient, **overrides: Any) -> AletheiaOrchestrator:
    return AletheiaOrchestrator(
        client,
        OrchestratorConfig(model="Qwen/Qwen3.8-27B", **overrides),
    )


def test_builtin_registry_resolves_nine_unique_source_hashed_harnesses() -> None:
    expected_ids = {
        "direct",
        "gvr",
        "gvr_subagents",
        "gvr_reference",
        "value_tool",
        "gvr_rationale_score",
        "value_tool_rationale_score",
        "gvr_reference_rationale_score",
        "cch_plan_work_review",
    }

    assert len(BUILTIN_HARNESSES) == 9
    specs = [resolve_harness(entrypoint) for entrypoint in BUILTIN_HARNESSES]
    assert {spec.harness_id for spec in specs} == expected_ids
    assert len({spec.source_sha256 for spec in specs}) == len(specs)

    for entrypoint, spec in zip(BUILTIN_HARNESSES, specs, strict=True):
        module_name, attribute = entrypoint.split(":", 1)
        harness_class = getattr(importlib.import_module(module_name), attribute)
        source_path = Path(inspect.getsourcefile(harness_class) or "")
        assert re.fullmatch(r"[0-9a-f]{64}", spec.source_sha256)
        assert spec.source_sha256 == hashlib.sha256(source_path.read_bytes()).hexdigest()
        instance = load_harness(entrypoint, source_sha256=spec.source_sha256)
        assert isinstance(instance, ProofHarness)
        assert instance.spec == spec

    with pytest.raises(ValueError, match="source hash changed"):
        load_harness(BUILTIN_HARNESSES[0], source_sha256="0" * 64)


def test_condition_adapters_and_reference_access_are_explicit() -> None:
    cch_spec = resolve_harness(CCH_ENTRYPOINT)
    assert cch_spec.condition is None
    assert cch_spec.access == "blind"
    assert not cch_spec.requires_reference

    for condition in Condition:
        entrypoint = builtin_entrypoint_for_condition(condition.value)
        spec = resolve_harness(entrypoint)
        assert spec.condition is condition
        assert spec.harness_id == condition.value

    for harness_id in ("gvr_reference", "gvr_reference_rationale_score"):
        spec = next(
            spec
            for spec in map(resolve_harness, BUILTIN_HARNESSES)
            if spec.harness_id == harness_id
        )
        assert spec.access == "reference_assisted"
        assert spec.requires_reference


def test_nine_harness_schedule_has_exact_expected_experiment_size() -> None:
    problems = {
        "imo_proof": [
            {
                "id": f"imo-{index}",
                "problem": f"IMO proof problem {index}",
                "solution": "A reference proof.",
            }
            for index in range(60)
        ],
        "proofbench": [
            {
                "id": f"proofbench-{index}",
                "problem": f"ProofBench problem {index}",
                "solution": "A reference proof.",
            }
            for index in range(145)
        ],
    }
    config = ExperimentConfig(
        evaluation=EvaluationConfig(
            harnesses=BUILTIN_HARNESSES,
            benchmarks=("imo_proof", "proofbench"),
            seeds=(0, 1, 2),
        )
    )

    schedule = build_schedule(problems, config)

    assert len(schedule) == 5_535
    assert Counter(item.harness_id for item in schedule) == {
        harness_id: 615
        for harness_id in (
            "direct",
            "gvr",
            "gvr_subagents",
            "gvr_reference",
            "value_tool",
            "gvr_rationale_score",
            "value_tool_rationale_score",
            "gvr_reference_rationale_score",
            "cch_plan_work_review",
        )
    }
    assert all(item.harness_source_sha256 for item in schedule)
    assert all(item.harness_access in {"blind", "reference_assisted"} for item in schedule)
    assert all(
        item.condition is None
        for item in schedule
        if item.harness_id == "cch_plan_work_review"
    )


@pytest.mark.asyncio
async def test_cch_approve_path_uses_fresh_review_context_and_forced_tools() -> None:
    client = FakeChatClient(
        _plan(),
        _completion("A complete initial proof.", reasoning="private worker reasoning"),
        _review("approve", severity="none", probability=0.91),
    )
    harness = load_harness(CCH_ENTRYPOINT)

    result = await _cch_orchestrator(client).run(_cch_request(), harness=harness)

    assert result.status is TrajectoryStatus.ACCEPTED
    assert result.final_output == "A complete initial proof."
    assert [call.label for call in result.calls] == [
        "cch.plan",
        "cch.work.0",
        "cch.review.0",
    ]
    assert [call.role for call in result.calls] == [
        Role.PLANNER,
        Role.WORKER,
        Role.REVIEWER,
    ]
    assert [call.max_tokens for call in result.calls] == [
        PLAN_CAP,
        INITIAL_PROOF_CAP,
        REVIEW_CAP,
    ]
    assert result.candidates[0].reasoning == "private worker reasoning"
    assert result.verdicts[0].verdict is Verdict.CORRECT
    assert result.verdicts[0].success_probability == pytest.approx(0.91)
    assert result.transitions[-1].action == "approve"

    assert client.calls[0][1]["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_plan"},
    }
    assert client.calls[2][1]["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_review"},
    }
    review_context = json.dumps(client.calls[2][0], sort_keys=True)
    assert "A complete initial proof." in review_context
    assert "Approved plan" not in review_context
    assert "private worker reasoning" not in review_context


@pytest.mark.asyncio
async def test_cch_normalizes_qwen_stringified_array_arguments() -> None:
    client = FakeChatClient(
        _completion(
            tool_name="submit_plan",
            arguments={
                "outline": "Prove the claim directly.",
                "obligations": json.dumps(["Check the only case."]),
                "risks": json.dumps(["Do not skip the definition."]),
            },
        ),
        _completion("A complete proof."),
        _completion(
            tool_name="submit_review",
            arguments={
                "decision": "approve",
                "severity": "none",
                "success_probability": 0.99,
                "summary": "The proof is complete.",
                "findings": json.dumps([]),
            },
        ),
    )

    result = await _cch_orchestrator(client).run(
        _cch_request(),
        harness=load_harness(CCH_ENTRYPOINT),
    )

    assert result.status is TrajectoryStatus.ACCEPTED
    assert result.final_output == "A complete proof."


@pytest.mark.asyncio
async def test_cch_recovers_qwen_folded_risks_marker() -> None:
    client = FakeChatClient(
        _completion(
            tool_name="submit_plan",
            arguments={
                "outline": "Prove the claim directly.",
                "obligations": (
                    '["Check the only case."],\n<risks>\n'
                    '["Do not skip the definition."]'
                ),
            },
        ),
        _completion("A complete proof."),
        _completion(
            tool_name="submit_review",
            arguments={
                "decision": " APPROVE ",
                "severity": " NONE ",
                "success_probability": "0.99",
                "summary": "The proof is complete.",
                "findings": "No findings",
            },
        ),
    )

    result = await _cch_orchestrator(client).run(
        _cch_request(),
        harness=load_harness(CCH_ENTRYPOINT),
    )

    assert result.status is TrajectoryStatus.ACCEPTED
    assert result.final_output == "A complete proof."
    assert SUBMIT_PLAN_TOOL["function"]["parameters"]["properties"][
        "obligations"
    ]["type"] == "string"
    assert SUBMIT_REVIEW_TOOL["function"]["parameters"]["properties"]["findings"][
        "type"
    ] == "string"


@pytest.mark.asyncio
async def test_cch_accepts_usable_plan_truncated_before_optional_metadata() -> None:
    client = FakeChatClient(
        _completion(
            tool_name="submit_plan",
            arguments={
                "outline": "A complete usable outline produced before the token cap.",
                "obligations": "Check the main lemma.",
            },
        ),
        _completion("A complete proof."),
        _completion(
            tool_name="submit_review",
            arguments={
                "decision": "request_changes",
                "severity": "minor",
                "success_probability": 0.6,
                "summary": "Clarify the equality case.",
            },
        ),
        _completion("A repaired proof."),
        _review("approve", severity="none", probability=0.95),
    )

    result = await _cch_orchestrator(client).run(
        _cch_request(),
        harness=load_harness(CCH_ENTRYPOINT),
    )

    assert result.status is TrajectoryStatus.ACCEPTED
    assert result.final_output == "A repaired proof."
    assert result.verdicts[0].critique == "Clarify the equality case."


@pytest.mark.asyncio
@pytest.mark.parametrize("stage_tokens", [None, 262_144])
async def test_cch_two_repairs_preserve_stage_allowances_and_resume(
    stage_tokens: int | None,
) -> None:
    client = FakeChatClient(
        _plan(),
        _completion("Initial proof."),
        _review(
            "request_changes",
            severity="minor",
            probability=0.45,
            findings=("Justify the base case.",),
        ),
        _completion("First repaired proof."),
        _review(
            "request_changes",
            severity="major",
            probability=0.25,
            findings=("The induction step is circular.",),
        ),
        _completion("Second repaired proof."),
        _review(
            "request_changes",
            severity="minor",
            probability=0.60,
            findings=("Handle the equality case.",),
        ),
    )

    expected_caps = (
        [stage_tokens] * 7
        if stage_tokens is not None
        else [
            PLAN_CAP,
            INITIAL_PROOF_CAP,
            REVIEW_CAP,
            FIRST_REPAIR_CAP,
            REVIEW_CAP,
            SECOND_REPAIR_CAP,
            REVIEW_CAP,
        ]
    )
    # Charge substantial usage so cumulative spending, reservation release,
    # and checkpoint replay exercise the shared ledger across all stages.
    client.responses = [
        replace(
            response,
            usage=TokenUsage(
                prompt_tokens=2,
                completion_tokens=min(cap, 100_000),
                total_tokens=2 + min(cap, 100_000),
                reasoning_tokens=min(cap, 100_000) - 1,
            ),
        )
        for response, cap in zip(client.responses, expected_caps, strict=True)
    ]
    resume_responses = copy.deepcopy(client.responses[-2:])
    snapshots = []
    requested_budgets = []

    class RecordingOrchestrator(AletheiaOrchestrator):
        async def _call(self, *args: Any, **kwargs: Any) -> tuple[ChatCompletion, int]:
            requested_budgets.append((kwargs["cap"], kwargs["keep"]))
            return await super()._call(*args, **kwargs)

    config = OrchestratorConfig(
        cch_stage_tokens=stage_tokens,
        total_generated_tokens=8_388_608 if stage_tokens is not None else 229_376,
    )
    orchestrator = RecordingOrchestrator(
        client,
        config,
        checkpoint=lambda snapshot: snapshots.append(copy.deepcopy(snapshot)),
        count_messages=lambda messages, tools: 2,
    )
    result = await orchestrator.run(
        _cch_request(),
        harness=load_harness(CCH_ENTRYPOINT),
    )

    assert result.status is TrajectoryStatus.CYCLE_LIMIT
    assert result.final_output == "Second repaired proof."
    assert [call.label for call in result.calls] == [
        "cch.plan",
        "cch.work.0",
        "cch.review.0",
        "cch.work.1",
        "cch.review.1",
        "cch.work.2",
        "cch.review.2",
    ]
    assert requested_budgets == [
        (cap, sum(expected_caps[index + 1 :]))
        for index, cap in enumerate(expected_caps)
    ]
    context_cap = config.context_tokens - config.context_headroom_tokens - 2
    assert [call.max_tokens for call in result.calls] == [
        min(cap, context_cap) for cap in expected_caps
    ]
    assert result.usage.completion_tokens == sum(min(cap, 100_000) for cap in expected_caps)
    assert result.budget["reserved_generated_tokens"] == 0
    if stage_tokens is not None:
        assert result.usage.completion_tokens > config.context_tokens
    else:
        assert sum(call.max_tokens for call in result.calls) == 229_376
    assert [candidate.cycle for candidate in result.candidates] == [0, 1, 2]
    assert [verdict.verdict for verdict in result.verdicts] == [
        Verdict.MINOR_FIX,
        Verdict.CRITICAL_FLAW,
        Verdict.MINOR_FIX,
    ]
    assert [transition.action for transition in result.transitions].count("retake") == 2
    assert result.transitions[-1].action == "retake_limit"

    checkpoint = next(
        snapshot
        for snapshot in snapshots
        if len(snapshot.calls) == 5 and len(snapshot.verdicts) == 2
    )
    replay_client = FakeChatClient(*resume_responses)
    replayed = await AletheiaOrchestrator(
        replay_client,
        config,
        count_messages=lambda messages, tools: 2,
    ).run(_cch_request(), resume=checkpoint, harness=load_harness(CCH_ENTRYPOINT))

    assert len(replay_client.calls) == 2
    assert replayed.status is result.status
    assert replayed.final_output == result.final_output
    assert replayed.usage == result.usage
    assert replayed.budget == result.budget
    assert replayed.calls == result.calls
    assert replayed.candidates == result.candidates
    assert replayed.verdicts == result.verdicts
    assert replayed.transitions == result.transitions


@pytest.mark.asyncio
@pytest.mark.parametrize("stage_tokens", [None, 262_144])
async def test_cch_completed_calls_replay_without_redispatch(stage_tokens: int | None) -> None:
    first_client = FakeChatClient(
        _plan(),
        _completion("Replayable proof."),
        _review("approve", severity="none", probability=0.88),
    )
    harness = load_harness(CCH_ENTRYPOINT)
    options = {
        "cch_stage_tokens": stage_tokens,
        "total_generated_tokens": 8_388_608 if stage_tokens is not None else 229_376,
    }
    first = await _cch_orchestrator(first_client, **options).run(
        _cch_request(), harness=harness
    )
    checkpoint = copy.deepcopy(first)
    checkpoint.status = TrajectoryStatus.RUNNING
    checkpoint.final_output = None

    replay_client = FakeChatClient()
    replayed = await _cch_orchestrator(replay_client, **options).run(
        _cch_request(),
        resume=checkpoint,
        harness=load_harness(CCH_ENTRYPOINT, source_sha256=harness.spec.source_sha256),
    )

    assert not replay_client.calls
    assert replayed.status is TrajectoryStatus.ACCEPTED
    assert replayed.final_output == "Replayable proof."
    assert replayed.calls == first.calls
    assert replayed.candidates == first.candidates
    assert replayed.verdicts == first.verdicts
    assert replayed.transitions == first.transitions


@pytest.mark.asyncio
async def test_blind_harness_runtime_redacts_reference_and_sensitive_metadata() -> None:
    class CaptureHarness:
        spec = HarnessSpec("capture_blind", "Capture blind runtime")

        def __init__(self) -> None:
            self.seen: TrajectoryRequest | None = None

        async def run(self, runtime: HarnessRuntime) -> None:
            self.seen = runtime.request
            await runtime.finish(
                "A proof independent of the reference.",
                status=TrajectoryStatus.COMPLETED,
            )

    harness = CaptureHarness()
    request = TrajectoryRequest(
        benchmark="imo_proof",
        problem_id="redaction-test",
        problem="Prove P.",
        condition=None,
        seed=3,
        harness_id=harness.spec.harness_id,
        reference_proof="TOP SECRET REFERENCE",
        metadata={
            "answer_key": "TOP SECRET ANSWER",
            "grading_notes": "TOP SECRET GRADING",
            "reference_uri": "TOP SECRET URI",
            "rubric": "TOP SECRET RUBRIC",
            "solution_notes": "TOP SECRET SOLUTION",
            "public_tag": "safe",
        },
    )

    result = await AletheiaOrchestrator(FakeChatClient()).run(request, harness=harness)

    assert result.status is TrajectoryStatus.COMPLETED
    assert harness.seen is not None
    assert harness.seen.reference_proof is None
    assert harness.seen.metadata == {"public_tag": "safe"}
    # The immutable persisted request remains the full evaluator-owned record.
    assert result.request.reference_proof == "TOP SECRET REFERENCE"
    assert result.request.metadata["rubric"] == "TOP SECRET RUBRIC"


class _FakeTokenizer:
    token_ids = {"<think>": 701, "</think>": 702, "\n": 17}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [self.token_ids[text]]

    def decode(
        self,
        values: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        reverse = {token_id: text for text, token_id in self.token_ids.items()}
        return "".join(reverse[value] for value in values)


def test_dynamic_thinking_profile_builds_a_working_model_specific_processor() -> None:
    profile = derive_thinking_token_profile(_FakeTokenizer())
    assert profile == ThinkingTokenProfile(701, 702, 17)

    serialized = build_thinking_budget_processor(profile)
    assert serialized == build_thinking_budget_processor(profile)
    processor_type = dill.loads(bytes.fromhex(json.loads(serialized)["callable"]))
    processor = processor_type()
    assert (
        processor.THINKING_START_TOKEN_ID,
        processor.THINKING_END_TOKEN_ID,
        processor.NEW_LINE_TOKEN_ID,
    ) == (701, 702, 17)

    class Request:
        origin_input_ids = [701]
        output_ids = [50, 51]

    logits = np.zeros((1, 800))
    clamped = processor(logits, [{"thinking_budget": 2, "__req__": Request()}])
    assert clamped[0, 17] == 0
    assert np.isneginf(np.delete(clamped[0], 17)).all()

    Request.output_ids = [50, 17]
    logits = np.zeros((1, 800))
    clamped = processor(logits, [{"thinking_budget": 2, "__req__": Request()}])
    assert clamped[0, 702] == 0
    assert np.isneginf(np.delete(clamped[0], 702)).all()


def test_dynamic_thinking_profile_rejects_non_atomic_control_tokens() -> None:
    class SplitTokenizer(_FakeTokenizer):
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            if text == "<think>":
                return [70, 1]
            return super().encode(text, add_special_tokens=add_special_tokens)

    with pytest.raises(ValueError, match="does not encode.*as one token"):
        derive_thinking_token_profile(SplitTokenizer())
