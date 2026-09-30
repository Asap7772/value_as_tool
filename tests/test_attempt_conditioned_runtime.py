from __future__ import annotations

import copy
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from value_as_tool.conditioning import _pack_content
from value_as_tool.harnesses import (
    ATTEMPT_CONDITIONED_HARNESSES,
    HarnessRuntime,
    HarnessSpec,
    load_harness,
    resolve_harness,
)
from value_as_tool.orchestrator import (
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
)

EVIDENCE = "attempt-0 incorrect: amber birch cedar dogwood elm fir ginkgo hemlock."
EVIDENCE_SPAN = "amber birch cedar dogwood elm fir ginkgo hemlock"
GOLD = "Reference: azure bronze copper diamond emerald fuchsia garnet hazel."
GOLD_SPAN = "azure bronze copper diamond emerald fuchsia garnet hazel"
CANDIDATE = "The candidate omits the boundary case."


def _completion(
    content: str | None = None, *, tool: str | None = None, arguments: dict[str, Any] | None = None
) -> ChatCompletion:
    return ChatCompletion(
        id="offline-response",
        model="Qwen/Qwen3.5-9B",
        message=AssistantMessage(
            content=content,
            reasoning="Solver's own intermediate work."
            if tool == "query_success_probability"
            else None,
            tool_calls=(ToolCall("tool-call", tool, json.dumps(arguments or {})),) if tool else (),
        ),
        finish_reason="tool_calls" if tool else "stop",
        usage=TokenUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
    )


def _verdict(
    rationale: bool, *, outcome: str = "correct", feedback: str = "The current candidate is sound."
) -> ChatCompletion:
    arguments: dict[str, Any] = {
        "outcome": outcome,
        "fault_category": "logic",
        "candidate_excerpt": CANDIDATE,
    }
    if rationale:
        arguments.update(success_probability=0.8, rationale=feedback)
    else:
        arguments["critique"] = feedback
    return _completion(tool="submit_verdict", arguments=arguments)


class ScriptedClient:
    def __init__(self, *responses: ChatCompletion) -> None:
        self.responses = list(responses)
        self.calls: list[Any] = []

    async def complete(self, messages: Any, **kwargs: Any) -> ChatCompletion:
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        assert self.responses, "unexpected model call"
        return self.responses.pop(0)


def _config() -> OrchestratorConfig:
    return OrchestratorConfig(
        model="offline-qwen",
        total_generated_tokens=120,
        context_tokens=160,
        context_headroom_tokens=8,
        initial_generator_cap=60,
        verifier_cap=10,
        correction_pool=50,
        minimum_call_tokens=1,
        max_cycles=3,
        subagent_cap=8,
        final_candidate_reserve_tokens=15,
        value_tool_max_queries=3,
        value_verifier_cap=10,
        value_final_response_reserve_tokens=15,
    )


def _harness(
    interaction: str = "gvr", feedback: str = "legacy", gold: bool = False, mode: str = "solutions"
) -> Any:
    suffix = "gold" if gold else "no_gold"
    return load_harness(
        f"value_as_tool.harnesses.attempt_conditioned:"
        f"attempt_{mode}_{interaction}_{feedback}_{suffix}"
    )


def _request(harness: Any) -> TrajectoryRequest:
    return TrajectoryRequest(
        benchmark="imo_proof",
        problem_id="attempt-test",
        problem="Prove this statement.",
        condition=harness.spec.condition,
        seed=8,
        harness_id=harness.spec.harness_id,
        solver_prompt="Original solver prompt.",
        reference_proof=GOLD,
        verifier_evidence={
            "mode": harness.spec.conditioning_mode,
            "pack_sha256": "a" * 64,
            "content": EVIDENCE,
        },
    )


def test_registry_has_exactly_the_twenty_four_factor_combinations() -> None:
    specs = [resolve_harness(entrypoint) for entrypoint in ATTEMPT_CONDITIONED_HARNESSES]
    assert len(specs) == len({spec.harness_id for spec in specs}) == 24
    assert {
        (spec.conditioning_mode, spec.condition, spec.requires_reference) for spec in specs
    } == {
        (mode, condition, gold)
        for mode in ("solutions", "solution_summary", "thinking_summary")
        for condition in (
            Condition.GVR,
            Condition.GVR_RATIONALE_SCORE,
            Condition.VALUE_TOOL,
            Condition.VALUE_TOOL_RATIONALE_SCORE,
        )
        for gold in (False, True)
    }
    for spec in specs:
        assert spec.source_sha256 and spec.entrypoint
        assert spec.access == (
            "attempt_and_reference_assisted" if spec.requires_reference else "attempt_assisted"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ATTEMPT_CONDITIONED_HARNESSES)
async def test_all_arms_expose_evidence_and_gold_only_to_the_verifier(entrypoint: str) -> None:
    harness = load_harness(entrypoint)
    request = _request(harness)
    rationale = harness.spec.condition in {
        Condition.GVR_RATIONALE_SCORE,
        Condition.VALUE_TOOL_RATIONALE_SCORE,
    }
    value_tool = harness.spec.condition in {
        Condition.VALUE_TOOL,
        Condition.VALUE_TOOL_RATIONALE_SCORE,
    }
    if value_tool:
        arguments: dict[str, Any] = {"success_probability": 0.8}
        if rationale:
            arguments["rationale"] = "The current trace is promising."
        responses = (
            _completion(tool="query_success_probability"),
            _completion(tool="submit_probability", arguments=arguments),
            _completion(CANDIDATE),
        )
    else:
        responses = (_completion(CANDIDATE), _verdict(rationale))
    client = ScriptedClient(*responses)
    result = await AletheiaOrchestrator(client, _config()).run(request, harness=harness)
    assert result.status in {TrajectoryStatus.ACCEPTED, TrajectoryStatus.COMPLETED}
    assert len(result.calls) == len(responses)
    for call in result.calls:
        prompt = json.dumps(call.messages)
        if call.role in {Role.VERIFIER, Role.VALUE_VERIFIER}:
            assert EVIDENCE in prompt
            assert request.verifier_evidence["pack_sha256"] in prompt
            assert (GOLD in prompt) is harness.spec.requires_reference
        else:
            assert EVIDENCE not in prompt
            assert GOLD not in prompt
    # The initial solver request and seed stay identical to the existing
    # corresponding protocol, so conditioning cannot alter its initial draw.
    baseline_request = replace(request, harness_id=None, verifier_evidence=None)
    baseline = await AletheiaOrchestrator(ScriptedClient(*responses), _config()).run(
        baseline_request
    )
    assert result.calls[0].messages == baseline.calls[0].messages
    assert result.calls[0].seed == baseline.calls[0].seed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "replacement",
    [
        None,
        {},
        {"mode": "thinking_summary", "pack_sha256": "a" * 64, "content": EVIDENCE},
        {"mode": "solutions", "pack_sha256": "invalid", "content": EVIDENCE},
        {"mode": "solutions", "pack_sha256": "a" * 64, "content": "  "},
    ],
)
async def test_invalid_evidence_fails_before_any_model_call(replacement: Any) -> None:
    harness = _harness()
    client = ScriptedClient()
    result = await AletheiaOrchestrator(client, _config()).run(
        replace(_request(harness), verifier_evidence=replacement), harness=harness
    )
    assert result.status is TrajectoryStatus.FAILED
    assert "verifier_evidence" in (result.error or "")
    assert not client.calls


@pytest.mark.asyncio
async def test_gold_arm_requires_nonempty_reference_before_any_model_call() -> None:
    harness = _harness(gold=True)
    client = ScriptedClient()
    result = await AletheiaOrchestrator(client, _config()).run(
        replace(_request(harness), reference_proof="  "), harness=harness
    )
    assert result.status is TrajectoryStatus.FAILED
    assert "reference_proof" in (result.error or "")
    assert not client.calls


@pytest.mark.parametrize("gold", [False, True])
def test_runtime_facade_hides_evidence_and_enforces_reference_access(gold: bool) -> None:
    harness = _harness(gold=gold)
    request = replace(
        _request(harness),
        metadata={"conditioning_content": EVIDENCE, "reference_answer": GOLD, "safe": "metadata"},
    )
    runtime = HarnessRuntime(None, SimpleNamespace(result=TrajectoryResult(request)), harness.spec)
    assert runtime.request.verifier_evidence is None
    assert runtime.request.reference_proof == (GOLD if gold else None)
    if not gold:
        assert runtime.request.metadata == {"safe": "metadata"}


def test_request_and_result_serialization_preserve_old_shape_and_new_evidence() -> None:
    request = _request(_harness())
    assert TrajectoryRequest.from_dict(request.to_dict()) == request
    result = TrajectoryResult(request=request)
    assert TrajectoryResult.from_dict(result.to_dict()).request == request
    legacy = replace(request, verifier_evidence=None)
    assert "verifier_evidence" not in legacy.to_dict()
    assert "verifier_evidence" not in TrajectoryResult(legacy).to_dict()["request"]
    with pytest.raises(ValueError, match="verifier_evidence"):
        TrajectoryRequest.from_dict({**request.to_dict(), "verifier_evidence": "not an object"})


@pytest.mark.asyncio
@pytest.mark.parametrize("feedback", ["legacy", "rationale"])
async def test_gvr_copy_guard_covers_attempt_evidence_and_gold(feedback: str) -> None:
    harness = _harness(feedback=feedback, gold=True)
    rationale = feedback == "rationale"
    copied = f"Safe diagnosis. {EVIDENCE_SPAN}. {GOLD_SPAN}. Retained warning."
    result = await AletheiaOrchestrator(
        ScriptedClient(
            _completion(CANDIDATE),
            _verdict(rationale, outcome="minor_fix", feedback=copied),
            _completion("Repaired proof."),
            _verdict(rationale),
        ),
        _config(),
    ).run(_request(harness), harness=harness)
    assert result.status is TrajectoryStatus.ACCEPTED
    verdict = result.verdicts[0]
    scrubbed = verdict.rationale if rationale else verdict.critique
    assert scrubbed == "Safe diagnosis. Retained warning."
    reviser_prompt = json.dumps(result.calls[2].messages)
    assert EVIDENCE_SPAN not in reviser_prompt and GOLD_SPAN not in reviser_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["solutions", "solution_summary", "thinking_summary"])
@pytest.mark.parametrize(
    "interaction,feedback", [("gvr", "legacy"), ("gvr", "rationale"), ("value_tool", "rationale")]
)
async def test_json_pack_copy_guard_decodes_multiline_math_and_preserves_candidate_quotes(
    mode: str, interaction: str, feedback: str
) -> None:
    harness = _harness(interaction, feedback, mode=mode)
    private = (
        "The bound for \\alpha follows from\n"
        "applying \\beta to the remaining terms."
    )
    quoted_candidate = (
        r"Every admissible \gamma satisfies the stated boundary condition by construction."
    )
    material = f"{private}\n{quoted_candidate}"
    pack = _pack_content(
        mode,
        {
            "attempts": [
                {"attempt_id": "prior-0", "seed": 0, "correct": False, "solution": material}
            ]
        },
        summary=material if mode != "solutions" else None,
    )
    request = replace(
        _request(harness),
        verifier_evidence={"mode": mode, "pack_sha256": "a" * 64, "content": pack},
    )
    private_copy = " ".join(private.split())
    copied = f"Safe diagnosis. {private_copy} {quoted_candidate} Retained warning."
    candidate = f"{quoted_candidate} {CANDIDATE}"
    if interaction == "gvr":
        responses = (
            _completion(candidate),
            _verdict(feedback == "rationale", outcome="minor_fix", feedback=copied),
            _completion("Repaired proof."),
            _verdict(feedback == "rationale"),
        )
    else:
        responses = (
            _completion(candidate, tool="query_success_probability"),
            _completion(
                tool="submit_probability",
                arguments={"success_probability": 0.4, "rationale": copied},
            ),
            _completion("Repaired proof."),
        )
    result = await AletheiaOrchestrator(ScriptedClient(*responses), _config()).run(
        request, harness=harness
    )
    assert result.status in {TrajectoryStatus.ACCEPTED, TrajectoryStatus.COMPLETED}
    if interaction == "gvr":
        verdict = result.verdicts[0]
        scrubbed = verdict.rationale if feedback == "rationale" else verdict.critique
    else:
        scrubbed = result.value_estimates[0].rationale
    assert scrubbed == f"Safe diagnosis. {quoted_candidate} Retained warning."
    assert all(private_copy not in message["content"] for message in result.calls[2].messages)


@pytest.mark.asyncio
async def test_value_rationale_scrubbing_replays_exactly_and_binds_pack_identity() -> None:
    harness = _harness("value_tool", "rationale", True)
    request = _request(harness)
    snapshots: list[TrajectoryResult] = []
    query = _completion(tool="query_success_probability")
    probability = _completion(
        tool="submit_probability",
        arguments={
            "success_probability": 0.4,
            "rationale": f"{EVIDENCE_SPAN}. {GOLD_SPAN}.",
        },
    )
    final = _completion(CANDIDATE)
    complete = await AletheiaOrchestrator(
        ScriptedClient(query, probability, final),
        _config(),
        checkpoint=lambda value: snapshots.append(copy.deepcopy(value)),
    ).run(request, harness=harness)
    boundary = next(
        value for value in snapshots if len(value.calls) == 2 and not value.value_estimates
    )
    resumed_client = ScriptedClient(final)
    resumed = await AletheiaOrchestrator(resumed_client, _config()).run(
        request,
        harness=harness,
        resume=boundary.to_dict(),
    )
    assert resumed.to_dict() == complete.to_dict()
    assert len(resumed_client.calls) == 1
    feedback = json.loads(complete.calls[-1].messages[-1]["content"])
    assert feedback["rationale"]
    assert EVIDENCE_SPAN not in feedback["rationale"]
    assert GOLD_SPAN not in feedback["rationale"]
    with pytest.raises(NonResumableTrajectoryError, match="identity"):
        await AletheiaOrchestrator(ScriptedClient(), _config()).run(
            replace(
                request, verifier_evidence={**request.verifier_evidence, "pack_sha256": "b" * 64}
            ),
            harness=harness,
            resume=boundary,
        )


@pytest.mark.asyncio
async def test_gvr_recovery_uses_same_full_evidence_and_replays_checkpoint() -> None:
    harness = _harness(gold=True)
    request = _request(harness)
    snapshots: list[TrajectoryResult] = []
    valid = _verdict(False)
    complete = await AletheiaOrchestrator(
        ScriptedClient(_completion(CANDIDATE), _completion("Invalid verdict format"), valid),
        _config(),
        checkpoint=lambda value: snapshots.append(copy.deepcopy(value)),
    ).run(request, harness=harness)
    assert complete.status is TrajectoryStatus.ACCEPTED
    assert complete.calls[1].messages[1] == complete.calls[2].messages[1]
    boundary = next(value for value in snapshots if len(value.calls) == 2 and not value.verdicts)
    resumed_client = ScriptedClient(valid)
    resumed = await AletheiaOrchestrator(resumed_client, _config()).run(
        request,
        harness=harness,
        resume=boundary.to_dict(),
    )
    assert resumed.to_dict() == complete.to_dict()
    assert len(resumed_client.calls) == 1


def test_spec_rejects_inconsistent_access_or_unknown_conditioning() -> None:
    with pytest.raises(ValueError, match="conditioning_mode"):
        HarnessSpec("bad", "Bad", access="attempt_assisted")
    with pytest.raises(ValueError, match="unsupported conditioning"):
        HarnessSpec("bad", "Bad", access="attempt_assisted", conditioning_mode="unknown")
    with pytest.raises(ValueError, match="requires_reference"):
        HarnessSpec(
            "bad", "Bad", access="attempt_and_reference_assisted", conditioning_mode="solutions"
        )


@pytest.mark.asyncio
async def test_evidence_cannot_be_silently_ignored_without_the_loaded_harness() -> None:
    client = ScriptedClient()
    with pytest.raises(ValueError, match="explicitly loaded conditioned harness"):
        await AletheiaOrchestrator(client, _config()).run(_request(_harness()))
    assert not client.calls


@pytest.mark.asyncio
async def test_oversized_evidence_stops_before_verifier_dispatch_without_clipping() -> None:
    harness = _harness()
    counted: list[str] = []

    def count_messages(messages: Any, tools: Any = None) -> int:
        del tools
        prompt = json.dumps(messages)
        counted.append(prompt)
        return 161 if EVIDENCE in prompt else 2

    client = ScriptedClient(_completion(CANDIDATE))
    result = await AletheiaOrchestrator(
        client, _config(), count_messages=count_messages
    ).run(_request(harness), harness=harness)
    assert result.status is TrajectoryStatus.CONTEXT_EXHAUSTED
    assert result.final_output == CANDIDATE
    assert len(client.calls) == 1
    assert any(EVIDENCE in prompt for prompt in counted)
