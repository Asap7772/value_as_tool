from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from value_as_tool.tokenization import HuggingFaceTokenCounter


@pytest.fixture
def audit_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_budget_run.py"
    spec = importlib.util.spec_from_file_location("audit_budget_run", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Schedule(list):
        fingerprint = "schedule-fingerprint"

    class Counter:
        def __init__(self, path: str, *, enable_thinking: bool) -> None:
            assert path == "pinned-model"
            assert enable_thinking is True

        def count_messages(self, messages: Any, tools: Any) -> int:
            del messages, tools
            return 128

    context = SimpleNamespace(
        artifact_root=tmp_path,
        config_fingerprint="config-fingerprint",
        config=SimpleNamespace(
            sampling=SimpleNamespace(enable_thinking=True, thinking_content_reserve_tokens=0),
            evaluation=SimpleNamespace(seeds=(0,)),
            models=SimpleNamespace(solver=SimpleNamespace(name="Qwen/Qwen3.5-9B")),
            budget=SimpleNamespace(
                generated_tokens=1_000_000, context_tokens=4_096, context_headroom_tokens=1_024
            ),
        ),
    )
    schedule = Schedule(
        [SimpleNamespace(ordinal=0, run_id="run-test", harness_id="direct", benchmark="imo_proof")]
    )
    monkeypatch.setattr(module, "load_context", lambda _: context)
    monkeypatch.setattr(module, "_load_schedule", lambda _: schedule)
    monkeypatch.setattr(module, "_model_entry", lambda *_: {"path": "pinned-model"})
    monkeypatch.setattr(module, "HuggingFaceTokenCounter", Counter)

    result = {
        "status": "completed",
        "error": None,
        "usage": {"completion_tokens": 100, "reasoning_tokens": 80},
        "calls": [
            {
                "index": 0,
                "role": "direct",
                "label": "direct",
                "max_tokens": 2_944,
                "messages": [{"role": "user", "content": "Prove P."}],
                "tools": [],
                "response": {"finish_reason": "stop"},
                "error": None,
            }
        ],
    }
    judged = {"judge_status": "completed", "judge_result": {"context_recovery": None}}

    def run(**kwargs: Any) -> dict[str, Any]:
        for stage, value in (("solve", result), ("judge", judged)):
            destination = tmp_path / stage / "runs" / "run-test" / "result.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(value))
        kwargs.setdefault("check_context", True)
        return module.audit("unused.yaml", **kwargs)

    return SimpleNamespace(result=result, judged=judged, run=run, module=module)


@pytest.mark.parametrize(
    "schema_name",
    [
        "QUERY_SUCCESS_PROBABILITY_TOOL",
        "SPAWN_SUBAGENTS_TOOL",
        "SUBMIT_PROBABILITY_TOOL",
        "SUBMIT_RATIONALE_SCORE_TOOL",
        "SUBMIT_RATIONALE_SCORE_VERDICT_TOOL",
        "SUBMIT_VERDICT_TOOL",
        "SUBMIT_PLAN_TOOL",
        "SUBMIT_REVIEW_TOOL",
    ],
)
def test_budget_audit_restores_schema_order_after_sorted_json_roundtrip(
    audit_fixture: Any, schema_name: str
) -> None:
    module = audit_fixture.module
    original = [getattr(module, schema_name)]
    saved = json.loads(json.dumps(original, sort_keys=True))
    saved_serialization = json.dumps(saved)
    assert saved == original
    assert saved_serialization != json.dumps(original)

    restored = module._restore_runtime_tools(saved)

    assert json.dumps(restored) == json.dumps(original)
    assert json.dumps(saved) == saved_serialization


def test_budget_audit_preserves_tool_list_and_schema_array_order(audit_fixture: Any) -> None:
    module = audit_fixture.module
    original = [module.SUBMIT_REVIEW_TOOL, module.SUBMIT_PROBABILITY_TOOL]
    saved = json.loads(json.dumps(original, sort_keys=True))

    restored = module._restore_runtime_tools(saved)

    assert json.dumps(restored) == json.dumps(original)
    saved[0]["function"]["parameters"]["required"].reverse()
    with pytest.raises(ValueError, match="does not match a known runtime schema"):
        module._restore_runtime_tools(saved)


@pytest.mark.parametrize("mutation", ["unknown_name", "changed_limit", "changed_type"])
def test_budget_audit_rejects_unknown_or_mutated_tool_schemas(
    audit_fixture: Any, mutation: str
) -> None:
    tool = json.loads(json.dumps(audit_fixture.module.SUBMIT_PROBABILITY_TOOL, sort_keys=True))
    if mutation == "unknown_name":
        tool["function"]["name"] = "unknown_tool"
    elif mutation == "changed_limit":
        tool["function"]["parameters"]["properties"]["success_probability"]["minimum"] = 0.5
    else:
        # JSON boolean false and numeric zero must not match through Python's ==.
        tool["function"]["parameters"]["additionalProperties"] = 0
    audit_fixture.result["calls"][0]["tools"] = [tool]

    result = audit_fixture.run()

    assert result["ready"] is False
    assert [issue["issue"] for issue in result["issues"]] == ["unrecognized_tool_schema"]
    assert result["summary"]["imo_proof/direct"]["unrecognized_tool_schema"] == 1
    assert audit_fixture.run(check_context=False)["issues"] == []


@pytest.mark.parametrize("schema_name", ["SUBMIT_PROBABILITY_TOOL", "SUBMIT_REVIEW_TOOL"])
@pytest.mark.parametrize(
    ("cap_difference", "expected_issue"),
    [(0, None), (-1, "harness_cap_below_context"), (1, "request_exceeds_context")],
)
def test_budget_audit_checks_exact_native_cap_after_restoring_tool_order(
    audit_fixture: Any,
    monkeypatch: pytest.MonkeyPatch,
    schema_name: str,
    cap_difference: int,
    expected_issue: str | None,
) -> None:
    module = audit_fixture.module
    original = [getattr(module, schema_name)]
    call = audit_fixture.result["calls"][0]
    call["tools"] = json.loads(json.dumps(original, sort_keys=True))
    call["max_tokens"] = 2_944 + cap_difference
    call["response"]["finish_reason"] = "length"
    audit_fixture.result.update(status="protocol_error", error="missing required tool call")

    class OrderSensitiveCounter:
        def __init__(self, path: str, *, enable_thinking: bool) -> None:
            pass

        def count_messages(self, messages: Any, tools: Any) -> int:
            # Model the observed one-token difference caused by sorted schema keys.
            return 128 if json.dumps(tools) == json.dumps(original) else 129

    monkeypatch.setattr(module, "HuggingFaceTokenCounter", OrderSensitiveCounter)

    result = audit_fixture.run()

    assert result["ready"] is (expected_issue is None)
    assert [issue["issue"] for issue in result["issues"]] == (
        [] if expected_issue is None else [expected_issue]
    )
    limited_call = result["length_affected_runs"][0]["calls"][0]
    assert limited_call["prompt_tokens"] == 128
    assert limited_call["context_cap"] == 2_944


@pytest.mark.parametrize("offer_tools", [False, True])
@pytest.mark.parametrize(
    ("cap_difference", "expected_issue"),
    [(0, None), (-1, "harness_cap_below_context"), (1, "request_exceeds_context")],
)
def test_budget_audit_restores_message_order_for_runtime_json_fallback(
    audit_fixture: Any,
    monkeypatch: pytest.MonkeyPatch,
    offer_tools: bool,
    cap_difference: int,
    expected_issue: str | None,
) -> None:
    module = audit_fixture.module
    original_messages = [
        {"role": "system", "content": "Solve P."},
        {"role": "user", "content": "Prove P."},
        {
            "role": "assistant",
            "content": "\n\n",
            "reasoning_content": "Check the cases independently.",
            "tool_calls": [
                {
                    "id": "call-b",
                    "type": "function",
                    "function": {
                        "name": "spawn_subagents",
                        "arguments": '{"tasks":[{"task":"Check P.","context_excerpt":"P"}]}',
                    },
                },
                {
                    "id": "call-a",
                    "type": "function",
                    "function": {"name": "query_success_probability", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-b",
            "name": "spawn_subagents",
            "content": '{"result":"Checked P."}',
        },
        {
            "role": "tool",
            "tool_call_id": "call-a",
            "name": "query_success_probability",
            "content": '{"success_probability":0.5}',
        },
    ]
    original_tools = [module.SPAWN_SUBAGENTS_TOOL] if offer_tools else []
    expected_fallback = json.dumps(
        {"messages": original_messages, "tools": original_tools}, ensure_ascii=False, default=str
    )
    call = audit_fixture.result["calls"][0]
    call["messages"] = json.loads(json.dumps(original_messages, sort_keys=True))
    call["tools"] = json.loads(json.dumps(original_tools, sort_keys=True))
    call["max_tokens"] = 2_944 + cap_difference
    assert json.dumps(call["messages"]) != json.dumps(original_messages)

    class FallbackCounter(HuggingFaceTokenCounter):
        def __init__(self, path: str, *, enable_thinking: bool) -> None:
            self.enable_thinking = enable_thinking
            self.tokenizer = SimpleNamespace(apply_chat_template=self.reject_string_arguments)

        @staticmethod
        def reject_string_arguments(messages: Any, **kwargs: Any) -> None:
            # The pinned Qwen template applies |items to these literal strings.
            assert isinstance(messages[2]["tool_calls"][0]["function"]["arguments"], str)
            raise TypeError("Can only get item pairs from a mapping.")

        def count_text(self, text: str) -> int:
            # Exercise the real counter's fallback and require its original bytes.
            assert text == expected_fallback
            return 128

    monkeypatch.setattr(module, "HuggingFaceTokenCounter", FallbackCounter)

    result = audit_fixture.run()

    assert result["ready"] is (expected_issue is None)
    assert [issue["issue"] for issue in result["issues"]] == (
        [] if expected_issue is None else [expected_issue]
    )
    assert json.dumps(module._restore_runtime_messages(call["messages"])) == json.dumps(
        original_messages
    )


@pytest.mark.parametrize(
    "message",
    [
        {"role": "developer", "content": "Unknown role."},
        {"role": "user", "content": {"text": "Unknown content shape."}},
        {"role": "assistant", "content": "P.", "unexpected_field": "cannot recover order"},
        {"role": "tool", "content": "Missing tool identity."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-a",
                    "type": "function",
                    "function": {"name": "query_success_probability", "arguments": {}},
                }
            ],
        },
    ],
)
def test_budget_audit_rejects_unrecoverable_message_shapes(
    audit_fixture: Any, message: dict[str, Any]
) -> None:
    audit_fixture.result["calls"][0]["messages"] = [message]

    result = audit_fixture.run()

    assert result["ready"] is False
    assert [issue["issue"] for issue in result["issues"]] == ["unrecognized_message_schema"]
    assert result["summary"]["imo_proof/direct"]["unrecognized_message_schema"] == 1


def test_budget_audit_allows_a_native_context_limited_completion(audit_fixture: Any) -> None:
    audit_fixture.result.update(
        status="budget_exhausted", error="direct completion reached its generation limit"
    )
    audit_fixture.result["calls"][0]["response"]["finish_reason"] = "length"

    result = audit_fixture.run()

    assert result["ready"] is True
    assert result["issues"] == []
    assert result["summary"]["imo_proof/direct"]["trajectories_with_length"] == 1


def test_budget_audit_blocks_a_smaller_harness_allowance_even_without_truncation(
    audit_fixture: Any,
) -> None:
    audit_fixture.result["calls"][0]["max_tokens"] = 2_000

    result = audit_fixture.run()

    assert result["ready"] is False
    assert "harness_cap_below_context" in {issue["issue"] for issue in result["issues"]}


def test_budget_audit_blocks_call_errors(audit_fixture: Any) -> None:
    audit_fixture.result["calls"][0].update(response=None, error="server unavailable")
    audit_fixture.result["status"] = "invalid_usage"

    result = audit_fixture.run()

    assert result["ready"] is False
    assert "call_error" in {issue["issue"] for issue in result["issues"]}


def test_budget_audit_blocks_runtime_failure_after_successful_response(audit_fixture: Any) -> None:
    audit_fixture.result.update(status="failed", error="could not persist request completion")
    audit_fixture.judged["judge_status"] = "solve_failed"

    result = audit_fixture.run()

    assert result["ready"] is False
    assert result["issues"]


def test_budget_audit_does_not_hide_phase_exhaustion_behind_an_earlier_length_call(
    audit_fixture: Any,
) -> None:
    audit_fixture.result["calls"][0]["response"]["finish_reason"] = "length"
    audit_fixture.result.update(
        status="budget_exhausted", error="generator candidate phase exhausted its budget"
    )
    audit_fixture.judged["judge_status"] = "solve_failed"

    result = audit_fixture.run()

    assert result["ready"] is False
    assert result["issues"]


def test_budget_audit_counts_nested_judge_context_recovery(audit_fixture: Any) -> None:
    audit_fixture.judged["judge_result"]["context_recovery"] = {"truncated": True}

    result = audit_fixture.run()

    assert result["summary"]["imo_proof/direct"]["judge_context_recovery"] == 1


def test_budget_audit_rejects_an_empty_pilot(audit_fixture: Any) -> None:
    result = audit_fixture.run(shards=(1,), shard_count=2)

    assert result["selected"] == 0
    assert result["ready"] is False


def test_budget_audit_retains_model_protocol_and_judge_parse_failures(audit_fixture: Any) -> None:
    audit_fixture.result.update(status="protocol_error", error="unparseable verifier verdict")
    audit_fixture.judged["judge_status"] = "solve_failed"
    assert audit_fixture.run()["ready"] is True

    audit_fixture.result.update(status="completed", error=None)
    audit_fixture.judged["judge_status"] = "parse_error"
    assert audit_fixture.run()["ready"] is True
