from __future__ import annotations

import copy
import csv
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_pipeline import CharacterCounter, _prepare_fake

from value_as_tool.conditioning import (
    SUMMARY_SYSTEM,
    build_bank,
    enumerate_summary_tasks,
    freeze_bank,
    load_conditioning_pack,
    summarize_task,
)
from value_as_tool.harnesses import ATTEMPT_CONDITIONED_HARNESSES
from value_as_tool.pipeline import joined_rows, judge, load_context, report, schedule_path, solve
from value_as_tool.schedule import load_schedule
from value_as_tool.schemas import AssistantMessage, ChatCompletion, Role, TokenUsage, ToolCall
from value_as_tool.storage import read_json

PRIOR_SUCCESS = "PRIOR_SUCCESS: The induction proves every required case."
PRIOR_FAILURE = "PRIOR_FAILURE: The argument omits its base case."
SOURCE_THINKING = "PRIVATE_SOURCE_THINKING: I considered an alternative invariant."
SUMMARY = "FROZEN_SUMMARY: The successful attempt checks the base case; the other omits it."
SUMMARY_THINKING = "PRIVATE_SUMMARIZER_THINKING: Compare the labeled attempts."
CANDIDATE = "A complete proof by induction, including the base case."


class OfflineClient:
    def __init__(self, respond: Callable[..., AssistantMessage]) -> None:
        self.respond = respond
        self.calls: list[tuple[Any, Any]] = []

    async def complete(self, messages: Any, **kwargs: Any) -> ChatCompletion:
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        message = self.respond(messages, kwargs)
        return ChatCompletion(
            id=f"offline-{len(self.calls)}",
            model=kwargs["model"],
            message=message,
            finish_reason="tool_calls" if message.tool_calls else "stop",
            usage=TokenUsage(
                prompt_tokens=11, completion_tokens=7, total_tokens=18, reasoning_tokens=3
            ),
        )


def _tool(name: str, arguments: dict[str, Any]) -> AssistantMessage:
    return AssistantMessage(
        content=None,
        tool_calls=(ToolCall(f"{name}-call", name, json.dumps(arguments)),),
        reasoning="The current work includes the necessary boundary case.",
    )


def _source_response(messages: Any, kwargs: Any) -> AssistantMessage:
    del messages
    assert kwargs["seed"] in {0, 1}
    return AssistantMessage(
        content=PRIOR_SUCCESS if kwargs["seed"] == 0 else PRIOR_FAILURE,
        reasoning=SOURCE_THINKING,
    )


def _summary_response(messages: Any, kwargs: Any) -> AssistantMessage:
    del kwargs
    assert messages[0]["content"] == SUMMARY_SYSTEM
    return AssistantMessage(content=SUMMARY, reasoning=SUMMARY_THINKING)


def _conditioned_response(messages: Any, kwargs: Any) -> AssistantMessage:
    tools = {
        tool["function"]["name"]: tool["function"] for tool in (kwargs.get("tools") or ())
    }
    if "submit_verdict" in tools:
        arguments: dict[str, Any] = {
            "outcome": "correct",
            "fault_category": "none",
            "candidate_excerpt": CANDIDATE,
        }
        if "rationale" in tools["submit_verdict"]["parameters"]["properties"]:
            arguments.update(success_probability=0.9, rationale="Every case has been checked.")
        else:
            arguments["critique"] = "Every case has been checked."
        return _tool("submit_verdict", arguments)
    if "submit_probability" in tools:
        arguments = {"success_probability": 0.9}
        if "rationale" in tools["submit_probability"]["parameters"]["properties"]:
            arguments["rationale"] = "The partial work covers the boundary case."
        return _tool("submit_probability", arguments)
    if "query_success_probability" in tools and not any(
        message["role"] == "tool" for message in messages
    ):
        return _tool("query_success_probability", {})
    return AssistantMessage(content=CANDIDATE, reasoning="Check the induction step.")


def _judge_response(messages: Any, kwargs: Any) -> AssistantMessage:
    del kwargs
    score = 0 if PRIOR_FAILURE in json.dumps(messages) else 7
    return AssistantMessage(content=f"External assessment. <points>{score}</points>")


def _write_config(path: Path, *, conditioned: bool) -> None:
    value: dict[str, Any] = {
        "paths": {
            "artifact_root": "conditioned" if conditioned else "source",
            "asset_root": "unused-offline-assets",
        },
        "sampling": {"enable_thinking": True, "thinking_content_reserve_tokens": 0},
        "evaluation": {
            "conditions": [] if conditioned else ["direct"],
            "benchmarks": ["imo_proof"],
            "seeds": [8, 9] if conditioned else [0, 1],
            "direct_answer_compatibility_seed": 8 if conditioned else 0,
            "bootstrap_samples": 10,
        },
        "runtime": {"max_retries": 0, "max_concurrency": 2},
    }
    if conditioned:
        value["evaluation"]["harnesses"] = list(ATTEMPT_CONDITIONED_HARNESSES)
        value["conditioning"] = {
            "source_artifact_root": "source",
            "bank_root": "bank",
            "source_seeds": [0, 1],
            "map_target_tokens": 512,
            "summary_target_tokens": 1024,
        }
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


@pytest.mark.asyncio
async def test_direct_attempts_to_all_conditioned_arms_and_reports_offline(tmp_path: Path) -> None:
    counter = CharacterCounter()
    source_path = tmp_path / "source.yaml"
    evaluation_path = tmp_path / "evaluation.yaml"
    _write_config(source_path, conditioned=False)
    _write_config(evaluation_path, conditioned=True)
    source = load_context(source_path, environment={})
    evaluation = load_context(evaluation_path, environment={})
    assert source.config.models.solver == evaluation.config.models.solver
    assert _prepare_fake(source, count=2)["scheduled"] == 4
    source_client = OfflineClient(_source_response)
    source_judge = OfflineClient(_judge_response)
    assert (await solve(source, client=source_client, token_counter=counter))["outcomes"] == {
        "completed": 4
    }
    assert (await judge(source, client=source_judge, token_counter=counter))["outcomes"] == {
        "completed": 4
    }
    assert Counter(row["correct"] for row in joined_rows(source)) == {True: 2, False: 2}
    assert (await solve(source, client=source_client, token_counter=counter))["outcomes"] == {
        "already_complete": 4
    }
    assert (await judge(source, client=source_judge, token_counter=counter))["outcomes"] == {
        "already_complete": 4
    }
    assert len(source_client.calls) == len(source_judge.calls) == 4

    bank_root = tmp_path / "bank"
    bank = build_bank(
        source.artifact_root, bank_root, source_seeds=(0, 1), config=evaluation.config
    )
    assert bank["totals"]["successful_attempts"] == 2
    assert bank["totals"]["unsuccessful_attempts"] == 2
    summary_context = load_context(
        evaluation_path, overrides={"conditioning.bank_sha256": bank["bank_sha256"]},
        environment={},
    )
    summarizer = OfflineClient(_summary_response)
    tasks = enumerate_summary_tasks(bank_root)
    assert len(tasks) == 12
    for task in tasks:
        start = len(summarizer.calls)
        summary = await summarize_task(
            summary_context, task["task_id"], client=summarizer, token_counter=counter
        )
        assert summary["status"] == "completed"
        assert len(summarizer.calls) == start + 1
        messages, kwargs = summarizer.calls[-1]
        rendered = json.dumps(messages)
        assert (SOURCE_THINKING in rendered) is (
            task["kind"] == "map" and task["mode"] == "thinking_summary"
        )
        assert "Reference proof" not in rendered
        assert SUMMARY_THINKING not in rendered
        assert kwargs["model"] == source.config.models.solver.name
        assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
        assert await summarize_task(
            summary_context, task["task_id"], client=summarizer, token_counter=counter
        ) == summary
        assert len(summarizer.calls) == start + 1
    assert not enumerate_summary_tasks(bank_root, ready_only=True)
    frozen = freeze_bank(bank_root, expected_bank_sha256=bank["bank_sha256"])
    assert freeze_bank(bank_root) == frozen
    evaluation = load_context(
        evaluation_path,
        overrides={
            "conditioning.bank_sha256": bank["bank_sha256"],
            "conditioning.manifest_sha256": frozen["manifest_sha256"],
        },
        environment={},
    )
    prepared = _prepare_fake(evaluation, count=2)
    assert prepared["scheduled"] == 24 * 2 * 2 == 96
    assert _prepare_fake(evaluation, count=2) == prepared
    schedule = load_schedule(schedule_path(evaluation))
    assert len({item.harness_id for item in schedule}) == 24
    assert {item.seed for item in schedule} == {8, 9}

    solver = OfflineClient(_conditioned_response)
    external_judge = OfflineClient(_judge_response)
    solved = await solve(evaluation, client=solver, token_counter=counter)
    assert solved["outcomes"] == {"accepted": 48, "completed": 48}
    assert (await judge(evaluation, client=external_judge, token_counter=counter))["outcomes"] == {
        "completed": 96
    }
    for item in schedule:
        result = read_json(
            evaluation.artifact_root / "solve" / "runs" / item.run_id / "result.json"
        )
        pack = load_conditioning_pack(
            bank_root, item.benchmark, item.problem_id, item.conditioning_mode,
            expected_sha256=item.conditioning_pack_sha256,
        )
        assert result["request"]["verifier_evidence"] == pack
        assert {label["correct"] for label in pack["labels"]} == {True, False}
        assert SOURCE_THINKING not in pack["content"]
        assert SUMMARY_THINKING not in pack["content"]
        for call in result["calls"]:
            rendered = json.dumps(call["messages"])
            verifier = call["role"] in {Role.VERIFIER.value, Role.VALUE_VERIFIER.value}
            assert (item.conditioning_pack_sha256 in rendered) is verifier
            assert (item.reference_proof in rendered) is (
                verifier and item.harness_access == "attempt_and_reference_assisted"
            )
            assert (PRIOR_SUCCESS in rendered) is (
                verifier and item.conditioning_mode == "solutions"
            )
            assert (PRIOR_FAILURE in rendered) is (
                verifier and item.conditioning_mode == "solutions"
            )
            assert (SUMMARY in rendered) is (verifier and item.conditioning_mode != "solutions")
            assert SOURCE_THINKING not in rendered and SUMMARY_THINKING not in rendered
    for messages, kwargs in external_judge.calls:
        assert kwargs["model"] == evaluation.config.models.judge.name
        rendered = json.dumps(messages)
        assert CANDIDATE in rendered and "Reference proof" in rendered
        for forbidden in (
            PRIOR_SUCCESS, PRIOR_FAILURE, SUMMARY, SOURCE_THINKING, "verifier_evidence"
        ):
            assert forbidden not in rendered
        assert not any(item.harness_id in rendered for item in schedule)

    call_counts = len(solver.calls), len(external_judge.calls)
    assert call_counts == (240, 96)
    assert (await solve(evaluation, client=solver, token_counter=counter))["outcomes"] == {
        "already_complete": 96
    }
    assert (await judge(evaluation, client=external_judge, token_counter=counter))["outcomes"] == {
        "already_complete": 96
    }
    assert (len(solver.calls), len(external_judge.calls)) == call_counts
    generated = report(evaluation)
    assert generated["complete"] is True
    assert generated["scheduled_cells"] == 96
    assert len(generated["summary"]) == 24
    assert generated["conditioning_analysis"]["complete"] is True
    assert generated["conditioning_analysis"]["comparisons_per_benchmark_per_k"] == 60
    destination = evaluation.artifact_root / "report"
    assert read_json(destination / "report.json") == generated
    for filename in generated["conditioning_analysis"]["files"]:
        assert (destination / filename).is_file()
    with (destination / "conditioning-pass-at-k.csv").open() as stream:
        curves = list(csv.DictReader(stream))
    assert len(curves) == 24 * 2
    assert {int(row["k"]) for row in curves} == {1, 2}
    assert {float(row["pass_at_k"]) for row in curves} == {1.0}
    costs = read_json(destination / "conditioning-preprocessing.json")
    assert costs["source_usage"]["completion_tokens"] == 4 * 7
    assert costs["summary_usage"]["completion_tokens"] == 12 * 7
    assert costs["source_judge_usage"]["completion_tokens"] == 4 * 7
    assert report(evaluation) == generated
