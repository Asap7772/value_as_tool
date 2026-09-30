from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from value_as_tool.client import ChatClientError
from value_as_tool.conditioning import (
    MODES,
    ConditioningError,
    build_bank,
    enumerate_summary_tasks,
    file_sha256,
    freeze_bank,
    load_conditioning_pack,
    load_frozen_manifest,
    summarize_task,
)
from value_as_tool.config import ModelConfig, ModelsConfig, SamplingConfig, canonical_json
from value_as_tool.schemas import AssistantMessage, ChatCompletion, TokenUsage
from value_as_tool.storage import ArtifactMismatchError, atomic_write_json, read_json


class Counter:
    def count_text(self, text: str) -> int:
        return (len(text) + 3) // 4

    def count_messages(self, messages: Any, tools: Any = None) -> int:
        return sum(self.count_text(message["content"]) + 4 for message in messages)


class FakeClient:
    def __init__(self, outputs: list[Any] | None = None) -> None:
        self.outputs = list(outputs or [])
        self.calls: list[dict[str, Any]] = []

    async def complete(self, messages: Any, **kwargs: Any) -> ChatCompletion:
        self.calls.append({"messages": messages, **kwargs})
        output = self.outputs.pop(0) if self.outputs else "Concise source evidence."
        if isinstance(output, Exception):
            raise output
        content, finish = output if isinstance(output, tuple) else (output, "stop")
        prompt = Counter().count_messages(messages)
        return ChatCompletion(
            id=f"call-{len(self.calls)}",
            model=kwargs["model"],
            message=AssistantMessage(content=content, reasoning="Summarizer's private thinking."),
            finish_reason=finish,
            usage=TokenUsage(
                prompt_tokens=prompt,
                completion_tokens=10,
                total_tokens=prompt + 10,
                reasoning_tokens=7,
            ),
        )


def make_source(
    tmp_path: Path,
    *,
    scores: Mapping[str, list[int | None]] | None = None,
    long_thinking: str | None = None,
) -> tuple[Path, Path, Any]:
    source, bank = tmp_path / "source", tmp_path / "bank"
    model = ModelConfig(
        name="Qwen/Qwen3.5-9B",
        revision="a" * 40,
        base_url="http://localhost:8000/v1",
        api_key_env="MODEL_API_KEY",
    )
    config = SimpleNamespace(
        models=ModelsConfig(solver=model),
        sampling=SamplingConfig(thinking_content_reserve_tokens=0),
        budget=SimpleNamespace(context_tokens=4096, context_headroom_tokens=64),
        conditioning=SimpleNamespace(
            bank_root=bank,
            source_artifact_root=source,
            bank_sha256=None,
            map_target_tokens=64,
            summary_target_tokens=128,
            max_summary_attempts=3,
        ),
        runtime=SimpleNamespace(request_timeout_seconds=10),
    )
    context = SimpleNamespace(config=config, config_path=tmp_path / "experiment.yaml")
    scores = scores or {"p1": [7, 0, None], "p2": [0, 0, 0]}
    items = []
    for problem_id, values in scores.items():
        for seed, score in enumerate(values):
            run_id = f"run-{problem_id}-{seed}"
            solution = f"FINAL SOLUTION {problem_id} {seed}."
            thinking = long_thinking or f"PRIVATE SOURCE THINKING {problem_id} {seed}."
            common = {
                "run_id": run_id,
                "config_fingerprint": "config-fp",
                "schedule_fingerprint": "schedule-fp",
                "problem_fingerprint": f"fingerprint-{problem_id}",
            }
            items.append(
                {
                    **common,
                    "harness_id": "direct",
                    "benchmark": "imo_proof",
                    "problem_id": problem_id,
                    "seed": seed,
                    "problem": f"PROBLEM {problem_id}",
                    "reference_proof": "SECRET GOLD PROOF",
                    "rubric": "SECRET RUBRIC",
                    "golden_answer": "SECRET GOLD ANSWER",
                }
            )
            usage = {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "reasoning_tokens": 15,
            }
            atomic_write_json(
                source / "solve" / "runs" / run_id / "result.json",
                {
                    **common,
                    "status": "completed",
                    "final_output": solution,
                    "calls": [
                        {
                            "response": {
                                "finish_reason": "stop",
                                "message": {
                                    "content": solution,
                                    "reasoning": thinking,
                                },
                            }
                        }
                    ],
                    "usage": usage,
                },
            )
            atomic_write_json(
                source / "judge" / "runs" / run_id / "result.json",
                {
                    **common,
                    "score": score,
                    "correct": score == 7,
                    "judge_status": "completed" if score is not None else "parse_error",
                    "solution_sha256": hashlib.sha256(solution.encode()).hexdigest(),
                    "judge_usage": usage,
                    "judge_result": {"messages": [{"content": "SECRET GOLD JUDGE RATIONALE"}]},
                },
            )
    atomic_write_json(
        source / "schedule.json",
        {
            "items": items,
            "config_fingerprint": "config-fp",
            "schedule_fingerprint": "schedule-fp",
        },
    )
    atomic_write_json(
        source / "solve" / "manifest.json",
        {
            "config": {"experiment_config": {"models": {"solver": model.model_dump()}}},
        },
    )
    return source, bank, context


def build_fixture(tmp_path: Path, **kwargs: Any) -> tuple[Path, Any, dict[str, Any]]:
    source, bank, context = make_source(tmp_path, **kwargs)
    manifest = build_bank(source, bank, source_seeds=[0, 1, 2], config=context.config)
    context.config.conditioning.bank_sha256 = manifest["bank_sha256"]
    return bank, context, manifest


def test_bank_excludes_unknown_labels_keeps_one_sided_problems_and_no_gold(tmp_path: Path) -> None:
    bank, context, manifest = build_fixture(tmp_path)
    assert manifest["totals"]["attempts"] == 5
    assert manifest["totals"]["excluded_attempts"] == 1
    assert manifest["totals"]["problems_without_successes"] == 1
    assert manifest["problems"]["imo_proof"]["p2"]["labels"] == [
        {"attempt_id": f"run-p2-{seed}", "seed": seed, "correct": False} for seed in range(3)
    ]
    assert manifest["problems"]["imo_proof"]["p1"]["source_usage"]["completion_tokens"] == 60
    for entry in manifest["problems"]["imo_proof"].values():
        raw = (bank / entry["path"]).read_text()
        assert "SECRET GOLD" not in raw
        assert "SECRET RUBRIC" not in raw
        assert "PRIVATE SOURCE THINKING" in raw
    resumed = build_bank(
        context.config.conditioning.source_artifact_root,
        bank,
        source_seeds=[0, 1, 2],
        config=context.config,
    )
    assert resumed["bank_sha256"] == manifest["bank_sha256"]


def test_bank_rejects_relabeling_source_changes_and_incomplete_seed_grid(tmp_path: Path) -> None:
    source, bank, context = make_source(tmp_path)
    judge_path = source / "judge" / "runs" / "run-p1-0" / "result.json"
    judge = read_json(judge_path)
    judge["solution_sha256"] = "wrong"
    atomic_write_json(judge_path, judge)
    with pytest.raises(ArtifactMismatchError, match="another solution"):
        build_bank(source, bank, source_seeds=[0, 1, 2], config=context.config)
    with pytest.raises(ConditioningError, match="missing or duplicate"):
        build_bank(source, bank, source_seeds=[0, 1, 2, 3], config=context.config)


@pytest.mark.parametrize(
    "source_model",
    [
        None,
        {},
        {"name": "Qwen/Qwen3.5-9B"},
        {"revision": "a" * 40},
        {"name": " ", "revision": "a" * 40},
        {"name": 9, "revision": "a" * 40},
        {"name": "Qwen/Qwen3.5-9B", "revision": "main"},
        {"name": "Qwen/Qwen3.5-9B", "revision": True},
    ],
)
def test_bank_requires_valid_source_model_provenance(tmp_path: Path, source_model: Any) -> None:
    source, bank, context = make_source(tmp_path)
    path = source / "solve" / "manifest.json"
    manifest = read_json(path)
    manifest["config"]["experiment_config"]["models"]["solver"] = source_model
    atomic_write_json(path, manifest)
    with pytest.raises(ArtifactMismatchError, match="requires a pinned solver model"):
        build_bank(source, bank, source_seeds=[0, 1, 2], config=context.config)
    assert not bank.exists()


@pytest.mark.parametrize("field,value", [("name", "Qwen/AnotherModel"), ("revision", "b" * 40)])
def test_bank_rejects_a_different_source_model(tmp_path: Path, field: str, value: str) -> None:
    source, bank, context = make_source(tmp_path)
    path = source / "solve" / "manifest.json"
    manifest = read_json(path)
    manifest["config"]["experiment_config"]["models"]["solver"][field] = value
    atomic_write_json(path, manifest)
    with pytest.raises(ArtifactMismatchError, match="model pins differ"):
        build_bank(source, bank, source_seeds=[0, 1, 2], config=context.config)
    assert not bank.exists()


@pytest.mark.asyncio
async def test_full_summary_roundtrip_freezes_labels_and_separates_thinking(tmp_path: Path) -> None:
    bank, context, bank_manifest = build_fixture(tmp_path)
    tasks = enumerate_summary_tasks(bank)
    assert len(tasks) == 14  # Ten maps and four problem/mode reductions.
    assert len(enumerate_summary_tasks(bank, ready_only=True)) == 10
    reducer = next(task for task in tasks if task["kind"] == "reduce")
    blocked = await summarize_task(
        context, reducer["task_id"], client=FakeClient(), token_counter=Counter()
    )
    assert blocked["status"] == "blocked"
    for task in sorted(tasks, key=lambda task: task["kind"] == "reduce"):
        client = FakeClient()
        result = await summarize_task(
            context, task["task_id"], client=client, token_counter=Counter()
        )
        assert result["status"] == "completed"
        serialized = json.dumps(client.calls)
        assert "SECRET GOLD" not in serialized
        assert "SECRET RUBRIC" not in serialized
        assert ("PRIVATE SOURCE THINKING" in serialized) == (
            task["kind"] == "map" and task["mode"] == "thinking_summary"
        )
        assert all(
            call["extra_body"]["chat_template_kwargs"]["enable_thinking"] for call in client.calls
        )
        assert all(
            call["max_tokens"] > context.config.conditioning.summary_target_tokens * 10
            for call in client.calls
        )
        # Completed tasks do not dispatch duplicate model calls.
        again = await summarize_task(
            context, task["task_id"], client=client, token_counter=Counter()
        )
        assert again == result
        assert len(client.calls) == 1
    assert not enumerate_summary_tasks(bank, ready_only=True)
    manifest = freeze_bank(bank, expected_bank_sha256=bank_manifest["bank_sha256"])
    assert manifest["summary_task_count"] == 14
    assert manifest["summary_usage"]["completion_tokens"] == 140
    assert manifest["summary_all_attempt_usage"]["completion_tokens"] == 140
    assert manifest["summary_unknown_usage"]["requests"] == 0
    assert load_frozen_manifest(bank, manifest["manifest_sha256"]) == manifest
    assert freeze_bank(bank) == manifest
    for problem_id, problem in manifest["problems"]["imo_proof"].items():
        for mode in MODES:
            pack = load_conditioning_pack(
                bank, "imo_proof", problem_id, mode, problem["packs"][mode]["sha256"]
            )
            assert pack["labels"] == problem["labels"]
            assert "PRIVATE SOURCE THINKING" not in pack["content"]
            assert "Summarizer's private thinking" not in pack["content"]
            if mode == "solutions":
                assert "FINAL SOLUTION" in pack["content"]
            assert json.loads(pack["content"])["authoritative_attempt_labels"] == problem["labels"]


@pytest.mark.asyncio
async def test_oversize_thinking_is_losslessly_split_with_room_for_reasoning(
    tmp_path: Path,
) -> None:
    long_thinking = "BEGIN TRACE " + "many mathematical steps. " * 1800 + " END TRACE"
    bank, context, manifest = build_fixture(tmp_path, long_thinking=long_thinking)
    task = next(
        task for task in enumerate_summary_tasks(bank) if task["mode"] == "thinking_summary"
    )
    client = FakeClient()
    result = await summarize_task(context, task["task_id"], client=client, token_counter=Counter())
    assert result["status"] == "completed"
    assert len(client.calls) > 1
    problem = read_json(bank / manifest["problems"]["imo_proof"]["p1"]["path"])
    attempt = problem["attempts"][0]
    expected = canonical_json({"solution": attempt["solution"], "thinking": attempt["thinking"]})
    original_pieces = []
    for receipt in result["calls"]:
        assert receipt["max_tokens"] >= 1024
        assert receipt["prompt_tokens_estimate"] + receipt["max_tokens"] == 4032
        if ".merge" not in receipt["label"]:
            text = receipt["messages"][1]["content"]
            original_pieces.append(
                text.split("<untrusted_attempt_material>\n", 1)[1].rsplit(
                    "\n</untrusted_attempt_material>", 1
                )[0]
            )
    assert "".join(original_pieces) == expected
    assert result["input_truncated"] is False


@pytest.mark.asyncio
async def test_length_and_overlong_outputs_retry_and_failed_summaries_block_freeze(
    tmp_path: Path,
) -> None:
    bank, context, _ = build_fixture(tmp_path)
    task = enumerate_summary_tasks(bank)[0]
    client = FakeClient(
        [("short but truncated", "length"), "x" * 1000, "Complete concise summary."]
    )
    result = await summarize_task(context, task["task_id"], client=client, token_counter=Counter())
    assert result["status"] == "completed"
    assert len(client.calls) == 3
    assert result["usage"]["completion_tokens"] == 30
    other = enumerate_summary_tasks(bank)[1]
    failure = await summarize_task(
        context,
        other["task_id"],
        client=FakeClient([("partial", "length")] * 3),
        token_counter=Counter(),
    )
    assert failure["status"] == "failed"
    assert failure["usage_exact"] is True
    with pytest.raises(ConditioningError, match="failed validation"):
        freeze_bank(bank)


@pytest.mark.asyncio
async def test_unknown_usage_is_invalidated_and_charged_on_resume(tmp_path: Path) -> None:
    bank, context, _ = build_fixture(tmp_path)
    task = enumerate_summary_tasks(bank)[0]
    with pytest.raises(ChatClientError):
        await summarize_task(
            context,
            task["task_id"],
            client=FakeClient([ChatClientError("network lost")]),
            token_counter=Counter(),
        )
    result = await summarize_task(
        context, task["task_id"], client=FakeClient(), token_counter=Counter()
    )
    assert result["attempt"] == 2
    invalidation = read_json(
        bank / "summary" / "runs" / task["task_id"] / "attempts" / "000001" / "invalidation.json"
    )
    assert invalidation["unknown_usage"] is True
    for pending in enumerate_summary_tasks(bank):
        await summarize_task(
            context, pending["task_id"], client=FakeClient(), token_counter=Counter()
        )
    manifest = freeze_bank(bank)
    assert manifest["summary_unknown_usage"]["requests"] == 1
    assert (
        manifest["summary_unknown_usage"]["completion_tokens_upper_bound"]
        == invalidation["unknown_usage_upper_bound"]
    )


@pytest.mark.asyncio
async def test_changed_settings_or_problem_digest_cannot_reuse_summaries(tmp_path: Path) -> None:
    bank, context, manifest = build_fixture(tmp_path)
    task = enumerate_summary_tasks(bank)[0]
    context.config.conditioning.map_target_tokens = 32
    with pytest.raises(ArtifactMismatchError, match="settings differ"):
        await summarize_task(context, task["task_id"], client=FakeClient(), token_counter=Counter())
    context.config.conditioning.map_target_tokens = 64
    problem_path = bank / manifest["problems"]["imo_proof"]["p1"]["path"]
    problem = read_json(problem_path)
    problem["attempts"][0]["correct"] = False
    atomic_write_json(problem_path, problem)
    with pytest.raises(ArtifactMismatchError, match="digest mismatch"):
        await summarize_task(context, task["task_id"], client=FakeClient(), token_counter=Counter())
    assert file_sha256(problem_path) != manifest["problems"]["imo_proof"]["p1"]["sha256"]


@pytest.mark.asyncio
async def test_modified_pack_is_rejected(tmp_path: Path) -> None:
    bank, context, _ = build_fixture(tmp_path)
    for task in enumerate_summary_tasks(bank):
        await summarize_task(context, task["task_id"], client=FakeClient(), token_counter=Counter())
    manifest = freeze_bank(bank)
    pack_path = bank / manifest["problems"]["imo_proof"]["p1"]["packs"]["solutions"]["path"]
    pack = read_json(pack_path)
    pack["content"] += "tampered"
    atomic_write_json(pack_path, pack)
    with pytest.raises(ArtifactMismatchError, match="digest mismatch"):
        load_conditioning_pack(bank, "imo_proof", "p1", "solutions")
    with pytest.raises(ArtifactMismatchError, match="manifest digest"):
        load_frozen_manifest(bank, "0" * 64)
