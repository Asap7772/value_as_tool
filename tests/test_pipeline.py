from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from value_as_tool import cli
from value_as_tool.benchmarks import BenchmarkItem
from value_as_tool.cli import build_parser
from value_as_tool.client import MissingUsageError
from value_as_tool.pipeline import (
    _orchestrator_config,
    joined_rows,
    judge,
    load_context,
    prepare,
    report,
    schedule_path,
    serve,
    smoke,
    solve,
    solve_store_path,
    status,
)
from value_as_tool.schedule import load_schedule
from value_as_tool.schemas import (
    AssistantMessage,
    ChatCompletion,
    TokenUsage,
    ToolCall,
    TrajectoryResult,
    TrajectoryStatus,
)
from value_as_tool.storage import ArtifactStore


class CharacterCounter:
    def count_messages(self, messages: Any, tools: Any = None) -> int:
        del tools
        return sum(len(str(message.get("content") or "")) for message in messages)

    def count_text(self, text: str) -> int:
        return len(text)

    def encode_text(self, text: str) -> tuple[int, ...]:
        return tuple(ord(character) for character in text)

    def decode_tokens(self, token_ids: Any) -> str:
        return "".join(chr(token) for token in token_ids)


class FakeClient:
    def __init__(self, response: str, *, delay: float = 0) -> None:
        self.response = response
        self.delay = delay
        self.calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
        self.active = 0
        self.maximum_active = 0

    async def complete(self, messages: Any, **kwargs: Any) -> ChatCompletion:
        copied = [dict(message) for message in messages]
        self.calls.append((copied, dict(kwargs)))
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            return ChatCompletion(
                id="fake",
                model=str(kwargs.get("model")),
                message=AssistantMessage(content=self.response),
                finish_reason="stop",
                usage=TokenUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
            )
        finally:
            self.active -= 1


class UnknownUsageClient:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages: Any, **kwargs: Any) -> ChatCompletion:
        del messages, kwargs
        self.calls += 1
        raise MissingUsageError("response omitted usage")


class SmokeSolverClient(FakeClient):
    def __init__(self) -> None:
        super().__init__("A complete proof.")

    async def complete(self, messages: Any, **kwargs: Any) -> ChatCompletion:
        copied = [dict(message) for message in messages]
        self.calls.append((copied, dict(kwargs)))
        tools = kwargs.get("tools") or []
        if any(tool.get("function", {}).get("name") == "submit_verdict" for tool in tools):
            message = AssistantMessage(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="verdict",
                        name="submit_verdict",
                        arguments=json.dumps(
                            {
                                "outcome": "correct",
                                "critique": "No substantive issue.",
                                "fault_category": "none",
                                "candidate_excerpt": "A complete proof.",
                            }
                        ),
                    ),
                )
            )
        else:
            message = AssistantMessage(content=self.response)
        return ChatCompletion(
            id="smoke",
            model=str(kwargs.get("model")),
            message=message,
            finish_reason="tool_calls" if message.tool_calls else "stop",
            usage=TokenUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        )


def _write_config(
    path: Path,
    *,
    retries: int = 1,
    concurrency: int = 2,
    backend: str = "sglang",
    conditions: str = "[direct]",
) -> None:
    path.write_text(
        "\n".join(
            [
                "paths:",
                "  artifact_root: run-artifacts",
                "  asset_root: model-assets",
                "evaluation:",
                f"  conditions: {conditions}",
                "  seeds: [0]",
                "  direct_answer_compatibility_seed: 0",
                "  bootstrap_samples: 10",
                "runtime:",
                f"  solver_backend: {backend}",
                f"  max_retries: {retries}",
                f"  max_concurrency: {concurrency}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _proof_items(count: int = 1) -> dict[str, list[BenchmarkItem]]:
    return {
        "imo_proof": [
            BenchmarkItem(
                benchmark="imo_proof",
                item_id=f"p-{index}",
                problem=f"Prove the sentinel identity number {index}.",
                solution=f"Reference proof {index}.",
                rubric={"7": "complete"},
                raw={},
            )
            for index in range(count)
        ]
    }


def _prepare_fake(context: Any, *, count: int = 1) -> dict[str, Any]:
    return prepare(
        context,
        benchmark_preparer=lambda config: {
            "benchmarks": {},
            "root": str(config.paths.artifact_root),
        },
        model_preparer=lambda config: {"models": {}, "root": str(config.paths.asset_root)},
        benchmark_loader=lambda config: _proof_items(count),
    )


def test_context_resolves_paths_without_changing_fingerprint(tmp_path: Path) -> None:
    first = tmp_path / "one" / "experiment.yaml"
    second = tmp_path / "two" / "experiment.yaml"
    first.parent.mkdir()
    second.parent.mkdir()
    _write_config(first)
    _write_config(second)

    left = load_context(first)
    right = load_context(second)

    assert left.config_fingerprint == right.config_fingerprint
    assert left.identity_config.paths.artifact_root == Path("run-artifacts")
    assert left.artifact_root == first.parent / "run-artifacts"
    assert right.artifact_root == second.parent / "run-artifacts"
    assert left.config.fingerprint != left.config_fingerprint

    operational = load_context(
        first,
        environment={
            "VALUE_AS_TOOL_ARTIFACT_ROOT": str(tmp_path / "shared-artifacts"),
            "VALUE_AS_TOOL_ASSET_ROOT": str(tmp_path / "shared-assets"),
        },
    )
    assert operational.config_fingerprint == left.config_fingerprint
    assert operational.artifact_root == tmp_path / "shared-artifacts"
    assert operational.asset_root == tmp_path / "shared-assets"
    assert left.identity_payload["experiment_config"] == left.identity_config.to_dict()
    assert left.identity_payload["implementation"]["package_source_sha256"]
    assert left.identity_payload["solver_protocol"]["tools"]["submit_verdict"]


def test_pipeline_propagates_minimum_solver_call_size(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    _write_config(config_path)
    context = load_context(config_path)

    runtime = _orchestrator_config(context.config)

    assert runtime.minimum_call_tokens == context.config.budget.minimum_call_tokens


def test_prepare_builds_an_immutable_schedule_only_when_called(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    _write_config(config_path)
    context = load_context(config_path)
    calls: list[str] = []

    def benchmarks(config: Any) -> dict[str, Any]:
        calls.append("benchmarks")
        return {"benchmarks": {}, "path": str(config.paths.artifact_root)}

    def models(config: Any) -> dict[str, Any]:
        calls.append("models")
        return {"models": {}, "path": str(config.paths.asset_root)}

    assert not schedule_path(context).exists()
    outcome = prepare(
        context,
        benchmark_preparer=benchmarks,
        model_preparer=models,
        benchmark_loader=lambda config: _proof_items(),
    )
    created = load_schedule(schedule_path(context))

    assert calls == ["benchmarks", "models"]
    assert outcome["scheduled"] == 1
    assert created.config_fingerprint == context.config_fingerprint
    assert created.fingerprint == outcome["schedule_fingerprint"]

    # A byte-equivalent rerun is accepted; a changed schedule is not silently
    # substituted by the write-once schedule layer.
    repeated = prepare(
        context,
        benchmark_preparer=benchmarks,
        model_preparer=models,
        benchmark_loader=lambda config: _proof_items(),
    )
    assert repeated["schedule_fingerprint"] == created.fingerprint


@pytest.mark.asyncio
async def test_solve_is_bounded_resumable_and_uses_qed_prompt(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    _write_config(config_path, concurrency=2)
    context = load_context(config_path)
    _prepare_fake(context, count=3)
    client = FakeClient("A complete proof.", delay=0.01)

    first = await solve(context, client=client, token_counter=CharacterCounter())
    second = await solve(context, client=client, token_counter=CharacterCounter())

    assert first["selected"] == 3
    assert first["outcomes"] == {"completed": 3}
    assert second["outcomes"] == {"already_complete": 3}
    assert len(client.calls) == 3
    assert client.maximum_active == 2
    assert all("Prove the sentinel identity" in call[0][0]["content"] for call in client.calls)
    assert all(call[1]["model"] == context.config.models.solver.name for call in client.calls)
    assert all(
        call[1]["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        for call in client.calls
    )

    schedule = load_schedule(schedule_path(context))
    store = ArtifactStore(
        solve_store_path(context),
        context.config_fingerprint,
        schedule.fingerprint,
    )
    assert all(store.is_complete(item.run_id) for item in schedule)
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert manifest["config"] == context.identity_payload


@pytest.mark.asyncio
async def test_unknown_solver_usage_invalidates_and_retries_fresh_attempts(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    _write_config(config_path, retries=1)
    context = load_context(config_path)
    _prepare_fake(context)
    client = UnknownUsageClient()

    outcome = await solve(context, client=client, token_counter=CharacterCounter())
    schedule = load_schedule(schedule_path(context))
    item = schedule[0]
    store = ArtifactStore(
        solve_store_path(context),
        context.config_fingerprint,
        schedule.fingerprint,
    )

    assert outcome["outcomes"] == {"invalidated": 1}
    assert client.calls == 2
    assert store.load_result(item.run_id) is None
    invalidations = store.invalidations(item.run_id)
    assert len(invalidations) == 2
    assert all(record["unknown_usage"] for record in invalidations)
    assert all(record["unknown_usage_upper_bound"] > 0 for record in invalidations)


@pytest.mark.asyncio
async def test_judge_is_condition_blind_separate_and_report_joins_schedule(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    _write_config(config_path)
    context = load_context(config_path)
    _prepare_fake(context)
    await solve(
        context,
        client=FakeClient("Candidate sentinel proof."),
        token_counter=CharacterCounter(),
    )
    judge_client = FakeClient("The proof is complete. <points>7</points>")

    outcome = await judge(
        context,
        client=judge_client,
        token_counter=CharacterCounter(),
    )
    rows = joined_rows(context)
    generated_report = report(context)
    stage_status = status(context)

    assert outcome["outcomes"] == {"completed": 1}
    assert len(judge_client.calls) == 1
    judge_messages = judge_client.calls[0][0]
    assert len(judge_messages) == 1 and judge_messages[0]["role"] == "user"
    assert "Candidate sentinel proof." in judge_messages[0]["content"]
    assert "condition" not in judge_messages[0]["content"].casefold()
    assert rows[0]["judge_status"] == "completed"
    assert rows[0]["score"] == 7
    assert rows[0]["correct"] is True
    assert generated_report["complete"] is True
    assert generated_report["scheduled_cells"] == 1
    persisted = json.loads(
        (context.artifact_root / "report" / "report.json").read_text(encoding="utf-8")
    )
    assert persisted == generated_report
    assert (context.artifact_root / "report" / "rows.jsonl").is_file()
    assert stage_status["solve"]["counts"] == {"completed": 1}
    assert stage_status["judge"]["counts"] == {"completed": 1}


@pytest.mark.asyncio
async def test_failed_solver_trajectory_is_zero_without_calling_judge(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    _write_config(config_path)
    context = load_context(config_path)
    _prepare_fake(context)
    schedule = load_schedule(schedule_path(context))
    item = schedule[0]
    solve_store = ArtifactStore(
        solve_store_path(context),
        context.config_fingerprint,
        schedule.fingerprint,
    )
    solve_store.initialize(config=context.identity_payload, metadata={"stage": "solve"})
    claim = solve_store.claim(item.run_id, identity={"test": "failed-solve"})
    assert claim.handle is not None
    failed = TrajectoryResult(
        request=item.to_request(),
        status=TrajectoryStatus.BUDGET_EXHAUSTED,
        final_output="A stale candidate that must never be externally judged.",
        usage=TokenUsage(prompt_tokens=13, completion_tokens=17, total_tokens=30),
    )
    claim.handle.finalize(
        {
            **failed.to_dict(),
            "ordinal": item.ordinal,
            "problem_fingerprint": item.problem_fingerprint,
            "usage_exact": True,
        }
    )
    judge_client = FakeClient("<points>7</points>")

    outcome = await judge(
        context,
        client=judge_client,
        token_counter=CharacterCounter(),
    )
    rows = joined_rows(context)
    generated = report(context)
    stage_status = status(context)

    assert outcome["outcomes"] == {"solve_failed": 1}
    assert judge_client.calls == []
    assert rows[0]["judge_status"] == "solve_failed"
    assert rows[0]["score"] is None
    assert rows[0]["judge_usage"]["total_tokens"] == 0
    assert generated["complete"] is False
    assert generated["summary"][0]["raw_success_rate"] == 0
    assert stage_status["solve"]["counts"] == {"budget_exhausted": 1}
    assert stage_status["judge"]["counts"] == {"solve_failed": 1}


@pytest.mark.asyncio
async def test_opt_in_smoke_runs_one_proof_across_all_four_conditions(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    _write_config(
        config_path,
        conditions="[direct, gvr, gvr_subagents, gvr_reference]",
    )
    context = load_context(config_path)
    _prepare_fake(context)
    solver_client = SmokeSolverClient()
    judge_client = FakeClient("<points>7</points>")

    outcome = await smoke(
        context,
        solver_client=solver_client,
        judge_client=judge_client,
        solver_token_counter=CharacterCounter(),
        judge_token_counter=CharacterCounter(),
    )

    assert outcome["problem_id"] == "p-0"
    assert outcome["ok"] is True
    assert outcome["failures"] == []
    assert outcome["conditions"] == [
        "direct",
        "gvr",
        "gvr_subagents",
        "gvr_reference",
    ]
    assert outcome["solve"]["outcomes"] == {"accepted": 3, "completed": 1}
    assert outcome["judge"]["outcomes"] == {"completed": 4}
    assert len(solver_client.calls) == 7
    assert len(judge_client.calls) == 4


@pytest.mark.asyncio
async def test_smoke_reports_failure_for_invalidated_solver_cells(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    _write_config(
        config_path,
        retries=0,
        conditions="[direct, gvr, gvr_subagents, gvr_reference]",
    )
    context = load_context(config_path)
    _prepare_fake(context)
    solver_client = UnknownUsageClient()
    judge_client = FakeClient("<points>7</points>")

    outcome = await smoke(
        context,
        solver_client=solver_client,
        judge_client=judge_client,
        solver_token_counter=CharacterCounter(),
        judge_token_counter=CharacterCounter(),
    )

    assert outcome["ok"] is False
    assert len(outcome["failures"]) == 4
    assert {failure["solve_status"] for failure in outcome["failures"]} == {
        "invalidated"
    }
    assert judge_client.calls == []


def test_report_materializes_missing_cells_as_zero(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    _write_config(config_path)
    context = load_context(config_path)
    _prepare_fake(context)

    rows = joined_rows(context)
    generated = report(context)

    assert len(rows) == 1
    assert rows[0]["solve_status"] == "missing"
    assert rows[0]["judge_status"] == "unjudged"
    assert rows[0]["generated_tokens"] == 0
    assert generated["complete"] is False
    assert generated["summary"][0]["raw_success_rate"] == 0


def test_serve_uses_the_pinned_model_manifest_without_exec_during_setup(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    _write_config(config_path, backend="vllm")
    context = load_context(config_path)
    model_path = context.asset_root / "solver"
    model_path.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "models": {
            "solver": {
                "name": context.config.models.solver.name,
                "revision": context.config.models.solver.revision,
                "path": str(model_path),
            }
        },
    }
    manifest_path = context.asset_root / "models" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    captured = []

    serve(context, "qwen", tensor_parallel_size=4, executor=captured.append)

    assert len(captured) == 1
    assert captured[0].model_path == str(model_path)
    assert captured[0].served_model_name == context.config.models.solver.name
    assert captured[0].tensor_parallel_size == 4
    assert captured[0].environment == {"VLLM_USE_RUST_FRONTEND": "1"}
    assert "qwen3_coder" in captured[0].command()

    manifest["models"]["solver"]["tokenizer_only"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="only tokenizer files"):
        serve(context, "qwen", executor=captured.append)


def test_cli_accepts_canonical_global_config_and_shard_arguments() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "custom.yaml",
            "--set",
            "runtime.max_concurrency=4",
            "solve",
            "--shard-index",
            "2",
            "--shard-count",
            "8",
        ]
    )
    assert args.config == Path("custom.yaml")
    assert args.overrides == ["runtime.max_concurrency=4"]
    assert args.command == "solve"
    assert (args.shard_index, args.shard_count) == (2, 8)

    smoke_args = build_parser().parse_args(
        [
            "--config",
            "custom.yaml",
            "smoke",
            "--benchmark",
            "proofbench",
            "--problem-id",
            "proof-7",
            "--seed",
            "2",
        ]
    )
    assert smoke_args.command == "smoke"
    assert smoke_args.benchmark == "proofbench"
    assert smoke_args.problem_id == "proof-7"
    assert smoke_args.seed == 2

    prepare_args = build_parser().parse_args(
        ["--config", "custom.yaml", "prepare", "--tokenizers-only"]
    )
    assert prepare_args.command == "prepare"
    assert prepare_args.tokenizers_only is True


def test_smoke_cli_returns_nonzero_on_failed_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failed_smoke(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {"stage": "smoke", "ok": False, "failures": [{"condition": "direct"}]}

    monkeypatch.setattr(cli, "load_context", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "smoke", failed_smoke)
    monkeypatch.setattr(cli, "_print", lambda value: None)

    assert cli.main(["--config", str(tmp_path / "missing.yaml"), "smoke"]) == 1
