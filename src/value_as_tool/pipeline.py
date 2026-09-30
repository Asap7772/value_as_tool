"""End-to-end experiment stages for the standalone evaluation harness.

The stage functions in this module are deliberately inert until called.  In
particular, importing the module never downloads assets, contacts a model
endpoint, or starts a server.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from .assets import (
    load_model_manifest,
    load_prepared_benchmarks,
    model_manifest_path,
    prepare_benchmark_assets,
    prepare_model_assets,
)
from .benchmarks import BenchmarkItem, QEDPromptSet
from .client import ChatClient, OpenAIChatClient
from .config import ExperimentConfig, load_config
from .harnesses import load_harness
from .identity import build_experiment_identity, experiment_fingerprint
from .judging import JUDGE_CONTEXT_TOKENS, JudgeResult, JudgeRunner, build_judge_request
from .orchestrator import (
    QWEN3_THINKING_BUDGET_PROCESSOR,
    AletheiaOrchestrator,
    NonResumableTrajectoryError,
    OrchestratorConfig,
)
from .reporting import write_report
from .schedule import (
    Schedule,
    ScheduleItem,
    build_schedule,
    items_for_shard,
    load_schedule,
    write_schedule,
)
from .schemas import (
    ADJUDICATABLE_TRAJECTORY_STATUSES,
    TrajectoryResult,
    TrajectoryStatus,
)
from .server import (
    PreflightCheck,
    PreflightReport,
    ServerSpec,
    exec_server,
)
from .server import (
    preflight as server_preflight,
)
from .storage import (
    ArtifactMismatchError,
    ArtifactStore,
    ClaimAction,
    RunClaimedError,
    atomic_write_text,
    canonical_json,
    read_jsonl,
)
from .thinking import build_thinking_budget_processor, derive_thinking_token_profile
from .tokenization import HuggingFaceTokenCounter


@dataclass(frozen=True, slots=True)
class PipelineContext:
    """Path-independent experiment identity plus runtime-resolved paths."""

    config_path: Path
    identity_config: ExperimentConfig
    config: ExperimentConfig
    identity_payload: Mapping[str, Any]
    config_fingerprint: str

    @property
    def artifact_root(self) -> Path:
        return self.config.paths.artifact_root

    @property
    def asset_root(self) -> Path:
        return self.config.paths.asset_root


StageSource = PipelineContext | str | Path


def load_context(
    config_path: str | Path = "experiment.yaml",
    *,
    overrides: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
) -> PipelineContext:
    """Load config and resolve output paths without changing its fingerprint."""

    source = Path(config_path).resolve()
    identity = load_config(source, overrides=overrides)
    runtime_paths = identity.paths.resolve(source.parent)
    environ = os.environ if environment is None else environment
    path_overrides: dict[str, Path] = {}
    for field, variable in (
        ("artifact_root", "VALUE_AS_TOOL_ARTIFACT_ROOT"),
        ("asset_root", "VALUE_AS_TOOL_ASSET_ROOT"),
    ):
        value = environ.get(variable)
        if value:
            path_overrides[field] = Path(value).expanduser().resolve()
    if path_overrides:
        runtime_paths = runtime_paths.model_copy(update=path_overrides)
    runtime = identity.model_copy(update={"paths": runtime_paths})
    identity_payload = build_experiment_identity(identity)
    return PipelineContext(
        config_path=source,
        identity_config=identity,
        config=runtime,
        identity_payload=identity_payload,
        config_fingerprint=experiment_fingerprint(identity_payload),
    )


def _context(source: StageSource) -> PipelineContext:
    return source if isinstance(source, PipelineContext) else load_context(source)


def schedule_path(source: StageSource) -> Path:
    return _context(source).artifact_root / "schedule.json"


def solve_store_path(source: StageSource) -> Path:
    return _context(source).artifact_root / "solve"


def judge_store_path(source: StageSource) -> Path:
    return _context(source).artifact_root / "judge"


def report_path(source: StageSource) -> Path:
    return _context(source).artifact_root / "report"


def _load_schedule(context: PipelineContext) -> Schedule:
    schedule = load_schedule(schedule_path(context))
    if schedule.config_fingerprint != context.config_fingerprint:
        raise ArtifactMismatchError(
            "schedule configuration fingerprint does not match the selected config"
        )
    return schedule


def _store(
    context: PipelineContext,
    schedule: Schedule,
    stage: Literal["solve", "judge"],
    *,
    initialize: bool = True,
) -> ArtifactStore:
    root = solve_store_path(context) if stage == "solve" else judge_store_path(context)
    store = ArtifactStore(
        root,
        config_fingerprint=context.config_fingerprint,
        schedule_fingerprint=schedule.fingerprint,
    )
    if initialize or store.manifest_path.exists():
        store.initialize(
            config=context.identity_payload,
            metadata={"stage": stage},
        )
    return store


def _model_entry(
    context: PipelineContext,
    role: Literal["solver", "judge"],
    *,
    require_weights: bool = False,
) -> dict[str, Any]:
    manifest = load_model_manifest(context.config)
    models = manifest.get("models")
    entry = models.get(role) if isinstance(models, Mapping) else None
    if not isinstance(entry, Mapping):
        raise ValueError(f"model manifest has no {role!r} entry")
    configured = getattr(context.identity_config.models, role)
    if entry.get("name") != configured.name or entry.get("revision") != configured.revision:
        raise ArtifactMismatchError(f"prepared {role} model does not match experiment config")
    if require_weights and entry.get("tokenizer_only") is True:
        raise ValueError(
            f"prepared {role} asset contains only tokenizer files; rerun prepare "
            "without --tokenizers-only before local serving"
        )
    path = Path(str(entry.get("path", "")))
    if not path.exists():
        raise FileNotFoundError(f"prepared {role} model path does not exist: {path}")
    return dict(entry)


def preflight(
    source: StageSource,
    *,
    check_endpoints: bool = False,
    require_server_binaries: bool = False,
    environment: Mapping[str, str] | None = None,
) -> PreflightReport:
    """Inspect local prerequisites without downloading or launching anything."""

    context = _context(source)
    checks: list[PreflightCheck] = []
    model_paths: dict[str, Path | None] = {"solver": None, "judge": None}
    try:
        QEDPromptSet()
        checks.append(PreflightCheck("qed_prompts", True, "vendored QED-Nano prompts verified"))
    except Exception as exc:
        checks.append(
            PreflightCheck(
                "qed_prompts", False, f"QED prompt verification failed: {exc}"
            )
        )

    manifest_path = model_manifest_path(context.config)
    if not manifest_path.exists():
        manifest_path = context.asset_root / "models" / "manifest.json"
    if manifest_path.exists():
        for role in ("solver", "judge"):
            try:
                entry = _model_entry(context, role)
                tokenizer_path = Path(entry["path"])
                tokenizers_only = entry.get("tokenizer_only") is True
                if not tokenizers_only:
                    model_paths[role] = tokenizer_path
                checks.append(
                    PreflightCheck(
                        f"{role}_manifest",
                        True,
                        (
                            f"prepared {role} tokenizer: {tokenizer_path}"
                            if tokenizers_only
                            else f"prepared {role} model: {tokenizer_path}"
                        ),
                    )
                )
                if tokenizers_only:
                    checks.append(
                        PreflightCheck(
                            f"{role}_weights",
                            False,
                            f"{role} weights are not prepared for local serving",
                            required=False,
                        )
                    )
            except Exception as exc:
                checks.append(
                    PreflightCheck(
                        f"{role}_manifest",
                        False,
                        f"invalid prepared {role} model: {exc}",
                    )
                )
    else:
        checks.append(
            PreflightCheck(
                "model_manifest",
                False,
                f"model assets are not prepared: {manifest_path}",
                required=check_endpoints,
            )
        )

    selected_schedule = schedule_path(context)
    if selected_schedule.exists():
        try:
            _load_schedule(context)
            checks.append(
                PreflightCheck("schedule", True, f"schedule verified: {selected_schedule}")
            )
        except Exception as exc:
            checks.append(
                PreflightCheck("schedule", False, f"invalid prepared schedule: {exc}")
            )
    else:
        checks.append(
            PreflightCheck(
                "schedule",
                False,
                f"schedule is not prepared: {selected_schedule}",
                required=check_endpoints,
            )
        )

    runtime = server_preflight(
        context.identity_config,
        experiment_fingerprint=context.config_fingerprint,
        solver_model_path=model_paths["solver"],
        judge_model_path=model_paths["judge"],
        check_endpoints=check_endpoints,
        require_server_binaries=require_server_binaries,
        environment=environment,
    )
    return PreflightReport(tuple(checks) + runtime.checks)


def prepare(
    source: StageSource,
    *,
    benchmark_preparer: Callable[[ExperimentConfig], Mapping[str, Any]] = (
        prepare_benchmark_assets
    ),
    model_preparer: Callable[..., Mapping[str, Any]] = prepare_model_assets,
    benchmark_loader: Callable[[ExperimentConfig], Mapping[str, Sequence[BenchmarkItem]]] = (
        load_prepared_benchmarks
    ),
    tokenizers_only: bool = False,
) -> dict[str, Any]:
    """Explicitly materialize pinned assets and the immutable full schedule."""

    context = _context(source)
    benchmark_manifest = dict(benchmark_preparer(context.config))
    model_manifest = dict(
        model_preparer(context.config, tokenizer_only=True)
        if tokenizers_only
        else model_preparer(context.config)
    )
    benchmarks = benchmark_loader(context.config)
    schedule = build_schedule(
        benchmarks,
        context.identity_config,
        config_fingerprint=context.config_fingerprint,
    )
    destination = write_schedule(schedule_path(context), schedule)
    return {
        "config_fingerprint": context.config_fingerprint,
        "schedule_fingerprint": schedule.fingerprint,
        "schedule_path": str(destination),
        "scheduled": len(schedule),
        "benchmark_manifest": benchmark_manifest,
        "model_manifest": model_manifest,
        "tokenizers_only": tokenizers_only,
    }


def _schedule_identity(item: ScheduleItem, *, stage: str) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "stage": stage,
        "ordinal": item.ordinal,
        "run_id": item.run_id,
        "benchmark": item.benchmark,
        "problem_id": item.problem_id,
        "condition": _method_id(item),
        "seed": item.seed,
        "problem_fingerprint": item.problem_fingerprint,
    }
    identity.update(_harness_identity(item))
    return identity


def _method_id(item: ScheduleItem) -> str:
    if item.harness_id:
        return item.harness_id
    if item.condition is not None:
        return item.condition.value
    raise ValueError(f"schedule item {item.run_id!r} has no method identity")


def _harness_identity(item: ScheduleItem) -> dict[str, Any]:
    if item.harness_id is None:
        return {}
    return {
        "harness_id": item.harness_id,
        "harness_entrypoint": item.harness_entrypoint,
        "harness_source_sha256": item.harness_source_sha256,
        "harness_access": item.harness_access,
    }


def _benchmark_item(item: ScheduleItem) -> BenchmarkItem:
    return BenchmarkItem(
        benchmark=item.benchmark,
        item_id=item.problem_id,
        problem=item.problem,
        solution=item.reference_proof,
        answer=item.golden_answer,
        rubric=item.rubric,
        raw=dict(item.metadata),
    )


def _orchestrator_config(
    config: ExperimentConfig,
    *,
    token_counter: Any | None = None,
) -> OrchestratorConfig:
    thinking_processor = QWEN3_THINKING_BUDGET_PROCESSOR
    tokenizer = getattr(token_counter, "tokenizer", None)
    if tokenizer is not None:
        thinking_processor = build_thinking_budget_processor(
            derive_thinking_token_profile(tokenizer)
        )
    return OrchestratorConfig(
        model=config.models.solver.name,
        total_generated_tokens=config.budget.generated_tokens,
        context_tokens=config.budget.context_tokens,
        context_headroom_tokens=config.budget.context_headroom_tokens,
        initial_generator_cap=config.budget.initial_generator_tokens,
        verifier_cap=config.budget.verifier_tokens,
        correction_pool=config.budget.correction_pool_tokens,
        cch_stage_tokens=config.budget.cch_stage_tokens,
        minimum_call_tokens=config.budget.minimum_call_tokens,
        max_cycles=config.budget.max_candidate_versions,
        subagent_cap=config.subagents.child_tokens,
        max_subagents=config.subagents.max_children,
        final_candidate_reserve_tokens=config.subagents.final_candidate_reserve_tokens,
        subagent_context_max_chars=config.subagents.max_context_chars,
        value_tool_max_queries=config.value_tool.max_queries,
        value_verifier_cap=config.value_tool.verifier_tokens,
        value_final_response_reserve_tokens=(
            config.value_tool.final_response_reserve_tokens
        ),
        temperature=config.sampling.temperature,
        top_p=config.sampling.top_p,
        top_k=config.sampling.top_k,
        min_p=config.sampling.min_p,
        presence_penalty=config.sampling.presence_penalty,
        repetition_penalty=config.sampling.repetition_penalty,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": config.sampling.enable_thinking}
        },
        thinking_content_reserve_tokens=config.sampling.thinking_content_reserve_tokens,
        thinking_budget_processor=thinking_processor,
    )


def _has_active_request(checkpoint: Mapping[str, Any] | None) -> bool:
    if not checkpoint:
        return False
    active = checkpoint.get("in_flight_requests")
    if isinstance(active, Mapping) and active:
        return True
    return isinstance(checkpoint.get("in_flight_request"), Mapping)


def _result_matches_item(item: ScheduleItem, result: Mapping[str, Any]) -> None:
    if result.get("run_id") != item.run_id:
        raise ArtifactMismatchError(f"result run ID mismatch for {item.run_id}")
    request = result.get("request")
    if not isinstance(request, Mapping):
        raise ArtifactMismatchError(f"solver result lacks request for {item.run_id}")
    expected = {
        "benchmark": item.benchmark,
        "problem_id": item.problem_id,
        "seed": item.seed,
    }
    if item.condition is not None:
        expected["condition"] = item.condition.value
    else:
        condition = request.get("condition")
        if condition is not None:
            raise ArtifactMismatchError(
                f"solver result condition mismatch for {item.run_id}: {condition!r}"
            )
    if item.harness_id is not None:
        expected["harness_id"] = item.harness_id
    for key, value in expected.items():
        if request.get(key) != value:
            raise ArtifactMismatchError(f"solver result {key} mismatch for {item.run_id}")
    if item.harness_id is not None:
        metadata = request.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ArtifactMismatchError(
                f"solver result lacks harness metadata for {item.run_id}"
            )
        for key, value in _harness_identity(item).items():
            if result.get(key) != value or metadata.get(key) != value:
                raise ArtifactMismatchError(
                    f"solver result {key} mismatch for {item.run_id}"
                )


async def _solve_one(
    item: ScheduleItem,
    *,
    context: PipelineContext,
    schedule: Schedule,
    store: ArtifactStore,
    client: ChatClient,
    token_counter: Any,
    prompts: QEDPromptSet,
) -> str:
    identity = _schedule_identity(item, stage="solve")
    harness = None
    if item.harness_entrypoint is not None:
        harness = load_harness(
            item.harness_entrypoint,
            source_sha256=item.harness_source_sha256,
        )
        if (
            harness.spec.harness_id != item.harness_id
            or harness.spec.access != item.harness_access
        ):
            raise ArtifactMismatchError(
                f"loaded harness identity does not match schedule item {item.run_id}"
            )
    retries = context.config.runtime.max_retries
    for retry in range(retries + 1):
        try:
            claim = store.claim(
                item.run_id,
                identity=identity,
                resume=context.config.runtime.resume,
            )
        except RunClaimedError:
            return "claimed_elsewhere"
        if claim.action is ClaimAction.COMPLETE:
            assert claim.result is not None
            _result_matches_item(item, claim.result)
            return "already_complete"
        handle = claim.handle
        assert handle is not None
        benchmark = _benchmark_item(item)
        request = replace(item.to_request(), solver_prompt=prompts.solve_prompt(benchmark))

        async def checkpoint(
            result: TrajectoryResult, attempt_handle: Any = handle
        ) -> None:
            attempt_handle.save_checkpoint("trajectory", result.to_dict())

        orchestrator = AletheiaOrchestrator(
            client,
            _orchestrator_config(context.config, token_counter=token_counter),
            checkpoint=checkpoint,
            request_lifecycle=handle,
            token_counter=token_counter,
        )
        try:
            result = await orchestrator.run(
                request,
                resume=claim.checkpoint,
                harness=harness,
            )
        except asyncio.CancelledError:
            handle.close()
            raise
        except NonResumableTrajectoryError as exc:
            unknown = _has_active_request(handle.load_checkpoint())
            handle.invalidate(
                f"nonresumable_checkpoint: {exc}",
                unknown_usage=unknown,
            )
            if retry < retries:
                continue
            return "invalidated"
        except Exception as exc:
            unknown = _has_active_request(handle.load_checkpoint())
            handle.invalidate(
                f"pipeline_error: {type(exc).__name__}: {exc}",
                unknown_usage=unknown,
            )
            if unknown and retry < retries:
                continue
            return "failed"

        checkpoint_value = handle.load_checkpoint()
        unknown = result.status is TrajectoryStatus.INVALID_USAGE or _has_active_request(
            checkpoint_value
        )
        if unknown:
            handle.invalidate(
                result.error or "trajectory ended with unknown provider usage",
                unknown_usage=True,
                details={"trajectory_status": result.status.value},
            )
            if retry < retries:
                continue
            return "invalidated"
        handle.finalize(
            {
                **result.to_dict(),
                "ordinal": item.ordinal,
                "problem_fingerprint": item.problem_fingerprint,
                "usage_exact": True,
                **_harness_identity(item),
            }
        )
        return result.status.value
    return "invalidated"


async def _bounded_map(
    items: Sequence[ScheduleItem],
    concurrency: int,
    operation: Callable[[ScheduleItem], Awaitable[str]],
) -> Counter[str]:
    queue: asyncio.Queue[ScheduleItem] = asyncio.Queue()
    for item in items:
        queue.put_nowait(item)
    outcomes: Counter[str] = Counter()

    async def worker() -> None:
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                outcomes[await operation(item)] += 1
            finally:
                queue.task_done()

    worker_count = min(max(1, concurrency), len(items))
    if worker_count:
        async with asyncio.TaskGroup() as group:
            for _ in range(worker_count):
                group.create_task(worker())
    return outcomes


async def solve(
    source: StageSource,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
    client: ChatClient | None = None,
    token_counter: Any | None = None,
    prompts: QEDPromptSet | None = None,
    environment: Mapping[str, str] | None = None,
    run_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run one resumable, stably sharded subset with bounded concurrency."""

    context = _context(source)
    schedule = _load_schedule(context)
    selected = _select_items(schedule, shard_index, shard_count, run_ids)
    store = _store(context, schedule, "solve")
    prompt_set = prompts or QEDPromptSet()
    if token_counter is None:
        token_counter = HuggingFaceTokenCounter(
            _model_entry(context, "solver")["path"],
            enable_thinking=context.config.sampling.enable_thinking,
        )
    environ = os.environ if environment is None else environment
    owns_client = client is None
    if client is None:
        model = context.config.models.solver
        client = OpenAIChatClient(
            context.config.models.operational_base_url("solver", environ),
            api_key=environ.get(model.api_key_env),
            timeout=context.config.runtime.request_timeout_seconds,
        )
    assert client is not None
    try:
        outcomes = await _bounded_map(
            selected,
            context.config.runtime.max_concurrency,
            lambda item: _solve_one(
                item,
                context=context,
                schedule=schedule,
                store=store,
                client=client,
                token_counter=token_counter,
                prompts=prompt_set,
            ),
        )
    finally:
        if owns_client:
            await client.aclose()  # type: ignore[attr-defined]
    return {
        "stage": "solve",
        "schedule_fingerprint": schedule.fingerprint,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "selected": len(selected),
        "outcomes": dict(sorted(outcomes.items())),
    }


def _judge_base_record(item: ScheduleItem, solve_result: Mapping[str, Any]) -> dict[str, Any]:
    usage = solve_result.get("usage")
    usage = dict(usage) if isinstance(usage, Mapping) else {}
    return {
        "benchmark": item.benchmark,
        "item_id": item.problem_id,
        "problem_id": item.problem_id,
        "condition": _method_id(item),
        "seed": item.seed,
        "ordinal": item.ordinal,
        "problem_fingerprint": item.problem_fingerprint,
        "solve_status": solve_result.get("status", "failed"),
        "generated_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
        "usage_exact": bool(solve_result.get("usage_exact", True)),
        "solution_sha256": hashlib.sha256(
            str(solve_result.get("final_output") or "").encode("utf-8")
        ).hexdigest(),
        **_harness_identity(item),
    }


def _judge_record(
    item: ScheduleItem,
    solve_result: Mapping[str, Any],
    result: JudgeResult,
) -> dict[str, Any]:
    return {
        **_judge_base_record(item, solve_result),
        "judge_status": result.status,
        "score": result.score,
        "correct": result.correct,
        "judge_usage": dict(result.usage),
        "judge_result": result.to_dict(),
    }


def _zero_judge_usage() -> dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "usage_exact": True,
    }


async def _judge_one(
    item: ScheduleItem,
    *,
    context: PipelineContext,
    solve_store: ArtifactStore,
    judge_store: ArtifactStore,
    client: ChatClient,
    runner: JudgeRunner,
) -> str:
    solve_result = solve_store.load_result(item.run_id)
    if solve_result is None:
        return "solve_missing"
    _result_matches_item(item, solve_result)
    final_output = str(solve_result.get("final_output") or "")
    identity = {
        **_schedule_identity(item, stage="judge"),
        "solution_sha256": hashlib.sha256(final_output.encode("utf-8")).hexdigest(),
        "solve_status": str(solve_result.get("status", "failed")),
    }
    retries = context.config.runtime.max_retries
    for retry in range(retries + 1):
        try:
            claim = judge_store.claim(
                item.run_id,
                identity=identity,
                resume=context.config.runtime.resume,
            )
        except RunClaimedError:
            return "claimed_elsewhere"
        if claim.action is ClaimAction.COMPLETE:
            assert claim.result is not None
            if claim.result.get("solution_sha256") != identity["solution_sha256"]:
                raise ArtifactMismatchError(f"judge solution mismatch for {item.run_id}")
            return "already_complete"
        handle = claim.handle
        assert handle is not None
        checkpoint = claim.checkpoint
        if checkpoint and checkpoint.get("state") == "judge.complete":
            payload = checkpoint.get("payload")
            if not isinstance(payload, Mapping):
                handle.invalidate("invalid completed judge checkpoint", unknown_usage=False)
                if retry < retries:
                    continue
                return "failed"
            handle.finalize(payload)
            return str(payload.get("judge_status", "completed"))

        base = _judge_base_record(item, solve_result)
        handle.save_checkpoint("judge.ready", base)
        if str(base["solve_status"]).casefold() not in ADJUDICATABLE_TRAJECTORY_STATUSES:
            record = {
                **base,
                "judge_status": "solve_failed",
                "score": None,
                "correct": False,
                "judge_usage": _zero_judge_usage(),
                "judge_result": None,
            }
            handle.finalize(record)
            return "solve_failed"
        benchmark = _benchmark_item(item)
        if not final_output.strip():
            record = {
                **base,
                "judge_status": "empty_answer",
                "score": 0,
                "correct": False,
                "judge_usage": _zero_judge_usage(),
                "judge_result": None,
            }
            handle.finalize(record)
            return "empty_answer"
        try:
            _, plan = build_judge_request(
                benchmark,
                final_output,
                prompts=runner.prompts,
                max_tokens=runner.max_tokens,
                token_counter=runner.token_counter,
            )
        except Exception as exc:
            record = {
                **base,
                "judge_status": "context_error",
                "score": None,
                "correct": False,
                "judge_usage": _zero_judge_usage(),
                "judge_result": None,
                "judge_error": str(exc),
            }
            handle.finalize(record)
            return "context_error"

        request_id = f"judge-{item.ordinal:06d}-{retry:02d}"
        handle.begin_request(
            request_id,
            role="judge",
            max_tokens=plan.max_tokens,
            metadata={"ordinal": item.ordinal, "input_tokens": plan.input_tokens},
        )
        try:
            result = await runner.judge(benchmark, final_output, client)
        except asyncio.CancelledError:
            handle.close()
            raise
        except Exception as exc:
            handle.invalidate(
                f"judge_pipeline_error: {type(exc).__name__}: {exc}",
                unknown_usage=True,
            )
            if retry < retries:
                continue
            return "invalidated"
        if not result.usage.get("usage_exact", False):
            handle.invalidate(
                result.error or "judge request returned without exact usage",
                unknown_usage=True,
                details={"judge_status": result.status},
            )
            if retry < retries:
                continue
            return "invalidated"
        record = _judge_record(item, solve_result, result)
        handle.complete_request(
            request_id,
            state="judge.complete",
            payload=record,
            usage=result.usage,
        )
        handle.finalize(record)
        return result.status
    return "invalidated"


async def judge(
    source: StageSource,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
    client: ChatClient | None = None,
    token_counter: Any | None = None,
    prompts: QEDPromptSet | None = None,
    environment: Mapping[str, str] | None = None,
    run_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Condition-blind judging of final trajectory outputs in separate storage."""

    context = _context(source)
    schedule = _load_schedule(context)
    selected = _select_items(schedule, shard_index, shard_count, run_ids)
    solve_store = _store(context, schedule, "solve", initialize=False)
    judge_store = _store(context, schedule, "judge")
    if token_counter is None:
        token_counter = HuggingFaceTokenCounter(
            _model_entry(context, "judge")["path"], enable_thinking=False
        )
    runner = JudgeRunner(
        prompts or QEDPromptSet(),
        max_tokens=context.config.evaluation.judge_output_tokens,
        token_counter=token_counter,
        reasoning_effort=context.config.evaluation.judge_reasoning_effort,
        model=context.config.models.judge.name,
    )
    environ = os.environ if environment is None else environment
    owns_client = client is None
    if client is None:
        model = context.config.models.judge
        client = OpenAIChatClient(
            context.config.models.operational_base_url("judge", environ),
            api_key=environ.get(model.api_key_env),
            timeout=context.config.runtime.request_timeout_seconds,
        )
    assert client is not None
    try:
        outcomes = await _bounded_map(
            selected,
            context.config.runtime.max_concurrency,
            lambda item: _judge_one(
                item,
                context=context,
                solve_store=solve_store,
                judge_store=judge_store,
                client=client,
                runner=runner,
            ),
        )
    finally:
        if owns_client:
            await client.aclose()  # type: ignore[attr-defined]
    return {
        "stage": "judge",
        "schedule_fingerprint": schedule.fingerprint,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "selected": len(selected),
        "outcomes": dict(sorted(outcomes.items())),
    }


def _select_items(
    schedule: Schedule,
    shard_index: int,
    shard_count: int,
    run_ids: Sequence[str] | None,
) -> tuple[ScheduleItem, ...]:
    if run_ids is None:
        return items_for_shard(schedule, shard_index, shard_count)
    if shard_index != 0 or shard_count != 1:
        raise ValueError("explicit run_ids cannot be combined with sharding")
    requested = tuple(run_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("run_ids must be unique")
    known = {item.run_id: item for item in schedule}
    unknown = sorted(set(requested) - set(known))
    if unknown:
        raise ValueError(f"run_ids are absent from the schedule: {', '.join(unknown[:5])}")
    return tuple(known[run_id] for run_id in requested)


def _smoke_cells(
    schedule: Schedule,
    *,
    benchmark: str,
    seed: int,
    problem_id: str | None,
    required: Sequence[str],
) -> tuple[ScheduleItem, ...]:
    required = tuple(required)
    problem_order: list[str] = []
    by_problem: dict[str, dict[str, ScheduleItem]] = {}
    for item in schedule:
        if item.benchmark != benchmark or item.seed != seed or not item.reference_proof:
            continue
        if item.problem_id not in by_problem:
            problem_order.append(item.problem_id)
            by_problem[item.problem_id] = {}
        by_problem[item.problem_id][_method_id(item)] = item
    selected_problem_ids = [problem_id] if problem_id is not None else problem_order
    for selected_problem_id in selected_problem_ids:
        cells = by_problem.get(selected_problem_id, {})
        if all(method in cells for method in required):
            return tuple(cells[method] for method in required)
    qualifier = f" problem {problem_id!r}" if problem_id is not None else ""
    raise ValueError(
        f"no reference-bearing {benchmark}{qualifier} has all configured harnesses "
        f"at seed {seed}"
    )


async def smoke(
    source: StageSource,
    *,
    benchmark: str = "imo_proof",
    seed: int = 0,
    problem_id: str | None = None,
    solver_client: ChatClient | None = None,
    judge_client: ChatClient | None = None,
    solver_token_counter: Any | None = None,
    judge_token_counter: Any | None = None,
    prompts: QEDPromptSet | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Opt-in live check of one proof problem across the configured conditions."""

    context = _context(source)
    schedule = _load_schedule(context)
    configured_methods = tuple(
        dict.fromkeys(_method_id(item) for item in schedule if item.benchmark == benchmark)
    )
    cells = _smoke_cells(
        schedule,
        benchmark=benchmark,
        seed=seed,
        problem_id=problem_id,
        required=configured_methods,
    )
    run_ids = tuple(item.run_id for item in cells)
    prompt_set = prompts or QEDPromptSet()
    solved = await solve(
        context,
        client=solver_client,
        token_counter=solver_token_counter,
        prompts=prompt_set,
        environment=environment,
        run_ids=run_ids,
    )
    judged = await judge(
        context,
        client=judge_client,
        token_counter=judge_token_counter,
        prompts=prompt_set,
        environment=environment,
        run_ids=run_ids,
    )
    solve_store = _store(context, schedule, "solve", initialize=False)
    judge_store = _store(context, schedule, "judge", initialize=False)
    failures: list[dict[str, str]] = []
    for item in cells:
        solve_result = solve_store.load_result(item.run_id)
        solve_state = solve_store.status(item.run_id)
        solve_status = str(
            (solve_result or {}).get("status", (solve_state or {}).get("status", "missing"))
        )
        judge_result = judge_store.load_result(item.run_id)
        judge_state = judge_store.status(item.run_id)
        judge_status = str(
            (judge_result or {}).get(
                "judge_status", (judge_state or {}).get("status", "missing")
            )
        )
        if solve_status not in ADJUDICATABLE_TRAJECTORY_STATUSES or judge_status != "completed":
            failures.append(
                {
                    "condition": _method_id(item),
                    "solve_status": solve_status,
                    "judge_status": judge_status,
                }
            )
    return {
        "stage": "smoke",
        "ok": not failures,
        "benchmark": benchmark,
        "problem_id": cells[0].problem_id,
        "seed": seed,
        "conditions": [_method_id(item) for item in cells],
        "run_ids": list(run_ids),
        "solve": solved,
        "judge": judged,
        "failures": failures,
    }


def _ensure_expected_runs(store: ArtifactStore, schedule: Schedule) -> None:
    expected = {item.run_id for item in schedule}
    unexpected = set(store.list_run_ids()) - expected
    if unexpected:
        examples = ", ".join(sorted(unexpected)[:5])
        raise ArtifactMismatchError(f"artifact store contains unexpected run IDs: {examples}")


def _invalidated_usage(store: ArtifactStore, run_id: str) -> dict[str, Any] | None:
    prompt_tokens = 0
    completion_tokens = 0
    unknown_upper_bound = 0
    exact = True
    records = store.invalidations(run_id)
    for invalidation in records:
        attempt = invalidation.get("attempt")
        if isinstance(attempt, int):
            events_path = (
                store.run_dir(run_id) / "attempts" / f"{attempt:06d}" / "events.jsonl"
            )
            if events_path.exists():
                for event in read_jsonl(events_path):
                    if event.get("event") != "request_completed":
                        continue
                    usage = event.get("usage")
                    if isinstance(usage, Mapping):
                        prompt_tokens += int(usage.get("prompt_tokens", 0))
                        completion_tokens += int(usage.get("completion_tokens", 0))
        if invalidation.get("unknown_usage"):
            exact = False
            unknown_upper_bound += int(invalidation.get("unknown_usage_upper_bound", 0))
    if not records:
        return None
    # For interrupted calls the provider-side amount is unknowable.  Preserve
    # the generated-token upper bound and mark the combined cost inexact.
    generated = completion_tokens + unknown_upper_bound
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": generated,
        "generated_tokens": generated,
        "total_tokens": prompt_tokens + generated,
        "usage_exact": exact,
        "unknown_usage_upper_bound": unknown_upper_bound,
        "attempts": len(records),
    }


def joined_rows(source: StageSource) -> list[dict[str, Any]]:
    """Materialize exactly one metric row for every immutable schedule cell."""

    context = _context(source)
    schedule = _load_schedule(context)
    solve_store = _store(context, schedule, "solve", initialize=False)
    judge_store = _store(context, schedule, "judge")
    _ensure_expected_runs(solve_store, schedule)
    _ensure_expected_runs(judge_store, schedule)
    rows: list[dict[str, Any]] = []
    for item in schedule:
        row: dict[str, Any] = {
            "run_id": item.run_id,
            "benchmark": item.benchmark,
            "item_id": item.problem_id,
            "problem_id": item.problem_id,
            "condition": _method_id(item),
            "seed": item.seed,
            "ordinal": item.ordinal,
            "applicable": True,
            "solve_status": "missing",
            "judge_status": "unjudged",
            "generated_tokens": 0,
            "total_tokens": 0,
            "usage_exact": False,
            "value_query_count": 0,
            "value_estimates": [],
            "verifier_assessments": [],
            "generated_tokens_by_role": {},
            **_harness_identity(item),
        }
        solve_result = solve_store.load_result(item.run_id)
        if solve_result is not None:
            _result_matches_item(item, solve_result)
            usage = solve_result.get("usage")
            normalized_usage = dict(usage) if isinstance(usage, Mapping) else {}
            raw_estimates = solve_result.get("value_estimates", [])
            value_estimates = (
                [dict(value) for value in raw_estimates if isinstance(value, Mapping)]
                if isinstance(raw_estimates, Sequence)
                and not isinstance(raw_estimates, (str, bytes))
                else []
            )
            raw_verdicts = solve_result.get("verdicts", [])
            verifier_assessments = (
                [dict(value) for value in raw_verdicts if isinstance(value, Mapping)]
                if isinstance(raw_verdicts, Sequence)
                and not isinstance(raw_verdicts, (str, bytes))
                else []
            )
            generated_by_role: Counter[str] = Counter()
            raw_calls = solve_result.get("calls", [])
            if isinstance(raw_calls, Sequence) and not isinstance(raw_calls, (str, bytes)):
                for call in raw_calls:
                    if not isinstance(call, Mapping):
                        continue
                    call_usage = call.get("usage")
                    if not isinstance(call_usage, Mapping):
                        continue
                    generated_by_role[str(call.get("role", "unknown"))] += int(
                        call_usage.get("completion_tokens", 0)
                    )
            row.update(
                solve_status=str(solve_result.get("status", "failed")),
                generated_tokens=int(normalized_usage.get("completion_tokens", 0)),
                total_tokens=int(normalized_usage.get("total_tokens", 0)),
                usage_exact=bool(solve_result.get("usage_exact", True)),
                value_query_count=len(value_estimates),
                value_estimates=value_estimates,
                verifier_assessments=verifier_assessments,
                generated_tokens_by_role=dict(sorted(generated_by_role.items())),
            )
        judge_result = judge_store.load_result(item.run_id)
        if judge_result is not None:
            expected_solution = hashlib.sha256(
                str((solve_result or {}).get("final_output") or "").encode("utf-8")
            ).hexdigest()
            if judge_result.get("solution_sha256") != expected_solution:
                raise ArtifactMismatchError(f"judge result solution mismatch for {item.run_id}")
            row.update(
                judge_status=str(judge_result.get("judge_status", "failed")),
                score=judge_result.get("score"),
                correct=judge_result.get("correct"),
                judge_usage=judge_result.get("judge_usage"),
                judge_result=judge_result.get("judge_result"),
            )
        invalidated_solve = _invalidated_usage(solve_store, item.run_id)
        invalidated_judge = _invalidated_usage(judge_store, item.run_id)
        row["invalidated_solve_usage"] = invalidated_solve
        row["invalidated_judge_usage"] = invalidated_judge
        invalidated_parts = [
            value
            for value in (invalidated_solve, invalidated_judge)
            if value is not None
        ]
        if invalidated_parts:
            row["invalidated_usage"] = {
                "prompt_tokens": sum(int(value["prompt_tokens"]) for value in invalidated_parts),
                "completion_tokens": sum(
                    int(value["completion_tokens"]) for value in invalidated_parts
                ),
                "generated_tokens": sum(
                    int(value["generated_tokens"]) for value in invalidated_parts
                ),
                "total_tokens": sum(int(value["total_tokens"]) for value in invalidated_parts),
                "usage_exact": all(bool(value["usage_exact"]) for value in invalidated_parts),
                "unknown_usage_upper_bound": sum(
                    int(value["unknown_usage_upper_bound"]) for value in invalidated_parts
                ),
                "attempts": sum(int(value["attempts"]) for value in invalidated_parts),
            }
        rows.append(row)
    return rows


def report(source: StageSource) -> dict[str, Any]:
    """Join the full schedule (including missing cells) and write all reports."""

    context = _context(source)
    schedule = _load_schedule(context)
    rows = joined_rows(context)
    destination = report_path(context)
    joined = "".join(canonical_json(row) + "\n" for row in rows)
    atomic_write_text(destination / "rows.jsonl", joined)
    result = write_report(
        destination,
        rows,
        bootstrap_samples=context.config.evaluation.bootstrap_samples,
        expected_seeds=context.config.evaluation.seeds,
        answer_compatibility_seed=(
            context.config.evaluation.direct_answer_compatibility_seed
        ),
    )
    result["schedule_fingerprint"] = schedule.fingerprint
    result["scheduled_cells"] = len(schedule)
    artifacts = result.get("artifacts")
    if isinstance(artifacts, dict):
        artifacts["rows_jsonl"] = "rows.jsonl"
    atomic_write_text(
        destination / "report.json",
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return result


def _stage_status(store: ArtifactStore, schedule: Schedule) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for item in schedule:
        result = store.load_result(item.run_id)
        if result is not None:
            terminal_status = result.get("judge_status", result.get("status", "completed"))
            counts[str(terminal_status)] += 1
            continue
        value = store.status(item.run_id)
        counts[str(value.get("status", "pending")) if value else "pending"] += 1
    return {"scheduled": len(schedule), "counts": dict(sorted(counts.items()))}


def status(source: StageSource) -> dict[str, Any]:
    context = _context(source)
    schedule = _load_schedule(context)
    solve_store = _store(context, schedule, "solve", initialize=False)
    judge_store = _store(context, schedule, "judge", initialize=False)
    _ensure_expected_runs(solve_store, schedule)
    _ensure_expected_runs(judge_store, schedule)
    return {
        "config_fingerprint": context.config_fingerprint,
        "schedule_fingerprint": schedule.fingerprint,
        "solve": _stage_status(solve_store, schedule),
        "judge": _stage_status(judge_store, schedule),
    }


def serve(
    source: StageSource,
    role: Literal["qwen", "judge"],
    *,
    tensor_parallel_size: int = 1,
    extra_args: Sequence[str] = (),
    executor: Callable[[ServerSpec], None] = exec_server,
) -> None:
    """Exec the selected prepared model server; called only by the serve CLI."""

    context = _context(source)
    model_role: Literal["solver", "judge"] = "solver" if role == "qwen" else "judge"
    if role not in ("qwen", "judge"):
        raise ValueError("serve role must be 'qwen' or 'judge'")
    model = getattr(context.config.models, model_role)
    entry = _model_entry(context, model_role, require_weights=True)
    backend = context.config.runtime.solver_backend if model_role == "solver" else "vllm"
    spec = ServerSpec(
        backend=backend,
        role=model_role,
        model_path=str(entry["path"]),
        served_model_name=model.name,
        base_url=context.config.models.operational_base_url(model_role),
        log_path=context.artifact_root / "logs" / f"serve-{model_role}.log",
        context_length=(
            context.config.budget.context_tokens
            if model_role == "solver"
            else JUDGE_CONTEXT_TOKENS
        ),
        tensor_parallel_size=tensor_parallel_size,
        extra_args=tuple(extra_args),
        environment=(
            {"VLLM_USE_RUST_FRONTEND": "1"}
            if model_role == "solver" and backend == "vllm"
            else {}
        ),
    )
    executor(spec)


async def run_all(
    source: StageSource,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run prepare, solve, judge, and report sequentially."""

    context = _context(source)
    prepared = prepare(context)
    solved = await solve(context, environment=environment)
    judged = await judge(context, environment=environment)
    reported = report(context)
    return {
        "prepare": prepared,
        "solve": solved,
        "judge": judged,
        "report": reported,
    }


# Descriptive aliases for callers that prefer explicit stage names.
preflight_stage = preflight
prepare_stage = prepare
solve_stage = solve
judge_stage = judge
report_stage = report
status_stage = status
serve_stage = serve
all_stages = run_all


__all__ = [
    "PipelineContext",
    "all_stages",
    "joined_rows",
    "judge",
    "judge_stage",
    "judge_store_path",
    "load_context",
    "preflight",
    "preflight_stage",
    "prepare",
    "prepare_stage",
    "report",
    "report_path",
    "report_stage",
    "run_all",
    "schedule_path",
    "serve",
    "serve_stage",
    "solve",
    "solve_stage",
    "solve_store_path",
    "smoke",
    "status",
    "status_stage",
]
