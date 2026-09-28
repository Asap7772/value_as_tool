from __future__ import annotations

import json
import tomllib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from value_as_tool import assets
from value_as_tool.assets import (
    model_snapshot_path,
    prepare_benchmark_assets,
    prepare_model_assets,
    prepared_benchmark_path,
)
from value_as_tool.benchmarks import BenchmarkItem
from value_as_tool.config import ExperimentConfig, PathsConfig, load_config
from value_as_tool.identity import package_source_sha256
from value_as_tool.schedule import (
    build_schedule,
    derive_seed,
    load_schedule,
    shard_items,
    write_schedule,
)
from value_as_tool.schemas import Condition
from value_as_tool.server import (
    SUPPORTED_SGLANG_VERSION,
    SUPPORTED_VLLM_VERSION,
    build_sglang_command,
    build_vllm_command,
)
from value_as_tool.storage import (
    ArtifactError,
    ArtifactMismatchError,
    ArtifactStore,
    ClaimAction,
    read_jsonl,
)


def _benchmark_rows() -> dict[str, list[dict[str, str]]]:
    return {
        "imo_proof": [
            {
                "id": f"imo-proof-{index:03d}",
                "problem": f"IMO proof problem {index}",
                "solution": f"IMO reference proof {index}",
            }
            for index in range(60)
        ],
        "proofbench": [
            {
                "id": f"proofbench-{index:03d}",
                "problem": f"ProofBench problem {index}",
                "solution": f"ProofBench reference proof {index}",
            }
            for index in range(145)
        ],
        "imo_answer": [
            {
                "id": f"imo-answer-{index:03d}",
                "problem": f"IMO answer problem {index}",
                "answer": str(index),
            }
            for index in range(400)
        ],
    }


def _small_schedule(config: ExperimentConfig):  # type: ignore[no-untyped-def]
    return build_schedule(
        {
            "imo_proof": [
                {
                    "id": "proof-1",
                    "problem": "Prove that 1 = 1.",
                    "solution": "By reflexivity.",
                }
            ]
        },
        config,
        conditions=(Condition.DIRECT,),
        seeds=(0,),
    )


def _artifact_store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(
        tmp_path / "artifacts",
        config_fingerprint="config-fingerprint",
        schedule_fingerprint="schedule-fingerprint",
    )
    manifest = store.initialize(metadata={"test": "offline"})
    assert manifest["config_fingerprint"] == "config-fingerprint"
    assert manifest["schedule_fingerprint"] == "schedule-fingerprint"
    return store


def test_checked_in_configuration_matches_strict_defaults(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    defaults = load_config(None)
    checked_in = load_config(repository_root / "experiment.yaml")

    assert checked_in == defaults
    assert checked_in.models.solver.name == "Qwen/Qwen3.5-9B"
    assert checked_in.models.solver.revision == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert checked_in.budget.generated_tokens == 229_376
    assert checked_in.budget.context_tokens == 262_144
    assert checked_in.evaluation.conditions == (
        "direct",
        "gvr",
        "gvr_subagents",
        "gvr_reference",
    )
    assert checked_in.evaluation.seeds == (0, 1, 2)

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("runtime:\n  misspelled_option: true\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="misspelled_option"):
        load_config(invalid)

    with pytest.raises(ValidationError, match="frozen"):
        checked_in.runtime.max_concurrency = 2

    with pytest.raises(ValidationError, match="exactly three candidate"):
        load_config(None, overrides={"budget.max_candidate_versions": 4})
    with pytest.raises(ValidationError, match="exactly three subagents"):
        load_config(None, overrides={"subagents.max_children": 4})
    with pytest.raises(ValidationError, match="value-tool limits must be integers"):
        load_config(None, overrides={"value_tool.max_queries": True})

    project = tomllib.loads((repository_root / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]
    assert extras["sglang"] == [f"sglang[all]=={SUPPORTED_SGLANG_VERSION}"]
    assert extras["vllm"] == [f"vllm=={SUPPORTED_VLLM_VERSION}"]


def test_operational_endpoint_overrides_do_not_change_config_identity() -> None:
    config = load_config(None)
    original_fingerprint = config.fingerprint
    original_dict = config.to_dict()
    environment = {
        "VALUE_AS_TOOL_SOLVER_BASE_URL": "https://solver.example/v1/",
        "VALUE_AS_TOOL_JUDGE_BASE_URL": "https://judge.example/api/v1/",
    }

    assert config.models.operational_base_url("solver", environment) == (
        "https://solver.example/v1"
    )
    assert config.models.operational_base_url("judge", environment) == (
        "https://judge.example/api/v1"
    )
    assert config.fingerprint == original_fingerprint
    assert config.to_dict() == original_dict
    assert config.models.solver.base_url == "http://127.0.0.1:8000/v1"


def test_package_source_hash_is_path_independent_and_content_bound(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for root in (first, second):
        (root / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
        (root / "nested").mkdir()
        (root / "nested" / "b.py").write_text("VALUE = 2\n", encoding="utf-8")

    original = package_source_sha256(first)
    assert package_source_sha256(second) == original

    (second / "nested" / "b.py").write_text("VALUE = 3\n", encoding="utf-8")
    assert package_source_sha256(second) != original


def test_benchmark_preparation_uses_configured_dataset_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = "a" * 40
    config = load_config(
        None,
        overrides={
            "paths": {
                "artifact_root": str(tmp_path / "artifacts"),
                "asset_root": str(tmp_path / "assets"),
            },
            "datasets": {
                "imo_proof": {
                    "name": "custom/imo-proof",
                    "revision": revision,
                    "split": "validation",
                }
            },
        },
    )
    seen: dict[str, tuple[str, str, str]] = {}

    def fake_load(spec: Any, **kwargs: Any) -> list[BenchmarkItem]:
        assert kwargs["check_size"] is True
        seen[spec.name] = (spec.dataset, spec.revision, spec.split)
        raw: dict[str, Any]
        if spec.kind == "answer":
            raw = {"Problem": "P", "Short Answer": "1"}
            item = BenchmarkItem(spec.name, "id", "P", raw, answer="1")
        else:
            rubric_key = "grading_scheme" if spec.name == "proofbench" else (
                "grading_guidelines"
            )
            raw = {"problem": "P", "solution": "S", rubric_key: {"7": "ok"}}
            item = BenchmarkItem(spec.name, "id", "P", raw, solution="S", rubric=raw[rubric_key])
        return [item]

    monkeypatch.setattr(assets, "load_benchmark", fake_load)
    manifest = prepare_benchmark_assets(config)

    assert seen["imo_proof"] == ("custom/imo-proof", revision, "validation")
    assert manifest["benchmarks"]["imo_proof"]["dataset"] == "custom/imo-proof"
    assert manifest["benchmarks"]["imo_proof"]["revision"] == revision


def test_tokenizer_only_preparation_avoids_model_weight_patterns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(
        None,
        overrides={"paths.asset_root": str(tmp_path / "assets")},
    )
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(dict(kwargs))
        destination = Path(kwargs["local_dir"])
        destination.mkdir(parents=True)
        (destination / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        return str(destination)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    manifest = prepare_model_assets(config, tokenizer_only=True)

    assert len(calls) == 2
    assert all("allow_patterns" in call for call in calls)
    assert all("*.safetensors" not in call["allow_patterns"] for call in calls)
    assert all(entry["tokenizer_only"] is True for entry in manifest["models"].values())


def test_schedule_has_expected_counts_reference_omission_and_stable_shards() -> None:
    config = load_config(None)
    rows = _benchmark_rows()
    schedule = build_schedule(rows, config)
    reordered = {
        "imo_answer": rows["imo_answer"],
        "proofbench": rows["proofbench"],
        "imo_proof": rows["imo_proof"],
    }
    repeated = build_schedule(reordered, config)

    assert len(schedule) == 6_060
    assert schedule.to_dict() == repeated.to_dict()
    assert schedule.fingerprint == repeated.fingerprint
    assert len({item.run_id for item in schedule}) == len(schedule)
    assert [item.ordinal for item in schedule] == list(range(len(schedule)))

    counts = Counter((item.benchmark, item.condition) for item in schedule)
    for condition in (
        Condition.DIRECT,
        Condition.GVR,
        Condition.GVR_SUBAGENTS,
        Condition.GVR_REFERENCE,
    ):
        assert counts["imo_proof", condition] == 180
        assert counts["proofbench", condition] == 435
    for condition in (Condition.DIRECT, Condition.GVR, Condition.GVR_SUBAGENTS):
        assert counts["imo_answer", condition] == 1_200
    assert counts["imo_answer", Condition.GVR_REFERENCE] == 0
    assert all(
        item.benchmark != "imo_answer"
        for item in schedule
        if item.condition is Condition.GVR_REFERENCE
    )

    shards = shard_items(schedule, 17)
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
    assert {item.run_id for shard in shards for item in shard} == {
        item.run_id for item in schedule
    }
    assert all(
        item.ordinal % len(shards) == shard_index
        for shard_index, shard in enumerate(shards)
        for item in shard
    )
    assert derive_seed(7, "proof-1", "generator", 0) == derive_seed(
        7, "proof-1", "generator", 0
    )
    assert derive_seed(7, "proof-1", "generator", 0) != derive_seed(
        7, "proof-1", "generator", 1
    )


def test_rationale_score_schedule_selects_only_proof_benchmarks() -> None:
    config = load_config(
        None,
        overrides={
            "evaluation.conditions": [
                "gvr_rationale_score",
                "value_tool_rationale_score",
                "gvr_reference_rationale_score",
            ],
            "evaluation.benchmarks": ["imo_proof", "proofbench"],
            "verifier_feedback.mode": "rationale_score",
        },
    )
    schedule = build_schedule(_benchmark_rows(), config)

    assert len(schedule) == (60 + 145) * 3 * 3
    assert {item.benchmark for item in schedule} == {"imo_proof", "proofbench"}
    assert {item.condition for item in schedule} == {
        Condition.GVR_RATIONALE_SCORE,
        Condition.VALUE_TOOL_RATIONALE_SCORE,
        Condition.GVR_REFERENCE_RATIONALE_SCORE,
    }


def test_schedule_is_frozen_and_write_once(tmp_path: Path) -> None:
    config = load_config(None)
    schedule = _small_schedule(config)
    destination = tmp_path / "schedule.json"

    with pytest.raises(FrozenInstanceError):
        schedule.items[0].seed = 99

    assert write_schedule(destination, schedule) == destination
    original_bytes = destination.read_bytes()
    assert write_schedule(destination, schedule) == destination
    assert destination.read_bytes() == original_bytes
    assert load_schedule(destination) == schedule

    different = build_schedule(
        {
            "imo_proof": [
                {
                    "id": "proof-1",
                    "problem": "Prove that 1 = 1.",
                    "solution": "By reflexivity.",
                }
            ]
        },
        config,
        conditions=(Condition.DIRECT,),
        seeds=(1,),
    )
    with pytest.raises(ArtifactMismatchError, match="refusing to replace"):
        write_schedule(destination, different)
    assert destination.read_bytes() == original_bytes

    tampered = json.loads(destination.read_text(encoding="utf-8"))
    tampered["items"][0]["problem"] = "silently changed"
    destination.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ArtifactMismatchError, match="fingerprint"):
        load_schedule(destination)


def test_artifact_store_new_resume_and_complete(tmp_path: Path) -> None:
    store = _artifact_store(tmp_path)
    identity = {"benchmark": "imo_proof", "ordinal": 0}

    claimed = store.claim("run-lifecycle", identity=identity)
    assert claimed.action is ClaimAction.NEW
    assert claimed.should_run
    assert claimed.handle is not None
    claimed.handle.save_checkpoint("generated", {"candidate": "proof"})
    claimed.handle.close()

    resumed = store.claim("run-lifecycle", identity=identity)
    assert resumed.action is ClaimAction.RESUME
    assert resumed.checkpoint is not None
    assert resumed.checkpoint["state"] == "generated"
    assert resumed.checkpoint["payload"] == {"candidate": "proof"}
    assert resumed.handle is not None
    completed = resumed.handle.finalize({"answer": "proof", "status": "accepted"})

    assert completed["attempt"] == 1
    assert store.is_complete("run-lifecycle")
    assert store.load_result("run-lifecycle") == completed
    assert store.status("run-lifecycle")["status"] == "completed"  # type: ignore[index]

    existing = store.claim("run-lifecycle", identity=identity)
    assert existing.action is ClaimAction.COMPLETE
    assert not existing.should_run
    assert existing.result == completed


def test_artifact_store_tracks_parallel_in_flight_requests(tmp_path: Path) -> None:
    store = _artifact_store(tmp_path)
    claim = store.claim("run-parallel")
    assert claim.handle is not None
    handle = claim.handle
    requests = (("child-a", 11), ("child-b", 13), ("child-c", 17))

    def begin_request(spec: tuple[str, int]) -> None:
        request_id, max_tokens = spec
        handle.begin_request(
            request_id,
            role="subagent",
            max_tokens=max_tokens,
            metadata={"branch": request_id},
        )

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(begin_request, requests))

    checkpoint = handle.load_checkpoint()
    assert checkpoint is not None
    assert set(checkpoint["in_flight_requests"]) == {"child-a", "child-b", "child-c"}
    assert checkpoint["sequence"] == 3
    assert store.status("run-parallel")["in_flight_count"] == 3  # type: ignore[index]

    handle.complete_request(
        "child-b",
        state="collecting_children",
        payload={"completed": ["child-b"]},
        usage={"prompt_tokens": 5, "completion_tokens": 7},
    )
    checkpoint = handle.load_checkpoint()
    assert checkpoint is not None
    assert set(checkpoint["in_flight_requests"]) == {"child-a", "child-c"}
    with pytest.raises(ArtifactError, match="requests are in flight"):
        handle.finalize({"status": "accepted"})

    handle.complete_request(
        "child-a",
        state="collecting_children",
        payload={"completed": ["child-a", "child-b"]},
        usage={"prompt_tokens": 3, "completion_tokens": 4},
    )
    handle.complete_request(
        "child-c",
        state="synthesizing",
        payload={"completed": ["child-a", "child-b", "child-c"]},
        usage={"prompt_tokens": 8, "completion_tokens": 9},
    )
    checkpoint = handle.load_checkpoint()
    assert checkpoint is not None
    assert checkpoint["in_flight_requests"] == {}
    assert checkpoint["in_flight_request"] is None

    result = handle.finalize({"status": "accepted", "answer": "candidate"})
    events = read_jsonl(
        store.run_dir("run-parallel") / "attempts" / "000001" / "events.jsonl"
    )
    completed_request_ids = {
        event["request_id"]
        for event in events
        if event["event"] == "request_completed"
    }
    assert completed_request_ids == {"child-a", "child-b", "child-c"}
    assert result["status"] == "accepted"


def test_interrupted_requests_invalidate_attempt_with_usage_upper_bound(
    tmp_path: Path,
) -> None:
    store = _artifact_store(tmp_path)
    first = store.claim("run-interrupted")
    assert first.handle is not None
    first.handle.begin_request("generator", role="generator", max_tokens=101)
    first.handle.begin_request("subagent", role="subagent", max_tokens=37)
    first.handle.close()

    recovered = store.claim("run-interrupted")
    assert recovered.action is ClaimAction.NEW
    assert recovered.handle is not None
    assert recovered.handle.attempt == 2

    invalidations = store.invalidations("run-interrupted")
    assert len(invalidations) == 1
    invalidation = invalidations[0]
    assert invalidation["reason"] == "interrupted_request_with_unknown_usage"
    assert invalidation["unknown_usage"] is True
    assert invalidation["unknown_usage_upper_bound"] == 138
    assert set(invalidation["in_flight_requests"]) == {"generator", "subagent"}
    assert invalidation["last_checkpoint_sequence"] == 2
    recovered.handle.close()


def test_sglang_command_has_exact_qwen_parsers_without_speculative_decoding() -> None:
    command = build_sglang_command(
        "/models/qwen",
        "Qwen/Qwen3.5-9B",
        host="10.0.0.8",
        port=8123,
        context_length=123_456,
        tensor_parallel_size=2,
        mem_fraction_static=0.75,
        executable="sglang-test",
        extra_args=("--log-level", "warning"),
    )

    assert command == [
        "sglang-test",
        "serve",
        "--model-path",
        "/models/qwen",
        "--served-model-name",
        "Qwen/Qwen3.5-9B",
        "--host",
        "10.0.0.8",
        "--port",
        "8123",
        "--context-length",
        "123456",
        "--tp-size",
        "2",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--mem-fraction-static",
        "0.75",
        "--enable-custom-logit-processor",
        "--attention-backend",
        "triton",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--log-level",
        "warning",
    ]
    assert not any(argument.startswith("--speculative") for argument in command)


def test_slurm_launcher_caps_both_gpu_arrays_at_128_by_default() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    launcher = (repository_root / "scripts" / "submit_slurm.sh").read_text(
        encoding="utf-8"
    )

    assert (
        "readonly MAX_CONCURRENT_GPUS=${VALUE_AS_TOOL_MAX_CONCURRENT_GPUS:-128}"
        in launcher
    )
    assert '--array="0-$((SOLVE_SHARDS - 1))%${SOLVE_ARRAY_CONCURRENCY}"' in launcher
    assert '--array="0-$((JUDGE_SHARDS - 1))%${MAX_CONCURRENT_GPUS}"' in launcher


def test_slurm_gpu_runner_reserves_and_pins_sglang_nccl_port() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    runner = (repository_root / "scripts" / "slurm" / "run_gpu.sbatch").read_text(
        encoding="utf-8"
    )

    assert "reserve_tcp_port" in runner
    assert "RESERVED_PORT_FDS" in runner
    assert "[[ $model_role == solver && $SOLVER_BACKEND == sglang ]]" in runner
    assert "readonly SGLANG_NCCL_PORT=${RESERVED_PORTS[1]}" in runner
    assert "--extra-arg=--nccl-port" in runner
    assert '"--extra-arg=${SGLANG_NCCL_PORT}"' in runner


def test_vllm_commands_have_exact_solver_and_judge_parsers() -> None:
    solver = build_vllm_command(
        "/models/qwen",
        "Qwen/Qwen3.5-9B",
        role="solver",
        host="10.0.0.9",
        port=8234,
        context_length=234_567,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.85,
        max_num_seqs=7,
        executable="vllm-test",
        extra_args=("--dtype", "bfloat16"),
    )
    assert solver == [
        "vllm-test",
        "serve",
        "/models/qwen",
        "--served-model-name",
        "Qwen/Qwen3.5-9B",
        "--host",
        "10.0.0.9",
        "--port",
        "8234",
        "--max-model-len",
        "234567",
        "--tensor-parallel-size",
        "4",
        "--gpu-memory-utilization",
        "0.85",
        "--max-num-seqs",
        "7",
        "--enable-prefix-caching",
        "--trust-remote-code",
        "--reasoning-parser",
        "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--language-model-only",
        "--dtype",
        "bfloat16",
    ]

    judge = build_vllm_command(
        "/models/judge",
        "openai/gpt-oss-20b",
        role="judge",
        executable="vllm-test",
    )
    assert judge == [
        "vllm-test",
        "serve",
        "/models/judge",
        "--served-model-name",
        "openai/gpt-oss-20b",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--max-model-len",
        "262144",
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.9",
        "--max-num-seqs",
        "16",
        "--enable-prefix-caching",
        "--trust-remote-code",
        "--reasoning-parser",
        "openai_gptoss",
        "--moe-backend",
        "triton",
    ]


def test_asset_and_config_path_helpers_are_side_effect_free(tmp_path: Path) -> None:
    artifact_root = tmp_path / "not-created-artifacts"
    asset_root = tmp_path / "not-created-assets"
    revision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"

    benchmark = prepared_benchmark_path(artifact_root, "imo_proof")
    model = model_snapshot_path(asset_root, "Qwen/Qwen3.5-9B", revision)
    resolved = PathsConfig(
        artifact_root=Path("runs/default"),
        asset_root=Path("cache/assets"),
    ).resolve(tmp_path)

    assert benchmark == artifact_root / "prepared" / "benchmarks" / "imo_proof.jsonl"
    assert model == asset_root / "models" / "Qwen--Qwen3.5-9B" / revision
    assert resolved.artifact_root == (tmp_path / "runs/default").resolve()
    assert resolved.asset_root == (tmp_path / "cache/assets").resolve()
    assert not artifact_root.exists()
    assert not asset_root.exists()
