from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from value_as_tool import conditioning
from value_as_tool.config import ConditioningConfig, ExperimentConfig, load_config
from value_as_tool.harnesses import ATTEMPT_CONDITIONED_HARNESSES, ATTEMPT_CONDITIONING_MODES
from value_as_tool.metrics import summarize_metrics
from value_as_tool.pipeline import _conditioning_manifest, _result_matches_item
from value_as_tool.schedule import ScheduleItem, build_schedule, stable_fingerprint
from value_as_tool.schemas import Condition
from value_as_tool.storage import ArtifactMismatchError

CONFIG_PATH = Path(__file__).resolve().parents[1] / "experiment_qwen35_9b_attempt_conditioning.yaml"
EVIDENCE_SENTINEL = "Private prior-attempt content must never be embedded in a schedule."


def _problems(*, full: bool = False) -> dict[str, list[dict[str, str]]]:
    counts = {"imo_proof": 60, "proofbench": 145} if full else {"imo_proof": 1}
    return {
        benchmark: [
            {
                "id": f"{benchmark}-{index}",
                "problem": f"Prove synthetic statement {benchmark} {index}.",
                "solution": f"Synthetic reference proof {benchmark} {index}.",
            }
            for index in range(count)
        ]
        for benchmark, count in counts.items()
    }


def _config(*, frozen: bool = True) -> ExperimentConfig:
    overrides = (
        {"conditioning.bank_sha256": "b" * 64, "conditioning.manifest_sha256": "c" * 64}
        if frozen
        else None
    )
    return load_config(CONFIG_PATH, overrides=overrides)


def _manifest(problems: Any, config: ExperimentConfig) -> dict[str, Any]:
    baseline = build_schedule(problems, config, conditions=(Condition.DIRECT,), seeds=(8,))
    result: dict[str, Any] = {"bank_sha256": "b" * 64, "problems": {}}
    for item in baseline:
        result["problems"].setdefault(item.benchmark, {})[item.problem_id] = {
            "problem_fingerprint": item.problem_fingerprint,
            "packs": {
                mode: {
                    "sha256": stable_fingerprint([item.benchmark, item.problem_id, mode]),
                    "content": EVIDENCE_SENTINEL,
                }
                for mode in ATTEMPT_CONDITIONING_MODES
            },
        }
    return result


def _small_schedule():  # type: ignore[no-untyped-def]
    problems = _problems()
    config = _config()
    return build_schedule(problems, config, conditioning_manifest=_manifest(problems, config))


def test_full_conditioning_schedule_has_39360_balanced_trajectories() -> None:
    config = _config()
    problems = _problems(full=True)
    schedule = build_schedule(problems, config, conditioning_manifest=_manifest(problems, config))
    assert len(schedule) == 24 * 205 * 8 == 39_360
    assert len({item.run_id for item in schedule}) == len(schedule)
    assert Counter(item.benchmark for item in schedule) == {
        "imo_proof": 24 * 60 * 8,
        "proofbench": 24 * 145 * 8,
    }
    assert set(Counter(item.harness_id for item in schedule).values()) == {205 * 8}
    assert set(Counter((item.benchmark, item.problem_id) for item in schedule).values()) == {24 * 8}
    assert {item.seed for item in schedule} == set(range(8, 16))
    assert {item.harness_entrypoint for item in schedule} == set(ATTEMPT_CONDITIONED_HARNESSES)


def test_checked_in_config_pins_model_budget_and_disjoint_source_and_evaluation_seeds() -> None:
    config = _config(frozen=False)
    assert config.models.solver.name == "Qwen/Qwen3.5-9B"
    assert config.models.solver.revision == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert config.models.judge.name == "openai/gpt-oss-20b"
    assert config.evaluation.judge_reasoning_effort == "medium"
    assert config.budget.generated_tokens == 8_388_608
    assert config.budget.context_tokens == 262_144
    assert config.sampling.enable_thinking
    assert config.runtime.solver_tensor_parallel_size == 1
    assert config.runtime.max_concurrency == 2
    assert config.conditioning is not None
    assert config.conditioning.source_seeds == tuple(range(8))
    assert config.evaluation.seeds == tuple(range(8, 16))
    assert set(config.evaluation.seeds).isdisjoint(config.conditioning.source_seeds)
    with pytest.raises(ValidationError, match="disjoint"):
        load_config(CONFIG_PATH, overrides={"evaluation.seeds": [0, 8]})


@pytest.mark.parametrize("source", [None, "unfrozen", "missing_argument"])
def test_conditioned_schedule_blocks_missing_frozen_bank(source: str | None) -> None:
    config = _config(frozen=source != "unfrozen")
    manifest = _manifest(_problems(), config)
    if source is None:
        config = config.model_copy(update={"conditioning": None})
    elif source == "missing_argument":
        manifest = None
    with pytest.raises(ValueError, match="frozen conditioning manifest"):
        build_schedule(_problems(), config, conditioning_manifest=manifest)


def test_pipeline_preparation_blocks_unfrozen_manifest_without_loading_bank_module() -> None:
    with pytest.raises(ValueError, match="freeze the conditioning manifest"):
        _conditioning_manifest(SimpleNamespace(config=_config(frozen=False)))


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("source_seeds", [0, 1], "source seeds"),
        ("source_seeds", [False, *range(1, 8)], "source seeds"),
        ("source_seeds", None, "source seeds"),
        ("source_model", None, "source model"),
        ("source_model", {"name": "other", "revision": "a" * 40}, "source model"),
        ("summary_settings", {}, "summary model"),
        ("summary_settings", {"model": {"name": "other"}}, "summary model"),
        ("bank_sha256", "0" * 64, "bank digest"),
    ],
)
def test_pipeline_rejects_frozen_evidence_from_different_protocol(
    monkeypatch: pytest.MonkeyPatch, field: str, value: Any, error: str
) -> None:
    config = _config()
    model = {"name": config.models.solver.name, "revision": config.models.solver.revision}
    manifest = {
        "bank_sha256": config.conditioning.bank_sha256,
        "source_seeds": list(config.conditioning.source_seeds),
        "source_model": model,
        "summary_settings": {"model": model},
    }
    monkeypatch.setattr(conditioning, "load_frozen_manifest", lambda *args: manifest)
    context = SimpleNamespace(config=config)
    assert _conditioning_manifest(context) is manifest
    manifest[field] = value
    with pytest.raises(ArtifactMismatchError, match=error):
        _conditioning_manifest(context)


def test_schedule_seed_override_cannot_reuse_bank_draws() -> None:
    config = _config()
    problems = _problems()
    with pytest.raises(ValueError, match="disjoint"):
        build_schedule(
            problems, config, seeds=(0,), conditioning_manifest=_manifest(problems, config)
        )


def test_changed_problem_and_missing_gold_reference_are_rejected() -> None:
    config = _config()
    problems = _problems()
    manifest = _manifest(problems, config)
    changed = copy.deepcopy(problems)
    changed["imo_proof"][0]["problem"] += " A new extra hypothesis."
    with pytest.raises(ValueError, match="problem fingerprint mismatch"):
        build_schedule(changed, config, conditioning_manifest=manifest)
    no_gold = copy.deepcopy(problems)
    no_gold["imo_proof"][0].pop("solution")
    with pytest.raises(ValueError, match="gold arm requires reference"):
        build_schedule(no_gold, config, conditioning_manifest=_manifest(no_gold, config))


def test_changed_pack_changes_only_its_64_trajectory_ids() -> None:
    config = _config()
    problems = _problems()
    manifest = _manifest(problems, config)
    original = build_schedule(problems, config, conditioning_manifest=manifest)
    changed_manifest = copy.deepcopy(manifest)
    changed_manifest["problems"]["imo_proof"]["imo_proof-0"]["packs"]["solution_summary"][
        "sha256"
    ] = "d" * 64
    changed = build_schedule(problems, config, conditioning_manifest=changed_manifest)
    assert original.config_fingerprint == changed.config_fingerprint
    assert original.fingerprint != changed.fingerprint
    differences = [
        before
        for before, after in zip(original, changed, strict=True)
        if before.run_id != after.run_id
    ]
    assert len(differences) == 8 * 8
    assert {item.conditioning_mode for item in differences} == {"solution_summary"}


def test_conditioned_schedule_round_trip_binds_digest_without_embedding_content() -> None:
    for item in _small_schedule():
        serialized = item.to_dict()
        assert ScheduleItem.from_dict(serialized) == item
        assert EVIDENCE_SENTINEL not in json.dumps(serialized)
        assert "verifier_evidence" not in serialized
        assert serialized["conditioning_mode"] == item.conditioning_mode
        assert serialized["conditioning_pack_sha256"] == item.conditioning_pack_sha256
        request = item.to_request(
            verifier_evidence={
                "mode": item.conditioning_mode,
                "pack_sha256": item.conditioning_pack_sha256,
                "content": EVIDENCE_SENTINEL,
            }
        )
        assert request.verifier_evidence["content"] == EVIDENCE_SENTINEL
        assert request.metadata["conditioning_pack_sha256"] == item.conditioning_pack_sha256


def test_optional_conditioning_preserves_legacy_serialization_and_config_fingerprint() -> None:
    legacy = load_config(None)
    assert legacy.conditioning is None
    assert "conditioning" not in legacy.to_dict()
    assert legacy.fingerprint == "1f2eac2f816e04009f8e8cb0a93394e4e0fb1c3c7bc49a6844c5cab14ed295a8"
    explicit_null = load_config(None, overrides={"conditioning": None})
    assert explicit_null.fingerprint == legacy.fingerprint
    item = build_schedule(_problems(), legacy, conditions=(Condition.DIRECT,), seeds=(0,))[0]
    assert "conditioning_mode" not in item.to_dict()
    assert "conditioning_pack_sha256" not in item.to_dict()
    assert "verifier_evidence" not in item.to_request().to_dict()


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_seeds", [True, 2]),
        ("source_seeds", ["0", 1]),
        ("map_target_tokens", True),
        ("map_target_tokens", "2048"),
        ("summary_target_tokens", False),
        ("max_summary_attempts", "3"),
    ],
)
def test_conditioning_integer_configuration_does_not_coerce_values(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        ConditioningConfig.model_validate(
            {
                "source_artifact_root": "source",
                "bank_root": "bank",
                field: value,
            }
        )


def test_conditioned_metric_rows_keep_all_four_factors_and_access_classes_separate() -> None:
    rows = []
    for item in _small_schedule():
        rows.append(
            {
                **item.to_dict(),
                "condition": item.harness_id,
                "item_id": item.problem_id,
                "solve_status": "completed",
                "judge_status": "completed",
                "score": 7 if item.seed % 2 == 0 else 0,
                "generated_tokens": 10,
                "total_tokens": 20,
                "usage_exact": True,
            }
        )
    summaries = summarize_metrics(rows, expected_seeds=tuple(range(8, 16)))
    assert len(summaries) == 24
    assert len({row["method_id"] for row in summaries}) == 24
    assert Counter(row["harness_access"] for row in summaries) == {
        "attempt_assisted": 12,
        "attempt_and_reference_assisted": 12,
    }
    assert all(row["raw_success_rate"] == 0.5 for row in summaries)


def test_pipeline_rejects_solver_result_with_wrong_evidence_digest() -> None:
    item = _small_schedule()[0]
    request = item.to_request(
        verifier_evidence={
            "mode": item.conditioning_mode,
            "pack_sha256": "0" * 64,
            "content": EVIDENCE_SENTINEL,
        }
    )
    with pytest.raises(ArtifactMismatchError, match="evidence mismatch"):
        _result_matches_item(item, {"run_id": item.run_id, "request": request.to_dict()})
