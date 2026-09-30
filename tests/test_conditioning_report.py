from __future__ import annotations

import asyncio
import copy
import csv
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from test_conditioning import Counter as TokenCounter
from test_conditioning import FakeClient, build_fixture

from value_as_tool.client import ChatClientError
from value_as_tool.conditioning import enumerate_summary_tasks, freeze_bank, summarize_task
from value_as_tool.conditioning_report import (
    factorial_comparisons,
    paired_factorial_bootstrap,
    write_conditioning_report,
)
from value_as_tool.harnesses import ATTEMPT_CONDITIONED_HARNESSES, resolve_harness
from value_as_tool.storage import read_json


@pytest.fixture(scope="module")
def frozen_bank(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, dict[str, Any]]:
    """Use the real bank/freezer, including a discarded receipt and unknown call."""
    bank, context, _ = build_fixture(
        tmp_path_factory.mktemp("report-bank"),
        scores={"mixed": [7, 0, 0], "positive": [7, 7, 7], "negative": [0, 0, 0]},
    )

    async def summarize() -> None:
        task = next(
            task
            for task in enumerate_summary_tasks(bank)
            if task["problem_id"] == "mixed"
            and task["mode"] == "solution_summary"
            and task["kind"] == "map"
        )
        with pytest.raises(ChatClientError):
            await summarize_task(
                context,
                task["task_id"],
                token_counter=TokenCounter(),
                client=FakeClient([("truncated summary", "length"), ChatClientError("lost")]),
            )
        for task in enumerate_summary_tasks(bank):
            await summarize_task(
                context, task["task_id"], client=FakeClient(), token_counter=TokenCounter()
            )

    asyncio.run(summarize())
    manifest = freeze_bank(bank)
    context.config.conditioning.manifest_sha256 = manifest["manifest_sha256"]
    context.config.evaluation = SimpleNamespace(
        harnesses=ATTEMPT_CONDITIONED_HARNESSES,
        benchmarks=("imo_proof",),
        seeds=(8, 9),
        bootstrap_samples=20,
    )
    return context, manifest


def _rows(context: Any, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "benchmark": benchmark,
            "problem_id": problem,
            "harness_id": spec.harness_id,
            "seed": seed,
            "solve_status": "completed",
            "judge_status": "completed",
            "score": 7 if seed == 8 else 0,
            "generated_tokens": 10,
            "reasoning_tokens": 7,
            "prompt_tokens": 30,
            "judge_usage": {"completion_tokens": 2},
        }
        for benchmark in context.config.evaluation.benchmarks
        for problem in manifest["problems"][benchmark]
        for spec in map(resolve_harness, context.config.evaluation.harnesses)
        for seed in context.config.evaluation.seeds
    ]


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_report_uses_frozen_strata_and_accounts_for_discarded_and_unknown_summary_calls(
    frozen_bank: tuple[Any, dict[str, Any]],
    tmp_path: Path,
) -> None:
    context, manifest = frozen_bank
    result = write_conditioning_report(context, _rows(context, manifest), tmp_path)
    assert result["complete"] and result["incomplete_cells"] == 0
    assert result["comparisons_per_benchmark_per_k"] == 60
    strata = _csv(tmp_path / "conditioning-strata.csv")
    assert Counter(row["bank_composition"] for row in strata) == {
        "mixed": 48,
        "positive_only": 48,
        "negative_only": 48,
    }
    assert {row["problems"] for row in strata} == {"1"}
    costs = read_json(tmp_path / "conditioning-preprocessing.json")
    assert costs["source_usage"]["completion_tokens"] == 180
    assert costs["source_judge_usage"]["completion_tokens"] == 180
    assert costs["summary_usage"] == manifest["summary_usage"]
    assert costs["summary_usage"]["completion_tokens"] == 240
    assert costs["summary_all_attempt_usage"] == manifest["summary_all_attempt_usage"]
    assert costs["summary_all_attempt_usage"]["completion_tokens"] == 250
    assert costs["summary_unknown_usage"] == manifest["summary_unknown_usage"]
    assert costs["summary_unknown_usage"]["requests"] == 1
    unknown_bound = manifest["summary_unknown_usage"]["completion_tokens_upper_bound"]
    assert unknown_bound > 0
    solution = costs["by_mode"]["solutions"]
    summary = costs["by_mode"]["solution_summary"]
    thinking = costs["by_mode"]["thinking_summary"]
    assert costs["method_count"] == 24
    assert {value["method_count"] for value in costs["by_mode"].values()} == {8}
    assert solution["standalone_usage"]["completion_tokens"] == 180
    assert summary["standalone_usage"]["completion_tokens"] == 310
    assert thinking["standalone_usage"]["completion_tokens"] == 300
    assert solution["amortized_usage"]["completion_tokens"] == 7.5
    assert summary["amortized_usage"]["completion_tokens"] == 23.75
    assert thinking["amortized_usage"]["completion_tokens"] == 22.5
    assert all(
        value["standalone_judge_usage"]["completion_tokens"] == 180
        and value["amortized_judge_usage"]["completion_tokens"] == 7.5
        for value in costs["by_mode"].values()
    )
    assert solution["usage_exact"] and thinking["usage_exact"] and not summary["usage_exact"]
    assert summary["standalone_unknown_usage"]["completion_tokens_upper_bound"] == unknown_bound
    assert summary["amortized_unknown_usage"]["completion_tokens_upper_bound"] == unknown_bound / 8
    assert (
        sum(
            value["method_count"] * value["amortized_usage"]["completion_tokens"]
            for value in costs["by_mode"].values()
        )
        == 430
    )  # Qwen source attempts + all persisted Qwen summary receipts; judges stay separate.
    curves = read_json(tmp_path / "conditioning-plot-data.json")["curves"]
    raw = next(row for row in curves if row["conditioning_mode"] == "solutions" and row["k"] == 1)
    assert raw["standalone_expected_generated_tokens"] == 70
    assert raw["amortized_expected_generated_tokens"] == 12.5
    assert raw["standalone_expected_judge_generated_tokens"] == 62
    assert raw["amortized_expected_judge_generated_tokens"] == 4.5
    summary_curve = next(
        row for row in curves if row["conditioning_mode"] == "solution_summary" and row["k"] == 2
    )
    assert summary_curve["standalone_expected_generated_tokens"] == pytest.approx(310 / 3 + 20)
    assert summary_curve["standalone_expected_judge_generated_tokens"] == 64
    assert summary_curve["standalone_unknown_completion_tokens_upper_bound"] == unknown_bound / 3
    assert not summary_curve["preprocessing_usage_exact"]


def test_amortization_uses_selected_methods_and_mode_counts(
    frozen_bank: tuple[Any, dict[str, Any]],
    tmp_path: Path,
) -> None:
    context, manifest = copy.deepcopy(frozen_bank)
    context.config.evaluation.harnesses = tuple(
        entrypoint
        for entrypoint in ATTEMPT_CONDITIONED_HARNESSES
        if resolve_harness(entrypoint).harness_id
        in {
            "attempt_solutions_gvr_legacy_no_gold",
            "attempt_solution_summary_gvr_legacy_no_gold",
            "attempt_solution_summary_gvr_legacy_gold",
        }
    )
    write_conditioning_report(context, _rows(context, manifest), tmp_path)
    costs = read_json(tmp_path / "conditioning-preprocessing.json")
    assert costs["method_count"] == 3
    assert set(costs["by_mode"]) == {"solutions", "solution_summary"}
    raw = costs["by_mode"]["solutions"]
    summary = costs["by_mode"]["solution_summary"]
    assert raw["method_count"] == 1 and summary["method_count"] == 2
    assert raw["amortized_usage"]["completion_tokens"] == 60
    assert summary["amortized_usage"]["completion_tokens"] == 125
    assert raw["amortized_judge_usage"]["completion_tokens"] == 60
    assert summary["amortized_unknown_usage"]["requests"] == 0.5
    assert costs["summary_usage"]["completion_tokens"] == 120
    assert costs["summary_all_attempt_usage"]["completion_tokens"] == 130


def test_judge_costs_do_not_change_qwen_cost_curves(
    frozen_bank: tuple[Any, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, manifest = copy.deepcopy(frozen_bank)
    rows = _rows(context, manifest)
    write_conditioning_report(context, rows, tmp_path / "original")
    for problem in manifest["problems"]["imo_proof"].values():
        problem["source_judge_usage"] = {
            key: value * 100 if isinstance(value, int) else value
            for key, value in problem["source_judge_usage"].items()
        }
    for row in rows:
        row["judge_usage"]["completion_tokens"] *= 100
    monkeypatch.setattr("value_as_tool.conditioning.load_frozen_manifest", lambda *args: manifest)
    write_conditioning_report(context, rows, tmp_path / "expensive-judge")
    original = read_json(tmp_path / "original" / "conditioning-plot-data.json")["curves"]
    expensive = read_json(tmp_path / "expensive-judge" / "conditioning-plot-data.json")["curves"]
    for before, after in zip(original, expensive, strict=True):
        for policy in ("standalone", "amortized"):
            assert (
                before[f"{policy}_expected_generated_tokens"]
                == after[f"{policy}_expected_generated_tokens"]
            )
            assert after[f"{policy}_expected_judge_generated_tokens"] == pytest.approx(
                100 * before[f"{policy}_expected_judge_generated_tokens"]
            )


@pytest.mark.parametrize("incomplete_benchmark", ["imo_proof", "proofbench"])
def test_any_incomplete_benchmark_suppresses_all_significance(
    frozen_bank: tuple[Any, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    incomplete_benchmark: str,
) -> None:
    context, manifest = copy.deepcopy(frozen_bank)
    context.config.evaluation.benchmarks = ("proofbench", "imo_proof")
    manifest["problems"]["proofbench"] = copy.deepcopy(manifest["problems"]["imo_proof"])
    monkeypatch.setattr("value_as_tool.conditioning.load_frozen_manifest", lambda *args: manifest)
    rows = _rows(context, manifest)
    missing = next(row for row in rows if row["benchmark"] == incomplete_benchmark)
    missing.update(judge_status="unjudged", score=None)
    result = write_conditioning_report(context, rows, tmp_path)
    assert result["complete"] is False
    assert result["incomplete_cells"] == 1
    assert _csv(tmp_path / "conditioning-significance.csv") == []
    assert {row["benchmark"] for row in _csv(tmp_path / "conditioning-pass-at-k.csv")} == {
        "imo_proof",
        "proofbench",
    }


@pytest.mark.parametrize("omission", ["seed", "method", "problem", "all", "duplicate", "extra"])
def test_report_rejects_a_grid_that_differs_from_the_frozen_bank_and_config(
    frozen_bank: tuple[Any, dict[str, Any]],
    tmp_path: Path,
    omission: str,
) -> None:
    context, manifest = frozen_bank
    rows = _rows(context, manifest)
    if omission == "seed":
        rows.pop()
    elif omission == "method":
        rows = [row for row in rows if row["harness_id"] != rows[0]["harness_id"]]
    elif omission == "problem":
        rows = [row for row in rows if row["problem_id"] != rows[0]["problem_id"]]
    elif omission == "all":
        rows = []
    elif omission == "duplicate":
        rows.append(rows[0])
    else:
        rows.append({**rows[0], "harness_id": "unexpected_method"})
    with pytest.raises(ValueError, match="exactly the scheduled sample grid"):
        write_conditioning_report(context, rows, tmp_path)


def test_finalized_failures_remain_zeroes_in_the_scheduled_denominator(
    frozen_bank: tuple[Any, dict[str, Any]],
    tmp_path: Path,
) -> None:
    context, manifest = frozen_bank
    rows = _rows(context, manifest)
    target = rows[0]["harness_id"]
    for row in rows:
        if row["harness_id"] == target:
            row.update(solve_status="context_exhausted", judge_status="solve_failed", score=7)
    result = write_conditioning_report(context, rows, tmp_path)
    assert result["complete"]
    curves = _csv(tmp_path / "conditioning-pass-at-k.csv")
    assert {float(row["pass_at_k"]) for row in curves if row["method"] == target} == {0}
    tokens = next(
        row for row in _csv(tmp_path / "conditioning-tokens.csv") if row["method"] == target
    )
    assert tokens["context_exhausted"] == "6"
    assert len(_csv(tmp_path / "conditioning-significance.csv")) == 60 * 2


def test_factorial_comparisons_and_bootstrap_keep_problem_pairs() -> None:
    specs = [resolve_harness(value) for value in ATTEMPT_CONDITIONED_HARNESSES]
    factors = {
        spec.harness_id: (
            spec.conditioning_mode,
            "value" if "value_tool" in str(spec.condition) else "gvr",
            "rationale" if "rationale" in str(spec.condition) else "legacy",
            spec.requires_reference,
        )
        for spec in specs
    }
    comparisons = factorial_comparisons(list(factors), factors)
    assert len(comparisons) == 60
    assert Counter(factor for _, _, factor in comparisons) == {
        "conditioning": 24,
        "interaction": 12,
        "feedback": 12,
        "gold_reference": 12,
    }
    # Every problem has a different baseline, but its treatment difference is
    # identical; paired resampling must preserve the zero-variance contrast.
    values = np.array([[[0.0], [0.25]], [[0.5], [0.75]], [[0.25], [0.5]]])
    result = paired_factorial_bootstrap(
        values,
        ["left", "right"],
        [("left", "right", "feedback")],
        samples=50,
        seed=2,
    )[0]
    assert result["delta"] == -0.25
    assert result["standard_error"] == 0
    assert result["simultaneous_ci_low"] == result["simultaneous_ci_high"] == -0.25
