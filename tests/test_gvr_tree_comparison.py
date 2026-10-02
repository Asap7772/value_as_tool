"""Statistics behind compare_gvr_tree_variants.py."""

from __future__ import annotations

import importlib
import random
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _compare(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("compare_gvr_tree_variants")


def test_auroc_counts_ties_half_and_needs_both_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    compare = _compare(monkeypatch)
    assert compare.auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert compare.auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert compare.auroc([0.5, 0.5], [True, False]) == 0.5
    assert compare.auroc([0.3, 0.7], [True, True]) is None


def test_calibration_reports_brier_and_ece(monkeypatch: pytest.MonkeyPatch) -> None:
    compare = _compare(monkeypatch)
    report = compare.calibration([0.9, 0.9, 0.1, 0.1], [True, False, False, False])
    assert report["brier"] == pytest.approx((0.01 + 0.81 + 0.01 + 0.01) / 4)
    assert report["ece"] == pytest.approx(0.5 * 0.4 + 0.5 * 0.1)
    assert report["accuracy"] == 0.25


def test_mcnemar_holm_and_sign_flip(monkeypatch: pytest.MonkeyPatch) -> None:
    compare = _compare(monkeypatch)
    six_to_none = compare.mcnemar_p([True] * 6 + [False] * 4, [False] * 6 + [False] * 4)
    assert six_to_none == {"only_first": 6, "only_second": 0, "p": pytest.approx(2 / 64)}
    assert compare.mcnemar_p([True], [True])["p"] == 1.0
    adjusted = compare.holm({"a": 0.01, "b": 0.04, "c": None})
    assert adjusted == {"a": pytest.approx(0.02), "b": pytest.approx(0.04), "c": None}
    rng = random.Random(0)
    assert compare.sign_flip_p([0.1] * 8, samples=100, rng=rng) == pytest.approx(2 / 256)
    assert compare.sign_flip_p([0.1, -0.1], samples=100, rng=rng) == 1.0


def _node(
    call: int, parent: int | None, cycle: int, answer: str, correct: bool, spine: bool
) -> dict[str, Any]:
    return {
        "node_call_index": call,
        "parent_call_index": parent,
        "cycle": cycle,
        "on_spine": spine,
        "extracted_answer": answer,
        "judge_correct": correct,
        "answer_changed": None if parent is None else answer != "1",
        "plan_show_current_solution": None if parent is None else call % 2 == 0,
    }


def test_tree_metrics_measure_diversity_against_the_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    compare = _compare(monkeypatch)
    nodes = [
        _node(0, None, 1, "1", False, True),
        _node(1, 0, 2, "1", False, True),
        _node(2, 0, 2, "2", True, False),
        _node(3, 1, 3, "1", False, True),
        _node(4, 1, 3, "3", False, False),
    ]
    plans = [
        {
            "point": 1,
            "valid": True,
            "title": "Restart",
            "brief": "solve again",
            "show_current_solution": False,
            "planner_recovered": False,
            "success_probability": 0.4,
            "node_judge_correct": True,
        },
        {
            "point": 1,
            "valid": True,
            "title": "restart",
            "brief": "solve it again",
            "show_current_solution": True,
            "planner_recovered": False,
            "success_probability": 0.6,
            "node_judge_correct": False,
        },
    ]
    result = compare.tree_metrics(
        {"generated_tokens": 10, "status": "cycle_limit"}, nodes, [], plans
    )
    metrics = result["metrics"]
    assert metrics["answer_change_rate"] == 0.5
    assert metrics["mixed_point_share"] == 0.5
    assert metrics["distinct_answers"] == 3 and metrics["semantic_distinct_answers"] == 3
    assert metrics["spine_never_changes"] and not metrics["single_answer"]
    assert metrics["any_correct"] and metrics["mixed_tree"] and metrics["final_correct"] is False
    assert metrics["distinct_titles_per_point"] == 1
    assert metrics["show_current_solution_share"] == 0.5
    assert sorted(result["transitions"]) == [
        (False, False, False),
        (False, False, False),
        (False, False, True),
        (False, True, True),
    ]
    assert result["plans"] == [(0.4, True), (0.6, False)]
