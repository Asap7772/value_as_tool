from __future__ import annotations

from typing import Any

import pytest

from value_as_tool.metrics import compute_metrics, summarize_metrics
from value_as_tool.reporting import render_markdown

DIRECT_SOURCE = "1" * 64
DIRECT_SOURCE_V2 = "2" * 64
REFERENCE_SOURCE = "3" * 64


def _row(
    method_id: str,
    source_sha256: str,
    item_id: str,
    seed: int,
    score: int,
    *,
    benchmark: str = "imo_proof",
    access: str = "blind",
) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "item_id": item_id,
        "condition": method_id,
        "harness_id": method_id,
        "harness_entrypoint": f"example.{method_id}:AgentHarness",
        "harness_source_sha256": source_sha256,
        "harness_access": access,
        "seed": seed,
        "solve_status": "completed",
        "judge_status": "completed",
        "score": score,
        "generated_tokens": 10,
        "total_tokens": 20,
        "usage_exact": True,
    }


def test_harness_metrics_do_not_pool_source_revisions() -> None:
    rows = [
        _row("direct", DIRECT_SOURCE, "p", 0, 7),
        _row("direct", DIRECT_SOURCE_V2, "p", 0, 0),
    ]

    summaries = summarize_metrics(rows, expected_seeds=(0,))

    assert [row["harness_source_sha256"] for row in summaries] == [
        DIRECT_SOURCE,
        DIRECT_SOURCE_V2,
    ]
    assert [row["raw_success_rate"] for row in summaries] == [1.0, 0.0]


def test_reference_harness_absence_is_na_and_bootstrap_pairs_only_applicable_items() -> None:
    rows = [
        _row("direct", DIRECT_SOURCE, "with-reference", 0, 0),
        _row("direct", DIRECT_SOURCE, "without-reference", 0, 7),
        _row(
            "gvr_reference",
            REFERENCE_SOURCE,
            "with-reference",
            0,
            7,
            access="reference_assisted",
        ),
        {
            **_row(
                "direct",
                DIRECT_SOURCE,
                "answer",
                0,
                0,
                benchmark="imo_answer",
            ),
            "score": None,
            "correct": True,
        },
    ]

    report = compute_metrics(
        rows,
        bootstrap_samples=20,
        ks=(1,),
        expected_seeds=(0,),
    )
    reference_summary = next(
        row for row in report["summary"] if row["method_id"] == "gvr_reference"
    )
    comparison = next(
        row
        for row in report["paired_bootstrap"]
        if row["method_id"] == "gvr_reference"
        and row["metric"] == "raw_success_rate"
    )

    assert reference_summary["problems"] == 1
    assert reference_summary["scheduled"] == 1
    assert comparison["paired_n"] == 1
    reference_line = next(
        line
        for line in render_markdown(report).splitlines()
        if line.startswith(f"| gvr_reference @ {REFERENCE_SOURCE[:12]} |")
    )
    assert reference_line.endswith("| 100.00% | N/A |")


def test_nested_harness_identity_is_supported_but_conflicts_are_rejected() -> None:
    row = _row("direct", DIRECT_SOURCE, "p", 0, 7)
    nested = {
        key: row.pop(key)
        for key in (
            "harness_id",
            "harness_entrypoint",
            "harness_source_sha256",
            "harness_access",
        )
    }
    row["request"] = {
        "harness_id": nested["harness_id"],
        "metadata": nested,
    }

    summary = summarize_metrics([row], expected_seeds=(0,))[0]
    assert summary["harness_source_sha256"] == DIRECT_SOURCE

    row["harness_source_sha256"] = DIRECT_SOURCE_V2
    with pytest.raises(ValueError, match="inconsistent harness_source_sha256"):
        summarize_metrics([row], expected_seeds=(0,))


def test_denormalized_method_id_cannot_override_harness_identity() -> None:
    row = _row("direct", DIRECT_SOURCE, "p", 0, 7)
    row["method_id"] = "gvr"

    with pytest.raises(ValueError, match="does not match resolved method identity"):
        summarize_metrics([row], expected_seeds=(0,))
