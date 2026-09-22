"""Deterministic JSON, CSV, and Markdown output for evaluation records."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from value_as_tool.metrics import DEFAULT_COMPARISONS, compute_metrics

CONDITION_ORDER = ("direct", "gvr", "gvr_subagents", "gvr_reference")
BENCHMARK_LABELS = {
    "imo_proof": "IMO-Proof",
    "proofbench": "ProofBench",
    "imo_answer": "IMO-Answer",
}


def load_jsonl(paths: str | Path | Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load one or more result JSONL files without requiring a dataframe."""

    selected = [paths] if isinstance(paths, (str, Path)) else list(paths)
    rows: list[dict[str, Any]] = []
    for raw_path in selected:
        path = Path(raw_path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object")
                rows.append(dict(value))
    return rows


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    from io import StringIO

    target = StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        flattened = {
            key: (
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (Mapping, list, tuple))
                else value
            )
            for key, value in row.items()
        }
        writer.writerow(flattened)
    return target.getvalue()


def _percent(value: Any) -> str:
    return "N/A" if value is None else f"{float(value) * 100:.2f}%"


def _number(value: Any, digits: int = 3) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _ordered(values: Iterable[str], preferred: Sequence[str]) -> list[str]:
    present = set(values)
    return [value for value in preferred if value in present] + sorted(
        present - set(preferred)
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable view of a metrics payload."""

    summary = list(report.get("summary", []))
    curves = list(report.get("curves", []))
    compatibility = list(report.get("compatibility", []))
    bootstrap = list(report.get("paired_bootstrap", []))
    complete = bool(report.get("complete"))
    lines = [
        "# IMO evaluation report",
        "",
        f"Status: **{'complete' if complete else 'incomplete'}**.",
        "",
        (
            "Missing, failed, and unjudged scheduled cells contribute zero. "
            "A condition absent from a benchmark is reported as N/A."
        ),
        "",
        "## QED-compatible headline",
        "",
    ]

    benchmarks = _ordered(
        (str(row["benchmark"]) for row in compatibility),
        ("imo_proof", "proofbench", "imo_answer"),
    )
    conditions = _ordered(
        (str(row["condition"]) for row in compatibility), CONDITION_ORDER
    )
    compatibility_lookup = {
        (str(row["benchmark"]), str(row["condition"])): row
        for row in compatibility
    }
    labels = [BENCHMARK_LABELS.get(name, name) for name in benchmarks]
    lines.extend(
        [
            "| Method | " + " | ".join(labels) + " |",
            "|---|" + "---:|" * len(benchmarks),
        ]
    )
    for condition in conditions:
        values = [
            (
                "N/A"
                if (benchmark, condition) not in compatibility_lookup
                else _percent(compatibility_lookup[(benchmark, condition)].get("value"))
            )
            for benchmark in benchmarks
        ]
        lines.append(f"| {condition} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "Proof cells are normalized mean grades using `avg@runs`; answer cells use "
            "the configured compatibility seed.",
            "",
            "## Raw outcomes and cost",
            "",
            "| Benchmark | Condition | Done | Strict success | Mean grade | Generated tokens |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        done = f"{row['completed']}/{row['scheduled']}"
        lines.append(
            "| {benchmark} | {condition} | {done} | {success} | {grade} | {tokens:,} |".format(
                benchmark=BENCHMARK_LABELS.get(str(row["benchmark"]), row["benchmark"]),
                condition=row["condition"],
                done=done,
                success=_percent(row.get("raw_success_rate")),
                grade=_number(row.get("mean_grade")),
                tokens=int(row.get("generated_tokens", 0)),
            )
        )

    lines.extend(
        [
            "",
            "## Budget curves",
            "",
            "| Benchmark | Condition | k | pass@k | best grade@k | "
            "final generated cost@k | incurred generated cost@k |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in curves:
        lines.append(
            "| {benchmark} | {condition} | {k} | {passed} | {best} | {generated:,.0f} | "
            "{incurred:,.0f} |".format(
                benchmark=BENCHMARK_LABELS.get(str(row["benchmark"]), row["benchmark"]),
                condition=row["condition"],
                k=row["k"],
                passed=_percent(row.get("pass_at_k")),
                best=_number(row.get("best_grade_at_k")),
                generated=float(row.get("generated_cost_at_k", 0)),
                incurred=float(row.get("incurred_generated_cost_at_k", 0)),
            )
        )

    lines.extend(
        [
            "",
            "`best grade@k` is retrospective oracle coverage. Final cost uses successful "
            "or terminal attempts; incurred cost also includes invalidated retries (an upper "
            "bound when interrupted usage is unknown). Both exclude the external judge.",
            "",
            "## Paired bootstrap deltas",
            "",
            "| Benchmark | Comparison | Metric | Delta | 95% CI | n |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in bootstrap:
        interval = f"[{_number(row.get('ci95_low'))}, {_number(row.get('ci95_high'))}]"
        lines.append(
            f"| {BENCHMARK_LABELS.get(str(row['benchmark']), row['benchmark'])} | "
            f"{row['condition']} − {row['baseline']} | {row['metric']} | "
            f"{_number(row.get('mean_delta'))} | {interval} | {row['paired_n']} |"
        )

    costs = report.get("cost_accounting", {})
    if isinstance(costs, Mapping):
        lines.extend(
            [
                "",
                "## Incurred token accounting",
                "",
                "| Scope | Generated | Total | Exact |",
                "|---|---:|---:|---:|",
            ]
        )
        for scope in (
            "active_evaluation",
            "invalidated_interrupted",
            "external_judge",
        ):
            value = costs.get(scope, {})
            if not isinstance(value, Mapping):
                continue
            lines.append(
                f"| {scope} | {int(value.get('generated_tokens', 0)):,} | "
                f"{int(value.get('total_tokens', 0)):,} | "
                f"{'yes' if value.get('all_token_counts_exact') else 'no'} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_report(
    output_dir: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    comparisons: Sequence[tuple[str, str]] = DEFAULT_COMPARISONS,
    ks: Sequence[int] = (1, 2, 3),
    expected_seeds: Sequence[int] | None = None,
    answer_compatibility_seed: int = 0,
) -> dict[str, Any]:
    """Compute metrics and atomically write all report formats."""

    destination = Path(output_dir)
    materialized = list(rows)
    report = compute_metrics(
        materialized,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        comparisons=comparisons,
        ks=ks,
        expected_seeds=expected_seeds,
        answer_compatibility_seed=answer_compatibility_seed,
    )
    report["artifacts"] = {
        "json": "report.json",
        "markdown": "report.md",
        "summary_csv": "summary.csv",
        "curves_csv": "curves.csv",
        "compatibility_csv": "compatibility.csv",
        "bootstrap_csv": "bootstrap.csv",
    }
    _atomic_text(
        destination / "report.json",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(destination / "report.md", render_markdown(report))
    _atomic_text(destination / "summary.csv", _csv_text(report["summary"]))
    _atomic_text(destination / "curves.csv", _csv_text(report["curves"]))
    _atomic_text(
        destination / "compatibility.csv", _csv_text(report["compatibility"])
    )
    _atomic_text(
        destination / "bootstrap.csv", _csv_text(report["paired_bootstrap"])
    )
    return report


generate_report = write_report


__all__ = [
    "generate_report",
    "load_jsonl",
    "render_markdown",
    "write_report",
]
