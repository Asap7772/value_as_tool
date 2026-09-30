"""Problem-paired factorial comparisons for a frozen attempt bank."""

from __future__ import annotations

import csv
import io
import itertools
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .harnesses import resolve_harness
from .metrics import pass_at_k, strict_success
from .schemas import TokenUsage
from .storage import atomic_write_json, atomic_write_text


def factorial_comparisons(
    methods: Sequence[str], factors: Mapping[str, tuple[str, str, str, bool]]
) -> list[tuple[str, str, str]]:
    """Compare settings differing in exactly one of the four factors."""
    names = ("conditioning", "interaction", "feedback", "gold_reference")
    result = []
    for left, right in itertools.combinations(methods, 2):
        changed = [
            i for i, (a, b) in enumerate(zip(factors[left], factors[right], strict=True)) if a != b
        ]
        if len(changed) == 1:
            result.append((left, right, names[changed[0]]))
    return result


def paired_factorial_bootstrap(
    values: np.ndarray,
    methods: Sequence[str],
    comparisons: Sequence[tuple[str, str, str]],
    *,
    samples: int,
    seed: int,
    benchmarks: int = 2,
) -> list[dict[str, Any]]:
    """Simultaneous intervals across planned pairs/k, Bonferroni over benchmarks.

    ``values`` has axes problem, method, k. Problems, including all their seeds
    and methods, are the resampling unit. Uncertainty is conditional on the
    frozen attempt bank. Zero-variance contrasts are reported as degenerate.
    """
    if len(values) < 2 or not comparisons:
        return []
    indices = {method: i for i, method in enumerate(methods)}
    contrasts = np.stack(
        [values[:, indices[a], :] - values[:, indices[b], :] for a, b, _ in comparisons],
        axis=1,
    )
    n, pairs, ks = contrasts.shape
    flat = contrasts.reshape(n, pairs * ks)
    delta = flat.mean(axis=0)
    se = flat.std(axis=0, ddof=1) / np.sqrt(n)
    varying = se > 1e-12
    rng = np.random.default_rng(seed)
    boot = np.empty((samples, pairs * ks), dtype=np.float32)
    for start in range(0, samples, 256):
        count = min(256, samples - start)
        draws = rng.multinomial(n, np.full(n, 1.0 / n), size=count)
        boot[start : start + count] = draws @ flat / n
    low, high = np.quantile(boot, (0.025, 0.975), axis=0)
    if varying.any():
        maximum = np.max(np.abs((boot[:, varying] - delta[varying]) / se[varying]), axis=1)
        critical = float(np.quantile(maximum, 1 - 0.05 / benchmarks))
    else:
        critical = 0.0
    result = []
    for index, (left, right, factor) in enumerate(comparisons):
        for k in range(ks):
            cell = index * ks + k
            lower = max(-1.0, float(delta[cell] - critical * se[cell]))
            upper = min(1.0, float(delta[cell] + critical * se[cell]))
            result.append(
                {
                    "method": left,
                    "baseline": right,
                    "factor": factor,
                    "k": k + 1,
                    "problems": n,
                    "delta": float(delta[cell]),
                    "pointwise_ci_low": float(low[cell]),
                    "pointwise_ci_high": float(high[cell]),
                    "simultaneous_ci_low": lower,
                    "simultaneous_ci_high": upper,
                    "significant_corrected": lower > 0 or upper < 0,
                    "standard_error": float(se[cell]),
                    "critical_value": critical,
                    "bootstrap_samples": samples,
                }
            )
    return result


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    stream = io.StringIO()
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, stream.getvalue())


def _sum_usage(usages: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    total = TokenUsage()
    for usage in usages:
        total += TokenUsage.from_dict(usage)
    return total.to_dict()


def _amortized_usage(
    source: Mapping[str, Any], summary: Mapping[str, Any], arms: int, mode_arms: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cached_prompt_tokens",
    ):
        result[key] = (
            None
            if source.get(key) is None and summary.get(key) is None
            else (source.get(key) or 0) / arms + (summary.get(key) or 0) / mode_arms
        )
    return result


def _preprocessing_costs(
    records: Sequence[Mapping[str, Any]], mode_counts: Mapping[str, int]
) -> dict[str, Any]:
    """Account for preprocessing once, retaining unknown completion bounds."""
    source = _sum_usage(record["source_usage"] for record in records)
    source_judge = _sum_usage(record["source_judge_usage"] for record in records)
    arms = sum(mode_counts.values())
    no_summary = TokenUsage().to_dict()
    by_mode = {}
    for mode, mode_arms in mode_counts.items():
        final = _sum_usage(
            record["summary_usage"][mode] for record in records if mode != "solutions"
        )
        all_attempts = _sum_usage(
            record["summary_all_attempt_usage"][mode] for record in records if mode != "solutions"
        )
        unknown = {
            key: sum(
                int(record["summary_unknown_usage"][mode][key])
                for record in records
                if mode != "solutions"
            )
            for key in ("requests", "completion_tokens_upper_bound")
        }
        standalone = _sum_usage((source, all_attempts))
        amortized = _amortized_usage(source, all_attempts, arms, mode_arms)
        by_mode[mode] = {
            "method_count": mode_arms,
            "summary_usage": final,
            "summary_all_attempt_usage": all_attempts,
            "summary_unknown_usage": unknown,
            "standalone_usage": standalone,
            "amortized_usage": amortized,
            "standalone_judge_usage": source_judge,
            "amortized_judge_usage": _amortized_usage(source_judge, no_summary, arms, mode_arms),
            "standalone_unknown_usage": unknown,
            "amortized_unknown_usage": {key: value / mode_arms for key, value in unknown.items()},
            "usage_exact": unknown["requests"] == 0,
        }
    return {
        "problems": len(records),
        "method_count": arms,
        "source_usage": source,
        "source_judge_usage": source_judge,
        "summary_usage": _sum_usage(value["summary_usage"] for value in by_mode.values()),
        "summary_all_attempt_usage": _sum_usage(
            value["summary_all_attempt_usage"] for value in by_mode.values()
        ),
        "summary_unknown_usage": {
            key: sum(value["summary_unknown_usage"][key] for value in by_mode.values())
            for key in ("requests", "completion_tokens_upper_bound")
        },
        "by_mode": by_mode,
    }


def write_conditioning_report(
    context: Any, rows: Sequence[Mapping[str, Any]], destination: Path
) -> dict[str, Any]:
    from .conditioning import load_frozen_manifest

    config = context.config
    conditioning = config.conditioning
    assert conditioning is not None
    manifest = load_frozen_manifest(Path(conditioning.bank_root), conditioning.manifest_sha256)
    specs = [resolve_harness(value) for value in config.evaluation.harnesses]
    if not specs or any(spec.conditioning_mode is None for spec in specs):
        raise ValueError("conditioned report requires attempt-conditioned harnesses")
    factors = {
        spec.harness_id: (
            spec.conditioning_mode,
            "value_tool" if str(spec.condition).startswith("value_tool") else "gvr",
            "rationale" if "rationale" in str(spec.condition) else "legacy",
            spec.requires_reference,
        )
        for spec in specs
    }
    methods = list(factors)
    if len(methods) != len(specs):
        raise ValueError("conditioned report requires unique harness IDs")
    mode_counts = Counter(str(spec.conditioning_mode) for spec in specs)
    comparisons = factorial_comparisons(methods, factors)
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["benchmark"]), str(row["problem_id"]), str(row["harness_id"]))].append(row)
    curve_rows: list[dict[str, Any]] = []
    significance: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    strata_rows: list[dict[str, Any]] = []
    expected_seeds = set(config.evaluation.seeds)
    benchmarks = sorted(config.evaluation.benchmarks)
    problems_by_benchmark = {
        benchmark: sorted(manifest["problems"][benchmark]) for benchmark in benchmarks
    }
    expected_groups = {
        (benchmark, problem, method)
        for benchmark, problems in problems_by_benchmark.items()
        for problem in problems
        for method in methods
    }
    if (
        not expected_groups
        or set(grouped) != expected_groups
        or any(
            {row["seed"] for row in selected} != expected_seeds
            or len(selected) != len(expected_seeds)
            for selected in grouped.values()
        )
    ):
        raise ValueError("conditioned report requires exactly the scheduled sample grid")
    # Decide this before any benchmark is analyzed: missing cells anywhere
    # suppress significance everywhere. Finalized failures remain scheduled zeroes.
    incomplete_statuses = {None, "unjudged", "missing", "pending", "running"}
    incomplete = sum(
        row.get("judge_status") in incomplete_statuses
        or row.get("solve_status") in incomplete_statuses
        for row in rows
    )
    complete = incomplete == 0
    all_records = [
        manifest["problems"][benchmark][problem]
        for benchmark, problems in problems_by_benchmark.items()
        for problem in problems
    ]
    preprocessing = {
        "bank_sha256": conditioning.bank_sha256,
        "manifest_sha256": conditioning.manifest_sha256,
        **_preprocessing_costs(all_records, mode_counts),
        "by_benchmark": {},
        "historical_bank_reused": True,
        "scope": "scheduled problems and selected conditioning modes",
        "usage_currency": (
            "Unqualified usage and generated-token curve fields count Qwen solver/summarizer "
            "tokens. Judge tokens use separate judge_usage and judge_generated_tokens fields."
        ),
        "bank_cost_policy": (
            "Standalone cost includes historical solver attempts once per problem, with their "
            "judge cost reported separately. Amortized costs divide both source components "
            "across all selected arms."
        ),
        "summary_cost_policy": (
            "Standalone cost includes all persisted summary attempts once per problem and mode. "
            "Amortized cost divides it across selected arms using that mode. Unknown completion "
            "usage is reported separately as an upper bound, never as exact usage."
        ),
    }
    for benchmark_index, benchmark in enumerate(benchmarks):
        problems = problems_by_benchmark[benchmark]
        costs = _preprocessing_costs(
            [manifest["problems"][benchmark][problem] for problem in problems], mode_counts
        )
        preprocessing["by_benchmark"][benchmark] = costs
        values = np.zeros((len(problems), len(methods), len(expected_seeds)))
        strata = []
        for pi, problem in enumerate(problems):
            record = manifest["problems"][benchmark][problem]
            positive = int(record["successes"])
            negative = int(record["failures"])
            strata.append(
                "mixed"
                if positive and negative
                else "positive_only"
                if positive
                else "negative_only"
            )
            for mi, method in enumerate(methods):
                selected = grouped.get((benchmark, problem, method), [])
                correct = sum(strict_success(row) for row in selected)
                values[pi, mi, :] = [
                    pass_at_k(len(selected), correct, k) for k in range(1, len(selected) + 1)
                ]
        for mi, method in enumerate(methods):
            mode, interaction, feedback, gold = factors[method]
            selected = [
                row for problem in problems for row in grouped[(benchmark, problem, method)]
            ]
            tokens = {
                "benchmark": benchmark,
                "method": method,
                "conditioning_mode": mode,
                "interaction": interaction,
                "feedback": feedback,
                "gold_reference": gold,
                "trajectories": len(selected),
                "problems": len(problems),
            }
            for key in ("generated_tokens", "reasoning_tokens", "prompt_tokens"):
                tokens[key] = sum(int(row.get(key) or 0) for row in selected)
                tokens[f"mean_{key}"] = tokens[key] / len(selected)
            tokens["length_calls"] = sum(int(row.get("length_call_count") or 0) for row in selected)
            tokens["trajectories_with_length"] = sum(
                bool(row.get("length_call_count")) for row in selected
            )
            for status in ("budget_exhausted", "context_exhausted", "protocol_error", "failed"):
                tokens[status] = sum(row.get("solve_status") == status for row in selected)
            tokens["judge_parse_errors"] = sum(
                row.get("judge_status") == "parse_error" for row in selected
            )
            tokens["judge_generated_tokens"] = sum(
                int((row.get("judge_usage") or {}).get("completion_tokens") or 0)
                for row in selected
            )
            mode_cost = costs["by_mode"][mode]
            for policy in ("standalone", "amortized"):
                usage = mode_cost[f"{policy}_usage"]
                judge_usage = mode_cost[f"{policy}_judge_usage"]
                unknown = mode_cost[f"{policy}_unknown_usage"]
                tokens[f"{policy}_preprocessing_generated_tokens"] = usage["completion_tokens"]
                tokens[f"{policy}_preprocessing_prompt_tokens"] = usage["prompt_tokens"]
                tokens[f"{policy}_preprocessing_judge_generated_tokens"] = judge_usage[
                    "completion_tokens"
                ]
                tokens[f"{policy}_preprocessing_judge_prompt_tokens"] = judge_usage["prompt_tokens"]
                tokens[f"{policy}_preprocessing_unknown_requests"] = unknown["requests"]
                tokens[f"{policy}_preprocessing_unknown_completion_tokens_upper_bound"] = unknown[
                    "completion_tokens_upper_bound"
                ]
            tokens["preprocessing_usage_exact"] = mode_cost["usage_exact"]
            token_rows.append(tokens)
            for k, rate in enumerate(values[:, mi, :].mean(axis=0), 1):
                curve = {
                    **{
                        key: tokens[key]
                        for key in (
                            "benchmark",
                            "method",
                            "conditioning_mode",
                            "interaction",
                            "feedback",
                            "gold_reference",
                            "problems",
                        )
                    },
                    "k": k,
                    "pass_at_k": float(rate),
                    "downstream_expected_generated_tokens": k * tokens["mean_generated_tokens"],
                    "downstream_expected_judge_generated_tokens": (
                        k * tokens["judge_generated_tokens"] / len(selected)
                    ),
                }
                for policy in ("standalone", "amortized"):
                    # Curves express expected cost per problem at k attempts;
                    # preprocessing is paid once, independently of k.
                    curve[f"{policy}_preprocessing_generated_tokens"] = tokens[
                        f"{policy}_preprocessing_generated_tokens"
                    ] / len(problems)
                    curve[f"{policy}_expected_generated_tokens"] = (
                        curve[f"{policy}_preprocessing_generated_tokens"]
                        + curve["downstream_expected_generated_tokens"]
                    )
                    curve[f"{policy}_preprocessing_judge_generated_tokens"] = tokens[
                        f"{policy}_preprocessing_judge_generated_tokens"
                    ] / len(problems)
                    curve[f"{policy}_expected_judge_generated_tokens"] = (
                        curve[f"{policy}_preprocessing_judge_generated_tokens"]
                        + curve["downstream_expected_judge_generated_tokens"]
                    )
                    curve[f"{policy}_unknown_completion_tokens_upper_bound"] = tokens[
                        f"{policy}_preprocessing_unknown_completion_tokens_upper_bound"
                    ] / len(problems)
                curve["preprocessing_usage_exact"] = mode_cost["usage_exact"]
                curve_rows.append(curve)
                for stratum in sorted(set(strata)):
                    mask = np.array([value == stratum for value in strata])
                    strata_rows.append(
                        {
                            "benchmark": benchmark,
                            "method": method,
                            "bank_composition": stratum,
                            "problems": int(mask.sum()),
                            "k": k,
                            "pass_at_k": float(values[mask, mi, k - 1].mean()),
                        }
                    )
        if complete:
            significance.extend(
                {"benchmark": benchmark, **row}
                for row in paired_factorial_bootstrap(
                    values,
                    methods,
                    comparisons,
                    samples=config.evaluation.bootstrap_samples,
                    seed=1729 + benchmark_index,
                    benchmarks=max(1, len(benchmarks)),
                )
            )
    _csv(destination / "conditioning-pass-at-k.csv", curve_rows)
    _csv(destination / "conditioning-tokens.csv", token_rows)
    _csv(destination / "conditioning-strata.csv", strata_rows)
    _csv(destination / "conditioning-significance.csv", significance)
    atomic_write_json(destination / "conditioning-preprocessing.json", preprocessing)
    plot_data = {"curves": curve_rows, "tokens": token_rows}
    atomic_write_json(destination / "conditioning-plot-data.json", plot_data)
    methodology = {
        "complete": complete,
        "incomplete_cells": incomplete,
        "comparisons_per_benchmark_per_k": len(comparisons),
        "bootstrap_unit": "problem; all methods and seeds remain paired",
        "uncertainty": "conditional on the frozen attempt bank",
        "correction": (
            "max absolute observed-SE-standardized centered bootstrap across planned contrasts "
            "and k; Bonferroni over benchmarks"
        ),
        "files": [
            "conditioning-pass-at-k.csv",
            "conditioning-tokens.csv",
            "conditioning-strata.csv",
            "conditioning-significance.csv",
            "conditioning-preprocessing.json",
            "conditioning-plot-data.json",
        ],
    }
    atomic_write_json(destination / "conditioning-analysis.json", methodology)
    return methodology
