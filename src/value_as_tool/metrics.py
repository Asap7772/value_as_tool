"""Budget-aware success, pass@k, continuous-reward, and paired metrics."""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import Any

import numpy as np

from .schemas import ADJUDICATABLE_TRAJECTORY_STATUSES

DEFAULT_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("gvr", "direct"),
    ("gvr_subagents", "direct"),
    ("gvr_reference", "direct"),
    ("value_tool", "direct"),
    ("gvr_rationale_score", "direct"),
    ("value_tool_rationale_score", "direct"),
    ("gvr_reference_rationale_score", "direct"),
    ("cch_plan_work_review", "direct"),
    ("gvr_rationale_score", "gvr"),
    ("value_tool_rationale_score", "value_tool"),
    ("gvr_reference_rationale_score", "gvr_reference"),
)
PROOF_BENCHMARKS = {
    "imo_proof",
    "imo-proofbench",
    "imoproofbench",
    "proofbench",
    "lm-provers/imoproofbench",
    "lm-provers/proofbench",
}
LEGACY_REFERENCE_METHODS = frozenset(
    {"gvr_reference", "gvr_reference_rationale_score"}
)
HARNESS_ACCESS_CLASSES = frozenset({
    "blind", "reference_assisted", "attempt_assisted", "attempt_and_reference_assisted"
})


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def _nested_mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key)
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return value if isinstance(value, Mapping) else {}


def _field(row: Mapping[str, Any], name: str, default: Any = None) -> Any:
    if name in row:
        return _plain(row[name])
    request = _nested_mapping(row, "request")
    if name in request:
        return _plain(request[name])
    if name == "item_id" and "problem_id" in request:
        return _plain(request["problem_id"])
    if name == "item_id" and "problem_id" in row:
        return _plain(row["problem_id"])
    return default


def _identity_field(row: Mapping[str, Any], name: str, default: Any = None) -> Any:
    """Read one consistent harness-identity value from all persisted locations.

    Final pipeline rows put harness identity at the top level, while raw
    trajectory artifacts retain part of it on the request and the rest in
    request metadata.  Accept both shapes, but never let a stale top-level
    value silently mask conflicting source-version metadata.
    """

    candidates: list[tuple[str, Any]] = []
    if name in row and row[name] is not None:
        candidates.append(("row", _plain(row[name])))
    request = _nested_mapping(row, "request")
    if name in request and request[name] is not None:
        candidates.append(("request", _plain(request[name])))
    metadata = _nested_mapping(request, "metadata")
    if name in metadata and metadata[name] is not None:
        candidates.append(("request.metadata", _plain(metadata[name])))
    if not candidates:
        return default
    value = candidates[0][1]
    conflicts = [location for location, candidate in candidates[1:] if candidate != value]
    if conflicts:
        locations = ", ".join([candidates[0][0], *conflicts])
        raise ValueError(f"inconsistent {name} across {locations}")
    return value


def _method_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return one source-versioned method identity with a legacy fallback."""

    harness_id_value = _identity_field(row, "harness_id")
    condition_value = _field(row, "condition")
    if harness_id_value is None:
        if condition_value is None or not str(condition_value):
            raise ValueError("each metric row requires a harness_id or condition")
        method_id = str(condition_value)
        identity = {
            "method_id": method_id,
            "condition": method_id,
            "harness_id": None,
            "harness_entrypoint": None,
            "harness_source_sha256": None,
            "harness_access": (
                "reference_assisted"
                if method_id in LEGACY_REFERENCE_METHODS
                else "blind"
            ),
        }
        _validate_reported_method_id(row, method_id)
        return identity

    method_id = str(harness_id_value)
    source_value = _identity_field(row, "harness_source_sha256")
    access_value = _identity_field(row, "harness_access")
    entrypoint_value = _identity_field(row, "harness_entrypoint")
    if not method_id:
        raise ValueError("harness_id must not be empty")
    source_sha256 = str(source_value or "")
    if len(source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_sha256
    ):
        raise ValueError("harness_source_sha256 must be a lowercase SHA-256 digest")
    access = str(access_value or "")
    if access not in HARNESS_ACCESS_CLASSES:
        raise ValueError("unsupported harness_access class")
    entrypoint = str(entrypoint_value or "")
    if not entrypoint:
        raise ValueError("harness_entrypoint must not be empty")
    identity = {
        "method_id": method_id,
        # Keep the historical field populated so downstream CSV consumers do
        # not need a flag day.  It is now a method ID, not necessarily a
        # schemas.Condition value.
        "condition": method_id,
        "harness_id": method_id,
        "harness_entrypoint": entrypoint,
        "harness_source_sha256": source_sha256,
        "harness_access": access,
    }
    _validate_reported_method_id(row, method_id)
    return identity


def _validate_reported_method_id(row: Mapping[str, Any], resolved: str) -> None:
    """Reject a denormalized method label that disagrees with its identity."""

    reported = row.get("method_id")
    if reported is not None and str(_plain(reported)) != resolved:
        raise ValueError(
            f"method_id {reported!r} does not match resolved method identity {resolved!r}"
        )


def _method_key(row: Mapping[str, Any]) -> tuple[str, str]:
    identity = _method_identity(row)
    return (
        str(identity["method_id"]),
        str(identity["harness_source_sha256"] or ""),
    )


def _method_output_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    identity = _method_identity(row)
    return {
        key: identity[key]
        for key in (
            "method_id",
            "condition",
            "harness_id",
            "harness_entrypoint",
            "harness_source_sha256",
            "harness_access",
        )
    }


def _is_proof(benchmark: str) -> bool:
    return benchmark.casefold() in PROOF_BENCHMARKS


def _status(row: Mapping[str, Any]) -> str:
    value = row.get("judge_status")
    if value is None:
        value = row.get("status", "missing")
    return str(_plain(value) or "missing").casefold()


def _is_applicable(row: Mapping[str, Any]) -> bool:
    return row.get("applicable", True) is not False and _status(row) not in {
        "n/a",
        "na",
        "not_applicable",
    }


def _is_adjudicated(row: Mapping[str, Any]) -> bool:
    solve_status = row.get("solve_status")
    if solve_status is not None and str(_plain(solve_status)).casefold() not in (
        ADJUDICATABLE_TRAJECTORY_STATUSES
    ):
        return False
    judge_status = row.get("judge_status")
    if judge_status is not None:
        status = str(_plain(judge_status)).casefold()
        if status == "empty_answer":
            return True
        if status != "completed":
            return False
        return (
            row.get("score") is not None
            or row.get("grade") is not None
            or row.get("correct") is not None
            or row.get("is_correct") is not None
        )
    status = _status(row)
    if status in {
        "api_error",
        "context_error",
        "failed",
        "invalid_usage",
        "missing",
        "parse_error",
        "protocol_error",
        "unjudged",
    }:
        return False
    return any(
        row.get(key) is not None for key in ("score", "grade", "correct", "is_correct")
    )


def _score(row: Mapping[str, Any], *, proof: bool) -> float:
    """Return reward, treating every unjudged scheduled cell as zero."""

    if not _is_adjudicated(row):
        return 0.0
    value: Any
    if proof:
        value = row.get("score", row.get("grade", 0))
        if isinstance(value, Mapping):
            value = value.get("points", 0)
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(score) or not 0 <= score <= 7:
            raise ValueError(f"proof score must be between 0 and 7, found {value!r}")
        return score
    value = row.get("correct", row.get("is_correct", row.get("score", 0)))
    if isinstance(value, str):
        return float(value.casefold() in {"1", "true", "correct", "yes"})
    return float(bool(value))


def strict_success(row: Mapping[str, Any]) -> bool:
    """Full-credit proof or correct final answer; partial proof credit is false."""

    proof = _is_proof(str(_field(row, "benchmark", "")))
    reward = _score(row, proof=proof)
    return reward == (7.0 if proof else 1.0)


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError("token counts must be non-negative")
    return parsed


def _token_cost(row: Mapping[str, Any]) -> tuple[int, int, bool]:
    usage = _nested_mapping(row, "usage")
    online = _nested_mapping(row, "online_cost")
    generated = None
    for source, keys in (
        (row, ("generated_tokens", "completion_tokens", "output_tokens")),
        (usage, ("generated_tokens", "completion_tokens", "output_tokens")),
        (online, ("generated_tokens", "completion_tokens", "output_tokens")),
    ):
        for key in keys:
            if key in source and source[key] is not None:
                generated = _nonnegative_int(source[key])
                break
        if generated is not None:
            break
    generated = generated or 0

    total = None
    for source in (row, usage, online):
        if source.get("total_tokens") is not None:
            total = _nonnegative_int(source["total_tokens"])
            break
    if total is None:
        prompt = None
        for source, keys in (
            (row, ("input_tokens", "prompt_tokens")),
            (usage, ("input_tokens", "prompt_tokens")),
            (online, ("input_tokens", "prompt_tokens")),
        ):
            for key in keys:
                if source.get(key) is not None:
                    prompt = _nonnegative_int(source[key])
                    break
            if prompt is not None:
                break
        total = (prompt or 0) + generated
    explicit_exact = row.get("usage_exact", usage.get("usage_exact"))
    exact = bool(explicit_exact) if explicit_exact is not None else bool(total or generated)
    return generated, total, exact


def canonical_tokens(cost: Mapping[str, Any] | int | None) -> int:
    """Return one token total without double-counting diagnostic subsets."""

    if cost is None:
        return 0
    if isinstance(cost, bool):
        raise ValueError("token count cannot be boolean")
    if isinstance(cost, int):
        if cost < 0:
            raise ValueError("token count must be non-negative")
        return cost
    total = _nonnegative_int(cost.get("total_tokens"))
    if total is not None:
        return total
    prompt = _nonnegative_int(cost.get("prompt_tokens", cost.get("input_tokens"))) or 0
    completion = (
        _nonnegative_int(cost.get("completion_tokens", cost.get("output_tokens"))) or 0
    )
    return prompt + completion


def _usage_token_pair(value: Any) -> tuple[int, int, bool]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        return 0, 0, False
    generated = _nonnegative_int(
        value.get("generated_tokens", value.get("completion_tokens", value.get("output_tokens")))
    )
    total = canonical_tokens(value)
    exact_value = value.get("usage_exact")
    exact = bool(exact_value) if exact_value is not None else generated is not None
    return generated or 0, total, exact


def _incurred_token_cost(row: Mapping[str, Any]) -> tuple[int, int, bool]:
    """Final solver attempt plus any earlier invalidated attempt spend."""

    generated, total, exact = _token_cost(row)
    invalidated = (
        row.get("invalidated_solve_usage")
        if "invalidated_solve_usage" in row
        else row.get("invalidated_usage", row.get("invalidated_cost"))
    )
    if invalidated is None:
        return generated, total, exact
    invalid_generated, invalid_total, invalid_exact = _usage_token_pair(invalidated)
    return generated + invalid_generated, total + invalid_total, exact and invalid_exact


def summarize_cost_accounting(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Separate solver, interrupted/invalidated, and external-judge spend."""

    buckets: dict[str, dict[str, Any]] = {
        name: {"generated_tokens": 0, "total_tokens": 0, "records": 0, "exact": True}
        for name in ("active_evaluation", "invalidated_interrupted", "external_judge")
    }
    interrupted_statuses = {
        "interrupted",
        "invalid_usage",
        "cancelled",
        "canceled",
    }

    def add(bucket: str, generated: int, total: int, exact: bool) -> None:
        target = buckets[bucket]
        target["generated_tokens"] += generated
        target["total_tokens"] += total
        target["records"] += 1
        target["exact"] = target["exact"] and exact

    for row in rows:
        generated, total, exact = _token_cost(row)
        solve_status = str(
            _plain(row.get("solve_status", row.get("status", ""))) or ""
        ).casefold()
        bucket = (
            "invalidated_interrupted"
            if row.get("invalidated") is True or solve_status in interrupted_statuses
            else "active_evaluation"
        )
        add(bucket, generated, total, exact)

        invalidated = row.get("invalidated_usage", row.get("invalidated_cost"))
        if invalidated is not None:
            add("invalidated_interrupted", *_usage_token_pair(invalidated))

        judge = row.get("judge_usage", row.get("judge_cost"))
        if judge is None:
            judge_result = _nested_mapping(row, "judge_result")
            judge = judge_result.get("usage")
        if judge is not None:
            add("external_judge", *_usage_token_pair(judge))

    for value in buckets.values():
        value["all_token_counts_exact"] = value.pop("exact")
    return buckets


def pass_at_k(n: int, correct: int, k: int) -> float:
    """Unbiased pass@k for ``correct`` successes among ``n`` samples."""

    if n < 0 or correct < 0 or correct > n:
        raise ValueError("pass@k requires 0 <= correct <= n")
    if k <= 0 or k > n:
        raise ValueError("pass@k requires 1 <= k <= n")
    if correct == 0:
        return 0.0
    if n - correct < k:
        return 1.0
    return 1.0 - math.comb(n - correct, k) / math.comb(n, k)


def expected_best_at_k(values: Sequence[float], k: int) -> float:
    """Exact expected maximum over uniformly selected size-k subsets."""

    if k <= 0 or k > len(values):
        raise ValueError("best@k requires 1 <= k <= number of samples")
    maxima = (max(combo) for combo in itertools.combinations(values, k))
    return float(np.mean(list(maxima)))


def best_grade_at_k(grades: Sequence[float], k: int) -> float:
    return expected_best_at_k(grades, k)


def expected_cost_at_k(costs: Sequence[float], k: int) -> float:
    """Expected total cost of a uniformly selected size-k subset."""

    if k <= 0 or k > len(costs):
        raise ValueError("cost@k requires 1 <= k <= number of samples")
    totals = (sum(combo) for combo in itertools.combinations(costs, k))
    return float(np.mean(list(totals)))


def _normalize_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Reject duplicates and materialize inferable missing scheduled cells."""

    supplied: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    by_benchmark_items: dict[str, set[str]] = defaultdict(set)
    by_benchmark_seeds: dict[str, set[int]] = defaultdict(set)
    by_benchmark_item_methods: dict[
        str, dict[str, set[tuple[str, str]]]
    ] = defaultdict(lambda: defaultdict(set))
    method_identities: dict[tuple[str, str], dict[str, Any]] = {}
    for original in rows:
        if not _is_applicable(original):
            continue
        row = dict(original)
        benchmark = str(_field(row, "benchmark", ""))
        item_id = str(_field(row, "item_id", ""))
        method_identity = _method_identity(row)
        method_id = str(method_identity["method_id"])
        source_sha256 = str(method_identity["harness_source_sha256"] or "")
        method_key = method_id, source_sha256
        seed_value = _field(row, "seed", 0)
        if not benchmark or not item_id:
            raise ValueError("each metric row requires benchmark and item_id")
        if isinstance(seed_value, bool):
            raise ValueError("seed must be an integer")
        try:
            seed = int(seed_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("seed must be an integer") from exc
        key = benchmark, item_id, method_id, source_sha256, seed
        if key in supplied:
            raise ValueError(f"duplicate scheduled cell: {key}")
        previous_identity = method_identities.get(method_key)
        if previous_identity is not None and previous_identity != method_identity:
            raise ValueError(
                f"inconsistent harness metadata for method {method_id!r} at "
                f"source {source_sha256 or 'legacy'}"
            )
        method_identities[method_key] = method_identity
        row.update(
            benchmark=benchmark,
            item_id=item_id,
            seed=seed,
            **method_identity,
        )
        supplied[key] = row
        by_benchmark_items[benchmark].add(item_id)
        by_benchmark_seeds[benchmark].add(seed)
        by_benchmark_item_methods[benchmark][item_id].add(method_key)

    output: list[dict[str, Any]] = []
    for benchmark in sorted(by_benchmark_items):
        if expected_seeds is not None:
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in expected_seeds
            ):
                raise ValueError("expected_seeds must contain only integers")
            seeds = tuple(expected_seeds)
            if len(seeds) != len(set(seeds)):
                raise ValueError("expected_seeds must be unique")
        else:
            seeds = tuple(sorted(by_benchmark_seeds[benchmark]))
        if not seeds:
            raise ValueError("expected_seeds must not be empty")
        for item_id in sorted(by_benchmark_items[benchmark]):
            # Only fill missing seeds for a method/problem pair that is known
            # to be scheduled.  A reference-assisted harness may legitimately
            # be absent for a problem with no reference proof.
            for method_key in sorted(by_benchmark_item_methods[benchmark][item_id]):
                method_id, source_sha256 = method_key
                method_identity = method_identities[method_key]
                for seed in seeds:
                    key = benchmark, item_id, method_id, source_sha256, seed
                    output.append(
                        supplied.get(
                            key,
                            {
                                "benchmark": benchmark,
                                "item_id": item_id,
                                "seed": seed,
                                "judge_status": "missing",
                                "_inferred_missing": True,
                                **method_identity,
                            },
                        )
                    )
    return output


def _group_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        method_id, source_sha256 = _method_key(row)
        grouped[(str(row["benchmark"]), method_id, source_sha256)].append(row)
    return grouped


def summarize_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    normalized = _normalize_rows(rows, expected_seeds=expected_seeds)
    summaries: list[dict[str, Any]] = []
    for (benchmark, _method_id, _source_sha256), group in sorted(
        _group_rows(normalized).items()
    ):
        proof = _is_proof(benchmark)
        rewards = [_score(row, proof=proof) for row in group]
        successes = [reward == (7 if proof else 1) for reward in rewards]
        generated = [_token_cost(row)[0] for row in group]
        totals = [_token_cost(row)[1] for row in group]
        exact = [_token_cost(row)[2] for row in group]
        value_queries = [int(row.get("value_query_count", 0) or 0) for row in group]
        value_verifier_tokens = [
            int(
                _nested_mapping(row, "generated_tokens_by_role").get(
                    "value_verifier", 0
                )
                or 0
            )
            for row in group
        ]
        by_seed: dict[int, list[float]] = defaultdict(list)
        for row, reward in zip(group, rewards, strict=True):
            by_seed[int(row["seed"])].append(reward)
        seed_means = [float(np.mean(values)) for _, values in sorted(by_seed.items())]
        completed = sum(_is_adjudicated(row) for row in group)
        statuses = Counter(_status(row) for row in group)
        solve_statuses = Counter(
            str(_plain(row.get("solve_status", "missing")) or "missing").casefold()
            for row in group
        )
        summary: dict[str, Any] = {
            "benchmark": benchmark,
            **_method_output_fields(group[0]),
            "kind": "proof" if proof else "answer",
            "problems": len({str(row["item_id"]) for row in group}),
            "evaluation_runs": len(by_seed),
            "scheduled": len(group),
            "completed": completed,
            "completion_rate": completed / len(group),
            "missing_or_failed": len(group) - completed,
            "complete": completed == len(group),
            "status_counts": dict(sorted(statuses.items())),
            "solve_status_counts": dict(sorted(solve_statuses.items())),
            "judge_status_counts": dict(sorted(statuses.items())),
            "raw_success_rate": float(np.mean(successes)) if successes else None,
            "generated_tokens": int(sum(generated)),
            "total_tokens": int(sum(totals)),
            "value_queries": int(sum(value_queries)),
            "value_queries_per_trajectory": float(np.mean(value_queries)),
            "value_verifier_generated_tokens": int(sum(value_verifier_tokens)),
            "generated_tokens_per_trajectory": (
                float(np.mean(generated)) if generated else None
            ),
            "total_tokens_per_trajectory": float(np.mean(totals)) if totals else None,
            "all_token_counts_exact": all(exact),
        }
        if proof:
            mean_grade = float(np.mean(seed_means)) if seed_means else None
            summary.update(
                mean_grade=mean_grade,
                normalized_average_grade=(mean_grade / 7 if mean_grade is not None else None),
                normalized_average_grade_percent=(
                    mean_grade * 100 / 7 if mean_grade is not None else None
                ),
                legacy_aggregation=f"avg@{len(seed_means)}",
                answer_accuracy=None,
            )
        else:
            accuracy = float(np.mean(seed_means)) if seed_means else None
            summary.update(
                mean_grade=None,
                normalized_average_grade=None,
                normalized_average_grade_percent=None,
                legacy_aggregation=f"avg@{len(seed_means)}",
                answer_accuracy=accuracy,
            )
        summaries.append(summary)
    return summaries


def _per_problem(
    group: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in group:
        result[str(row["item_id"])].append(row)
    for values in result.values():
        values.sort(key=lambda row: int(row["seed"]))
    return result


def compute_curves(
    rows: Iterable[Mapping[str, Any]],
    *,
    ks: Sequence[int] = (1, 2, 3),
    expected_seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    normalized = _normalize_rows(rows, expected_seeds=expected_seeds)
    curves: list[dict[str, Any]] = []
    for (benchmark, _method_id, _source_sha256), group in sorted(
        _group_rows(normalized).items()
    ):
        proof = _is_proof(benchmark)
        problems = _per_problem(group)
        for k in ks:
            if k <= 0:
                raise ValueError("k values must be positive")
            pass_values: list[float] = []
            best_values: list[float] = []
            generated_costs: list[float] = []
            total_costs: list[float] = []
            incurred_generated_costs: list[float] = []
            incurred_total_costs: list[float] = []
            exact: list[bool] = []
            incurred_exact: list[bool] = []
            for problem_rows in problems.values():
                if k > len(problem_rows):
                    continue
                rewards = [_score(row, proof=proof) for row in problem_rows]
                successes = sum(reward == (7 if proof else 1) for reward in rewards)
                generated = [_token_cost(row)[0] for row in problem_rows]
                totals = [_token_cost(row)[1] for row in problem_rows]
                incurred_generated = [
                    _incurred_token_cost(row)[0] for row in problem_rows
                ]
                incurred_totals = [_incurred_token_cost(row)[1] for row in problem_rows]
                pass_values.append(pass_at_k(len(rewards), successes, k))
                best_values.append(expected_best_at_k(rewards, k))
                generated_costs.append(expected_cost_at_k(generated, k))
                total_costs.append(expected_cost_at_k(totals, k))
                incurred_generated_costs.append(
                    expected_cost_at_k(incurred_generated, k)
                )
                incurred_total_costs.append(expected_cost_at_k(incurred_totals, k))
                exact.extend(_token_cost(row)[2] for row in problem_rows)
                incurred_exact.extend(
                    _incurred_token_cost(row)[2] for row in problem_rows
                )
            if not pass_values:
                continue
            curves.append(
                {
                    "benchmark": benchmark,
                    **_method_output_fields(group[0]),
                    "kind": "proof" if proof else "answer",
                    "k": int(k),
                    "problems": len(pass_values),
                    "pass_at_k": float(np.mean(pass_values)),
                    "best_grade_at_k": (
                        float(np.mean(best_values)) if proof else None
                    ),
                    "best_normalized_reward_at_k": float(
                        np.mean(best_values) / (7 if proof else 1)
                    ),
                    "generated_cost_at_k": float(np.mean(generated_costs)),
                    "total_cost_at_k": float(np.mean(total_costs)),
                    "final_attempt_generated_cost_at_k": float(
                        np.mean(generated_costs)
                    ),
                    "final_attempt_total_cost_at_k": float(np.mean(total_costs)),
                    "incurred_generated_cost_at_k": float(
                        np.mean(incurred_generated_costs)
                    ),
                    "incurred_total_cost_at_k": float(np.mean(incurred_total_costs)),
                    "all_token_counts_exact": all(exact),
                    "all_incurred_token_counts_exact": all(incurred_exact),
                }
            )
    return curves


def compatibility_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    answer_seed: int = 0,
    expected_seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Reproduce QED's proof avg@runs and answer single-seed headline."""

    normalized = _normalize_rows(rows, expected_seeds=expected_seeds)
    output: list[dict[str, Any]] = []
    for (benchmark, _method_id, _source_sha256), group in sorted(
        _group_rows(normalized).items()
    ):
        proof = _is_proof(benchmark)
        if proof:
            seed_values: dict[int, list[float]] = defaultdict(list)
            for row in group:
                seed_values[int(row["seed"])].append(_score(row, proof=True))
            means = [float(np.mean(values)) for _, values in sorted(seed_values.items())]
            output.append(
                {
                    "benchmark": benchmark,
                    **_method_output_fields(group[0]),
                    "metric": f"normalized_avg@{len(means)}",
                    "value": float(np.mean(means) / 7) if means else None,
                    "percent": float(np.mean(means) * 100 / 7) if means else None,
                }
            )
        else:
            selected = [row for row in group if int(row["seed"]) == answer_seed]
            values = [_score(row, proof=False) for row in selected]
            output.append(
                {
                    "benchmark": benchmark,
                    **_method_output_fields(group[0]),
                    "metric": f"accuracy_seed_{answer_seed}",
                    "value": float(np.mean(values)) if values else None,
                    "percent": float(np.mean(values) * 100) if values else None,
                }
            )
    return output


def _problem_metric_values(
    problem_rows: Sequence[Mapping[str, Any]],
    *,
    proof: bool,
    ks: Sequence[int],
) -> dict[str, float]:
    rewards = [_score(row, proof=proof) for row in problem_rows]
    successes = [reward == (7 if proof else 1) for reward in rewards]
    generated = [_token_cost(row)[0] for row in problem_rows]
    totals = [_token_cost(row)[1] for row in problem_rows]
    incurred_generated = [_incurred_token_cost(row)[0] for row in problem_rows]
    incurred_totals = [_incurred_token_cost(row)[1] for row in problem_rows]
    values = {
        "raw_success_rate": float(np.mean(successes)),
        "mean_grade" if proof else "answer_accuracy": float(np.mean(rewards)),
    }
    if proof:
        values["normalized_average_grade"] = float(np.mean(rewards) / 7)
    for k in ks:
        if k > len(rewards):
            continue
        values[f"pass@{k}"] = pass_at_k(len(rewards), sum(successes), k)
        values[f"best_grade@{k}" if proof else f"best_reward@{k}"] = (
            expected_best_at_k(rewards, k)
        )
        values[f"generated_cost@{k}"] = expected_cost_at_k(generated, k)
        values[f"total_cost@{k}"] = expected_cost_at_k(totals, k)
        values[f"incurred_generated_cost@{k}"] = expected_cost_at_k(
            incurred_generated, k
        )
        values[f"incurred_total_cost@{k}"] = expected_cost_at_k(incurred_totals, k)
    return values


def paired_bootstrap(
    rows: Iterable[Mapping[str, Any]],
    *,
    comparisons: Sequence[tuple[str, str]] = DEFAULT_COMPARISONS,
    samples: int = 10_000,
    seed: int = 0,
    ks: Sequence[int] = (1, 2, 3),
    expected_seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Paired problem bootstrap after collapsing repeated seeds per problem."""

    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    normalized = _normalize_rows(rows, expected_seeds=expected_seeds)
    by_benchmark: dict[
        str,
        dict[tuple[str, str], dict[str, list[Mapping[str, Any]]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in normalized:
        by_benchmark[str(row["benchmark"])][_method_key(row)][str(row["item_id"])].append(
            row
        )

    output: list[dict[str, Any]] = []
    for benchmark, methods in sorted(by_benchmark.items()):
        proof = _is_proof(benchmark)
        for treatment, baseline in comparisons:
            treatment_keys = sorted(
                key for key in methods if key[0] == treatment
            )
            baseline_keys = sorted(key for key in methods if key[0] == baseline)
            if not treatment_keys or not baseline_keys:
                continue  # An absent condition is N/A, not a zero-valued arm.
            # Preserve source isolation.  If a caller deliberately combines
            # multiple source revisions, report every explicit version pair
            # instead of silently pooling them under a shared harness ID.
            for treatment_key, baseline_key in itertools.product(
                treatment_keys, baseline_keys
            ):
                treatment_groups = methods[treatment_key]
                baseline_groups = methods[baseline_key]
                item_ids = sorted(set(treatment_groups) | set(baseline_groups))
                metric_deltas: dict[str, list[float]] = defaultdict(list)
                for item_id in item_ids:
                    treatment_rows = sorted(
                        treatment_groups.get(item_id, []),
                        key=lambda row: int(row["seed"]),
                    )
                    baseline_rows = sorted(
                        baseline_groups.get(item_id, []),
                        key=lambda row: int(row["seed"]),
                    )
                    if not treatment_rows or not baseline_rows:
                        continue
                    treatment_values = _problem_metric_values(
                        treatment_rows, proof=proof, ks=ks
                    )
                    baseline_values = _problem_metric_values(
                        baseline_rows, proof=proof, ks=ks
                    )
                    for metric in sorted(
                        set(treatment_values) & set(baseline_values)
                    ):
                        metric_deltas[metric].append(
                            treatment_values[metric] - baseline_values[metric]
                        )
                treatment_identity = _method_output_fields(
                    next(iter(next(iter(treatment_groups.values()))))
                )
                baseline_identity = _method_output_fields(
                    next(iter(next(iter(baseline_groups.values()))))
                )
                for metric, deltas_list in sorted(metric_deltas.items()):
                    deltas = np.asarray(deltas_list, dtype=float)
                    if not len(deltas):
                        continue
                    salt = (
                        f"{seed}:{benchmark}:{treatment_key[0]}:{treatment_key[1]}:"
                        f"{baseline_key[0]}:{baseline_key[1]}:{metric}"
                    )
                    group_seed = int.from_bytes(
                        hashlib.sha256(salt.encode("utf-8")).digest()[:8], "big"
                    )
                    rng = np.random.default_rng(group_seed)
                    draw_indices = rng.integers(
                        0, len(deltas), size=(samples, len(deltas))
                    )
                    draws = deltas[draw_indices].mean(axis=1)
                    output.append(
                        {
                            "benchmark": benchmark,
                            **treatment_identity,
                            "baseline": baseline_identity["method_id"],
                            "baseline_harness_id": baseline_identity["harness_id"],
                            "baseline_harness_entrypoint": baseline_identity[
                                "harness_entrypoint"
                            ],
                            "baseline_harness_source_sha256": baseline_identity[
                                "harness_source_sha256"
                            ],
                            "baseline_harness_access": baseline_identity[
                                "harness_access"
                            ],
                            "metric": metric,
                            "paired_n": len(deltas),
                            "bootstrap_samples": samples,
                            "mean_delta": float(deltas.mean()),
                            "ci95_low": float(np.quantile(draws, 0.025)),
                            "ci95_high": float(np.quantile(draws, 0.975)),
                        }
                    )
    return output


def compute_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    comparisons: Sequence[tuple[str, str]] = DEFAULT_COMPARISONS,
    ks: Sequence[int] = (1, 2, 3),
    expected_seeds: Sequence[int] | None = None,
    answer_compatibility_seed: int = 0,
) -> dict[str, Any]:
    """Build the complete JSON-serializable evaluation report payload."""

    materialized = list(rows)
    summary = summarize_metrics(materialized, expected_seeds=expected_seeds)
    curves = compute_curves(materialized, ks=ks, expected_seeds=expected_seeds)
    compatibility = compatibility_metrics(
        materialized,
        answer_seed=answer_compatibility_seed,
        expected_seeds=expected_seeds,
    )
    bootstrap = paired_bootstrap(
        materialized,
        comparisons=comparisons,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        ks=ks,
        expected_seeds=expected_seeds,
    )
    return {
        "schema_version": 1,
        "complete": bool(summary) and all(row["complete"] for row in summary),
        "summary": summary,
        "curves": curves,
        "compatibility": compatibility,
        "paired_bootstrap": bootstrap,
        "cost_accounting": summarize_cost_accounting(materialized),
        "methodology": {
            "strict_success": "proof score == 7; answer judged correct",
            "missing_policy": "missing, failed, or unjudged scheduled cells score zero",
            "pass_at_k": "unbiased estimator 1-C(n-c,k)/C(n,k)",
            "continuous_reward": "expected maximum grade over size-k subsets",
            "cost_at_k": (
                "expected summed final-attempt solver cost over size-k subsets; incurred "
                "fields additionally include invalidated retries and use an upper bound "
                "when interrupted usage is unknowable"
            ),
            "bootstrap": (
                "paired over problem IDs; matched seeds are collapsed within each problem"
            ),
            "absent_condition": "N/A; no synthetic rows or zero score",
            "method_identity": (
                "harness rows are grouped by harness_id and harness_source_sha256; "
                "legacy rows fall back to condition"
            ),
            "access_class": "blind and reference_assisted methods remain explicitly labeled",
        },
    }


# Compact compatibility alias for callers accustomed to a generic name.
summarize = summarize_metrics


__all__ = [
    "DEFAULT_COMPARISONS",
    "best_grade_at_k",
    "canonical_tokens",
    "compatibility_metrics",
    "compute_curves",
    "compute_metrics",
    "expected_best_at_k",
    "expected_cost_at_k",
    "paired_bootstrap",
    "pass_at_k",
    "strict_success",
    "summarize",
    "summarize_cost_accounting",
    "summarize_metrics",
]
