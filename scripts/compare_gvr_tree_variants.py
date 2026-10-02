"""Compare GVR tree collections problem by problem: diversity, performance, calibration.

Each arm is an artifact root holding one harness's trees (for example the
verdict-routed baseline and the two replanning variants). Rows are rebuilt with
export_gvr_tree_dataset.export_tree, so every number matches the exported
dataset. Only problems solved and node-judged in every arm are compared.

The unit is the problem. Each metric is computed per tree and then averaged
over problems; pooled rates (fix, break, calibration) are computed over all
nodes of an arm, with 95% intervals from a bootstrap over problems. Two primary
metrics, the answer-change rate and the share of verification points whose
children disagree in label, are compared between every pair of arms with a
paired sign-flip permutation test, Holm-corrected; everything else is
descriptive. Per-problem binary outcomes get an exact McNemar test.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from export_gvr_tree_dataset import export_tree

PRIMARY = ("answer_change_rate", "mixed_point_share")
VERDICT_ORDER = {"critical_flaw": 0, "minor_fix": 1, "correct": 2}
CAVEATS = [
    "The baseline differs from the replanning arms in three ways (verifier protocol, planned "
    "instead of verdict-routed actions, promotion rule); only joint vs independent is a clean "
    "contrast.",
    "With about 30 problems, performance differences are a guardrail; diversity carries the "
    "signal.",
    "Diversity from plans that hide the current solution may be restart noise; read change rates "
    "together with fix and break rates and the semantic answer count.",
    "Trees that end early have fewer nodes; rates are per node, and statuses are reported.",
    "Arms are not compute-matched; compare tokens per tree before reading efficiency into "
    "anything.",
]


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Probability that a positive outranks a negative, ties counted half."""

    positives = [s for s, y in zip(scores, labels, strict=True) if y]
    negatives = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not positives or not negatives:
        return None
    ranked = sorted(itertools.chain(((s, 1) for s in positives), ((s, 0) for s in negatives)))
    rank_sum, index = 0.0, 0
    while index < len(ranked):
        end = index
        while end + 1 < len(ranked) and ranked[end + 1][0] == ranked[index][0]:
            end += 1
        average = (index + end) / 2 + 1
        rank_sum += average * sum(flag for _, flag in ranked[index : end + 1])
        index = end + 1
    u = rank_sum - len(positives) * (len(positives) + 1) / 2
    return u / (len(positives) * len(negatives))


def calibration(probabilities: Sequence[float], labels: Sequence[bool]) -> dict[str, Any] | None:
    if not probabilities:
        return None
    brier = statistics.fmean((p - y) ** 2 for p, y in zip(probabilities, labels, strict=True))
    bins: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for p, y in zip(probabilities, labels, strict=True):
        bins[min(9, int(p * 10))].append((p, y))
    ece = sum(
        len(items)
        / len(probabilities)
        * abs(statistics.fmean(p for p, _ in items) - statistics.fmean(y for _, y in items))
        for items in bins.values()
    )
    return {
        "n": len(probabilities),
        "mean_probability": statistics.fmean(probabilities),
        "accuracy": statistics.fmean(labels),
        "auroc": auroc(probabilities, labels),
        "brier": brier,
        "ece": ece,
    }


def _jaccard(left: str, right: str) -> float:
    a, b = set(left.lower().split()), set(right.lower().split())
    return len(a & b) / len(a | b) if a | b else 1.0


def tree_metrics(
    tree: dict[str, Any],
    nodes: list[dict[str, Any]],
    verifications: list[dict[str, Any]],
    plans: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-tree metrics plus the pooled observations an arm aggregates."""

    by_call = {node["node_call_index"]: node for node in nodes}
    children: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        if node["parent_call_index"] is not None:
            children[node["parent_call_index"]].append(node)
    spine = sorted((n for n in nodes if n["on_spine"]), key=lambda n: n["cycle"])
    spine_answers = [n["extracted_answer"] for n in spine]
    labels = [n["judge_correct"] for n in nodes if n["judge_correct"] is not None]
    answers = {
        " ".join(n["extracted_answer"].split()) if n["extracted_answer"] else None for n in nodes
    }
    wrong = {
        " ".join(n["extracted_answer"].split()) if n["extracted_answer"] else None
        for n in nodes
        if n["judge_correct"] is False
    }
    points = [kids for parent, kids in children.items() if parent in by_call]
    mixed_points = [len({k["judge_correct"] for k in kids}) > 1 for kids in points]
    transitions = []
    for node in nodes:
        parent = by_call.get(node["parent_call_index"])
        if parent is not None and node["judge_correct"] is not None:
            transitions.append(
                (bool(parent["judge_correct"]), bool(node["judge_correct"]), node, parent)
            )
    titles_per_point, overlaps = [], []
    for _, rows in itertools.groupby(
        sorted(plans, key=lambda r: r["point"]), key=lambda r: r["point"]
    ):
        rows = [r for r in rows if r["valid"]]
        titles_per_point.append(len({" ".join((r["title"] or "").lower().split()) for r in rows}))
        overlaps += [_jaccard(a["brief"], b["brief"]) for a, b in itertools.combinations(rows, 2)]
    return {
        "metrics": {
            "answer_change_rate": _mean(
                [n["answer_changed"] for n in nodes if n["answer_changed"] is not None]
            ),
            "mixed_point_share": _mean(mixed_points),
            "distinct_answers": len(answers),
            "semantic_distinct_answers": len(wrong) + (1 if any(labels) else 0),
            "single_answer": len(answers) == 1,
            "spine_changes": sum(a != b for a, b in itertools.pairwise(spine_answers)),
            "spine_never_changes": len(set(spine_answers)) == 1,
            "mixed_tree": 0 < sum(labels) < len(labels) if labels else None,
            "any_correct": any(labels) if labels else None,
            "c1_correct": spine[0]["judge_correct"] if spine else None,
            "final_correct": spine[-1]["judge_correct"] if spine else None,
            "node_accuracy": _mean(labels),
            "generated_tokens": tree["generated_tokens"],
            "distinct_titles_per_point": _mean(titles_per_point),
            "brief_overlap_within_point": _mean(overlaps),
            "show_current_solution_share": _mean(
                [r["show_current_solution"] for r in plans if r["valid"]]
            ),
            "planner_recovered_share": _mean([r["planner_recovered"] for r in plans]),
        },
        "transitions": [
            (before, after, node.get("plan_show_current_solution"))
            for before, after, node, _ in transitions
        ],
        "verdicts": [
            (v["verdict"], v["success_probability"], v["candidate_judge_correct"])
            for v in verifications
            if v["candidate_judge_correct"] is not None
        ],
        "plans": [
            (r["success_probability"], r["node_judge_correct"])
            for r in plans
            if r["valid"]
            and r["success_probability"] is not None
            and r["node_judge_correct"] is not None
        ],
        "status": tree["status"],
        "failures": tree.get("failures", {}),
        "tokens_by_role": tree.get("tokens_by_role", {}),
    }


def load_arm(root: Path, problems: Sequence[str] | None) -> dict[str, dict[str, Any]]:
    """Per-problem tree metrics of one artifact root (one harness, one seed)."""

    schedule = json.loads((root / "schedule.json").read_text(encoding="utf-8"))
    prepared: dict[tuple[str, str], dict[str, Any]] = {}
    for path in (root / "prepared" / "benchmarks").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            prepared[(path.stem, str(row["item_id"]))] = row
    harnesses = {item["harness_id"] for item in schedule["items"]}
    if len(harnesses) != 1:
        raise SystemExit(f"{root} holds several harnesses: {sorted(harnesses)}")
    trees = {}
    for item in schedule["items"]:
        if problems is not None and item["problem_id"] not in problems:
            continue
        run = root / "solve" / "runs" / item["run_id"]
        judged_path = root / "node_judge" / "runs" / item["run_id"] / "result.json"
        if not (run / "result.json").exists() or not judged_path.exists():
            continue
        solve = json.loads((run / "result.json").read_text(encoding="utf-8"))
        judged = json.loads(judged_path.read_text(encoding="utf-8"))
        tree, nodes, verifications, plans = export_tree(
            item,
            solve,
            judged,
            prepared.get((item["benchmark"], item["problem_id"]), {}),
            include_reasoning=False,
        )
        trees[item["problem_id"]] = {
            "harness_id": item["harness_id"],
            "bucket": _bucket(prepared.get((item["benchmark"], item["problem_id"]), {})),
            **tree_metrics(tree, nodes, verifications, plans),
        }
    return trees


def _bucket(row: dict[str, Any]) -> str | None:
    rate = row.get("qwen36_pass_rate")
    if rate is None:
        return None
    return "zero" if rate == 0 else "one" if rate == 1 else "low" if rate < 0.5 else "high"


def bootstrap(
    problems: Sequence[str],
    statistic: Callable[[Sequence[str]], float | None],
    *,
    samples: int,
    rng: random.Random,
) -> tuple[float | None, float | None, float | None]:
    """A statistic over problems with a 95% percentile interval from resampling problems."""

    estimate = statistic(problems)
    if estimate is None or len(problems) < 2:
        return estimate, None, None
    draws = []
    for _ in range(samples):
        value = statistic([rng.choice(problems) for _ in problems])
        if value is not None:
            draws.append(value)
    draws.sort()
    return estimate, draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]


def sign_flip_p(differences: Sequence[float], *, samples: int, rng: random.Random) -> float | None:
    """Two-sided paired permutation test of a zero mean difference."""

    values = [d for d in differences if d is not None]
    if not values:
        return None
    observed = abs(statistics.fmean(values))
    if len(values) <= 16:
        flips = itertools.product((1, -1), repeat=len(values))
        extreme = total = 0
        for signs in flips:
            total += 1
            extreme += (
                abs(statistics.fmean(s * v for s, v in zip(signs, values, strict=True)))
                >= observed - 1e-12
            )
        return extreme / total
    extreme = sum(
        abs(statistics.fmean(v if rng.random() < 0.5 else -v for v in values)) >= observed - 1e-12
        for _ in range(samples)
    )
    return (extreme + 1) / (samples + 1)


def mcnemar_p(left: Sequence[bool], right: Sequence[bool]) -> dict[str, Any]:
    """Exact two-sided McNemar test on paired binary outcomes."""

    only_left = sum(a and not b for a, b in zip(left, right, strict=True))
    only_right = sum(b and not a for a, b in zip(left, right, strict=True))
    n = only_left + only_right
    tail = sum(math.comb(n, k) for k in range(min(only_left, only_right) + 1)) / 2**n if n else 1.0
    return {"only_first": only_left, "only_second": only_right, "p": min(1.0, 2 * tail)}


def holm(p_values: dict[str, float | None]) -> dict[str, float | None]:
    present = sorted((p, key) for key, p in p_values.items() if p is not None)
    adjusted: dict[str, float | None] = {key: None for key in p_values}
    running = 0.0
    for rank, (p, key) in enumerate(present):
        running = max(running, min(1.0, (len(present) - rank) * p))
        adjusted[key] = running
    return adjusted


def summarize_arm(
    arm: dict[str, dict[str, Any]], problems: Sequence[str], *, samples: int, rng: random.Random
) -> dict[str, Any]:
    def per_problem(name: str) -> Callable[[Sequence[str]], float | None]:
        return lambda chosen: _mean(
            [arm[p]["metrics"][name] for p in chosen if arm[p]["metrics"][name] is not None]
        )

    def pooled(
        rate: Callable[[list[tuple[Any, ...]]], float | None], key: str
    ) -> Callable[[Sequence[str]], float | None]:
        return lambda chosen: rate([obs for p in chosen for obs in arm[p][key]])

    def fix(observations: list[tuple[Any, ...]]) -> float | None:
        return _mean([after for before, after, _ in observations if not before])

    def broke(observations: list[tuple[Any, ...]]) -> float | None:
        return _mean([not after for before, after, _ in observations if before])

    def false_accept(observations: list[tuple[Any, ...]]) -> float | None:
        return _mean([verdict == "correct" for verdict, _, label in observations if not label])

    def true_accept(observations: list[tuple[Any, ...]]) -> float | None:
        return _mean([verdict == "correct" for verdict, _, label in observations if label])

    metrics = sorted({name for p in problems for name in arm[p]["metrics"]})
    summary: dict[str, Any] = {
        name: bootstrap(problems, per_problem(name), samples=samples, rng=rng) for name in metrics
    }
    summary |= {
        "fix_rate": bootstrap(problems, pooled(fix, "transitions"), samples=samples, rng=rng),
        "break_rate": bootstrap(problems, pooled(broke, "transitions"), samples=samples, rng=rng),
        "verifier_false_accept": bootstrap(
            problems, pooled(false_accept, "verdicts"), samples=samples, rng=rng
        ),
        "verifier_true_accept": bootstrap(
            problems, pooled(true_accept, "verdicts"), samples=samples, rng=rng
        ),
    }
    verdicts = [obs for p in problems for obs in arm[p]["verdicts"]]
    transitions = [obs for p in problems for obs in arm[p]["transitions"]]
    by_show = {
        str(show): {
            "fix_rate": fix([t for t in transitions if t[2] is show]),
            "break_rate": broke([t for t in transitions if t[2] is show]),
        }
        for show in {t[2] for t in transitions}
    }
    plans = [obs for p in problems for obs in arm[p]["plans"]]
    probabilities = [(p, label) for _, p, label in verdicts if p is not None]
    return {
        "harness_id": next(iter({arm[p]["harness_id"] for p in problems})),
        "metrics": {
            name: dict(zip(("estimate", "low", "high"), value, strict=True))
            for name, value in summary.items()
        },
        "by_show_current_solution": by_show,
        "verifier_verdict_auroc": auroc(
            [VERDICT_ORDER.get(v, 1) for v, _, _ in verdicts], [bool(y) for _, _, y in verdicts]
        ),
        "verifier_probability": calibration(
            [p for p, _ in probabilities], [bool(y) for _, y in probabilities]
        ),
        "planner_probability": calibration([p for p, _ in plans], [bool(y) for _, y in plans]),
        "statuses": dict(Counter(arm[p]["status"] for p in problems)),
        "failures": dict(sum((Counter(arm[p]["failures"]) for p in problems), Counter())),
        "tokens_by_role": {
            role: statistics.fmean(arm[p]["tokens_by_role"].get(role, 0) for p in problems)
            for role in sorted({role for p in problems for role in arm[p]["tokens_by_role"]})
        },
        "by_bucket": {
            bucket: {
                name: _mean(
                    [
                        arm[p]["metrics"][name]
                        for p in problems
                        if arm[p]["bucket"] == bucket and arm[p]["metrics"][name] is not None
                    ]
                )
                for name in (
                    "answer_change_rate",
                    "mixed_point_share",
                    "any_correct",
                    "final_correct",
                )
            }
            for bucket in ("zero", "low", "high", "one")
            if any(arm[p]["bucket"] == bucket for p in problems)
        },
    }


def compare(
    arms: dict[str, dict[str, dict[str, Any]]], problems: Sequence[str], *, samples: int, seed: int
) -> dict[str, Any]:
    rng = random.Random(seed)
    pairs = list(itertools.combinations(arms, 2))
    tests: dict[str, Any] = {}
    raw: dict[str, float | None] = {}
    for left, right in pairs:
        for metric in PRIMARY:
            differences = [
                arms[left][p]["metrics"][metric] - arms[right][p]["metrics"][metric]
                for p in problems
                if arms[left][p]["metrics"][metric] is not None
                and arms[right][p]["metrics"][metric] is not None
            ]
            key = f"{left} - {right}: {metric}"
            estimate, low, high = bootstrap(
                list(range(len(differences))),
                lambda chosen, d=differences: _mean([d[i] for i in chosen]),
                samples=samples,
                rng=rng,
            )
            raw[key] = sign_flip_p(differences, samples=samples, rng=rng)
            tests[key] = {
                "mean_difference": estimate,
                "low": low,
                "high": high,
                "p": raw[key],
                "problems": len(differences),
            }
    for key, adjusted in holm(raw).items():
        tests[key]["p_holm"] = adjusted
    binary = {}
    for left, right in pairs:
        for metric in ("mixed_tree", "single_answer", "any_correct", "final_correct", "c1_correct"):
            usable = [
                p
                for p in problems
                if arms[left][p]["metrics"][metric] is not None
                and arms[right][p]["metrics"][metric] is not None
            ]
            binary[f"{left} vs {right}: {metric}"] = mcnemar_p(
                [bool(arms[left][p]["metrics"][metric]) for p in usable],
                [bool(arms[right][p]["metrics"][metric]) for p in usable],
            )
    return {
        "problems": len(problems),
        "arms": {
            name: summarize_arm(arm, problems, samples=samples, rng=rng)
            for name, arm in arms.items()
        },
        "primary_tests": tests,
        "binary_tests": binary,
        "caveats": CAVEATS,
    }


def _problem_ids(manifest: Path) -> list[str]:
    data = json.loads(manifest.read_text())
    if data.get("pilot_problem_ids"):
        return list(data["pilot_problem_ids"])
    schedule = json.loads((Path(data["artifact_root"]) / "schedule.json").read_text())
    problem_of = {item["run_id"]: item["problem_id"] for item in schedule["items"]}
    return sorted(problem_of[run_id] for run_id in data["pilot_run_ids"])


def _table(report: dict[str, Any]) -> str:
    names = (
        "answer_change_rate",
        "mixed_point_share",
        "distinct_answers",
        "semantic_distinct_answers",
        "single_answer",
        "spine_never_changes",
        "mixed_tree",
        "any_correct",
        "c1_correct",
        "final_correct",
        "node_accuracy",
        "fix_rate",
        "break_rate",
        "verifier_false_accept",
        "show_current_solution_share",
        "generated_tokens",
    )
    arms = report["arms"]
    lines = ["metric".ljust(28) + "".join(name[:22].rjust(24) for name in arms)]
    for name in names:
        cells = []
        for arm in arms.values():
            value = arm["metrics"].get(name, {})
            estimate = value.get("estimate")
            if estimate is None:
                cells.append("–".rjust(24))
            elif name == "generated_tokens":
                cells.append(f"{estimate / 1e6:.2f}M".rjust(24))
            elif name in {"distinct_answers", "semantic_distinct_answers"}:
                cells.append(f"{estimate:.2f} [{value['low']:.2f}, {value['high']:.2f}]".rjust(24))
            else:
                cells.append(f"{estimate:.3f} [{value['low']:.2f}, {value['high']:.2f}]".rjust(24))
        lines.append(name.ljust(28) + "".join(cells))
    lines.append("")
    for key, test in report["primary_tests"].items():
        lines.append(
            f"{key}: {test['mean_difference']:+.3f} [{test['low']:+.3f}, {test['high']:+.3f}]  "
            f"p={test['p']:.4f}  p_holm={test['p_holm']:.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", action="append", required=True, help="name=artifact_root")
    parser.add_argument(
        "--problems-manifest", type=Path, help="launch manifest whose pilot to compare"
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wanted = _problem_ids(args.problems_manifest) if args.problems_manifest else None
    arms = {}
    for spec in args.arm:
        name, _, root = spec.partition("=")
        arms[name] = load_arm(Path(root), wanted)
    problems = sorted(set.intersection(*(set(arm) for arm in arms.values())))
    excluded = {name: sorted(set(wanted or arm) - set(problems)) for name, arm in arms.items()}
    report = compare(arms, problems, samples=args.bootstrap, seed=args.seed)
    report["excluded"] = excluded
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(_table(report))


if __name__ == "__main__":
    sys.exit(main())
