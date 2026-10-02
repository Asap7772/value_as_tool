"""Build the aggregate statistics behind the Space's Summary and Analysis tabs.

Reads each experiment's report/rows.jsonl: one row per scheduled trajectory, scored the way the
official report scores it (missing, failed or unjudged runs count as 0, strict success is 7/7).
Writes index/summary.json into the viewer's data directory. Calls per run come from that
directory's rollout index, so run build_data.py first.

Confidence intervals are 95% percentile bootstraps over problems (seeds stay with their problem;
the pooled scope resamples within each benchmark). Method-vs-Direct statistics compare every
Direct run with every method run on the same problem, since the two never share a sample.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from build_data import ARTIFACTS, BENCHMARKS, EXPERIMENTS, MODES, STATUS_CODES

# Where each experiment's Direct baseline comes from. The attempt-conditioned run has no Direct
# arm; its evidence bank is Direct seeds 0-7 of the 9B harness run.
DIRECT_SOURCE = {"q9_base": "q9_base", "q9_attempt": "q9_base", "q27_base": "q27_base"}
BUCKETS = {
    "direct": "generation",
    "generator": "generation",
    "reviser": "generation",
    "value_solver": "generation",
    "planner": "generation",
    "worker": "generation",
    "verifier": "verification",
    "value_verifier": "verification",
    "reviewer": "verification",
    "subagent": "subagents",
}
SCOPES = ("all", *BENCHMARKS)
STRATA = ("never", "sometimes", "always")
FIRST_CANDIDATE = ("fixed", "right_first", "revised_wrong", "kept_wrong", "no_output")
RESAMPLES = 2000
CALIBRATION_BINS = 10  # at most, of equal size
MIN_BIN_RUNS = 30


def family(arm):
    if arm.startswith("gvr") or re.match(r"attempt_.+_gvr_", arm):
        return "gvr"
    if arm.startswith("value_tool") or re.match(r"attempt_.+_value_tool_", arm):
        return "value_tool"
    return "other"


def load_rows(artifact):
    rows = []
    with open(ARTIFACTS / artifact / "report" / "rows.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("applicable", True):
                continue
            rows.append(
                {
                    "arm": r["harness_id"],
                    "key": (r["benchmark"], r["problem_id"]),
                    "seed": r["seed"],
                    "score": r["score"],
                    "win": r["score"] == 7,
                    "status": r["solve_status"],
                    "tokens": r["generated_tokens"] or 0,
                    "prompt": (r["total_tokens"] or 0) - (r["generated_tokens"] or 0),
                    "roles": r["generated_tokens_by_role"] or {},
                    "queries": r["value_query_count"] or 0,
                    # The value tool's last estimate is its forecast closest to the final answer.
                    "estimate": max(
                        r["value_estimates"], key=lambda e: e["query_index"]
                    )["probability"]
                    if r["value_estimates"]
                    else None,
                    "verdicts": [a["verdict"] for a in r["verifier_assessments"] or []],
                }
            )
    return rows


def load_calls(data):
    calls = {}
    for bench in BENCHMARKS:
        path = data / "index" / f"{bench}.json.gz"
        if not path.exists():
            continue
        for row in json.loads(gzip.decompress(path.read_bytes()))["problems"]:
            for exp, arms in row["cells"].items():
                for arm, cells in arms.items():
                    for cell in cells:
                        calls[(exp, bench, row["id"], arm, cell[0])] = cell[5]
    return calls


def resample_index(problems, seed):
    """Bootstrap problem indices, resampling within each benchmark."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, (bench, _) in enumerate(problems):
        groups[bench].append(i)
    return np.concatenate(
        [rng.choice(np.array(ix), size=(RESAMPLES, len(ix))) for ix in groups.values()], axis=1
    )


def interval(samples):
    samples = samples[np.isfinite(samples)]
    return [round(float(x), 4) for x in np.percentile(samples, [2.5, 97.5])]


def ratio(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


def per_problem(runs, problems):
    """Wins and run counts aligned to ``problems``."""
    position = {key: i for i, key in enumerate(problems)}
    wins, count = np.zeros(len(problems)), np.zeros(len(problems))
    for r in runs:
        i = position[r["key"]]
        count[i] += 1
        wins[i] += r["win"]
    return wins, count


def calibration(runs, problems, weights):
    """How well the value tool's last success_probability forecasts a 7/7 final answer.

    Brier score and AUROC get problem-level bootstrap intervals: ``weights`` holds each problem's
    multiplicity in every resample, so both statistics stay sums over problem blocks. Reliability
    bins hold equal numbers of runs (sorted by p), so every bin is equally well estimated.
    """
    runs = [r for r in runs if r["estimate"] is not None]
    if not runs:
        return None
    position = {key: i for i, key in enumerate(problems)}
    p = np.array([r["estimate"] for r in runs], dtype=float)
    y = np.array([r["win"] for r in runs], dtype=float)
    block = np.array([position[r["key"]] for r in runs])
    onehot = np.zeros((len(runs), len(problems)))
    onehot[np.arange(len(runs)), block] = 1
    squared, count = onehot.T @ (p - y) ** 2, onehot.sum(0)
    brier = weights @ squared / (weights @ count)
    rate = y.mean()
    out = {
        "runs": len(runs),
        "wins": int(y.sum()),
        "mean_p": round(float(p.mean()), 4),
        "rate": round(float(rate), 4),
        "brier": round(float(((p - y) ** 2).mean()), 4),
        "brier_ci": interval(brier),
        "brier_base": round(float(rate * (1 - rate)), 4),
        "bins": [
            # [runs, sum of p, wins, lowest p, highest p]
            [len(chunk), round(float(p[chunk].sum()), 6), int(y[chunk].sum()),
             round(float(p[chunk].min()), 4), round(float(p[chunk].max()), 4)]
            for chunk in np.array_split(
                np.argsort(p, kind="mergesort"),
                max(1, min(CALIBRATION_BINS, len(runs) // MIN_BIN_RUNS)),
            )
        ],
    }
    pos, neg = y == 1, y == 0
    if pos.any() and neg.any():
        # pairs[a, b]: (positive run in problem a) outranks (negative run in problem b), ties half.
        beats = (p[pos][:, None] > p[neg][None, :]) + 0.5 * (p[pos][:, None] == p[neg][None, :])
        pairs = onehot[pos].T @ beats @ onehot[neg]
        auroc = ratio(
            ((weights @ pairs) * weights).sum(1),
            (weights @ onehot[pos].sum(0)) * (weights @ onehot[neg].sum(0)),
        )
        out["auroc"] = round(float(beats.mean()), 4)
        out["auroc_ci"] = interval(auroc)
    return out


def first_candidate(r):
    if r["score"] is None or not r["verdicts"]:
        return "no_output"
    if r["verdicts"][0] == "correct":
        return "right_first" if r["win"] else "kept_wrong"
    return "fixed" if r["win"] else "revised_wrong"


def arm_stats(runs, problems, idx, weights, direct, strata, calls):
    n = len(runs)
    wins, count = per_problem(runs, problems)
    tokens = np.array([r["tokens"] for r in runs], dtype=float)
    by_status = defaultdict(lambda: [0, 0])
    for r in runs:
        by_status[r["status"]][0] += 1
        by_status[r["status"]][1] += r["win"]
    roles = Counter()
    for r in runs:
        roles.update(r["roles"])
    buckets = Counter()
    for role, value in roles.items():
        buckets[BUCKETS.get(role, "generation")] += value
    out = {
        "runs": n,
        "wins": int(wins.sum()),
        "success": round(float(wins.sum() / n), 4),
        "ci": interval(wins[idx].sum(1) / count[idx].sum(1)),
        "mean_grade": round(sum(r["score"] or 0 for r in runs) / n, 3),
        "judged": sum(r["score"] is not None for r in runs),
        "status": {status: v[0] for status, v in by_status.items()},
        "success_by_status": {status: v for status, v in by_status.items()},
        "tokens": {
            "mean": round(float(tokens.mean())),
            "median": round(float(np.median(tokens))),
            "p90": round(float(np.percentile(tokens, 90))),
            "prompt": round(sum(r["prompt"] for r in runs) / n),
            "buckets": {bucket: round(value / n) for bucket, value in buckets.items()},
            "roles": {role: round(value / n) for role, value in sorted(roles.items())},
        },
        "strata": {s: [0, 0] for s in STRATA},
    }
    run_calls = [calls[r["calls_key"]] for r in runs if r["calls_key"] in calls]
    if run_calls:
        out["calls"] = {"mean": round(sum(run_calls) / len(run_calls), 2), "max": max(run_calls)}
    for r in runs:
        out["strata"][strata[r["key"]]][0] += 1
        out["strata"][strata[r["key"]]][1] += r["win"]

    kind = family(runs[0]["arm"])
    if kind == "gvr":
        accepted_at = Counter(len(r["verdicts"]) for r in runs if r["status"] == "accepted")
        categories = {s: Counter() for s in (*STRATA, "all")}
        for r in runs:
            category = first_candidate(r)
            categories[strata[r["key"]]][category] += 1
            categories["all"][category] += 1
        out["gvr"] = {
            "verdict_counts": dict(sorted(Counter(len(r["verdicts"]) for r in runs).items())),
            "accepted_at": dict(sorted(accepted_at.items())),
            "first_verdict": dict(Counter(r["verdicts"][0] for r in runs if r["verdicts"])),
            "first_candidate": {
                s: {c: counter[c] for c in FIRST_CANDIDATE} for s, counter in categories.items()
            },
        }
    if kind == "value_tool":
        by_queries = defaultdict(lambda: [0, 0])
        for r in runs:
            by_queries[r["queries"]][0] += 1
            by_queries[r["queries"]][1] += r["win"]
        out["value_tool"] = {
            "query_counts": {q: v[0] for q, v in sorted(by_queries.items())},
            "success_by_queries": {q: v for q, v in sorted(by_queries.items())},
            "mean_queries": round(sum(r["queries"] for r in runs) / n, 3),
            "calibration": calibration(runs, problems, weights),
        }

    if runs[0]["arm"] == "direct":
        return out  # Direct against itself is the re-sampling reference in direct_reference.
    # Method vs Direct over all (Direct run, method run) pairs on the same problem.
    d, m = direct["rate"], ratio(wins, count)
    fixed, broken = (1 - d) * m, d * (1 - m)

    def stat(num, den):
        return round(float(num.sum() / den.sum()), 4), interval(
            ratio(num[idx].sum(1), den[idx].sum(1))
        )

    fix, fix_ci = stat(fixed, 1 - d)
    brk, brk_ci = stat(broken, d)
    net = m - d
    out["vs_direct"] = {
        "fix": fix,
        "fix_ci": fix_ci,
        "break": brk,
        "break_ci": brk_ci,
        "net": round(float(net.mean()), 4),
        "net_ci": interval(net[idx].mean(1)),
    }
    return out


def direct_reference(runs, problems, idx):
    """Direct's own success, strata and the flip rates of re-sampling Direct alone."""
    wins, count = per_problem(runs, problems)
    rate = wins / count
    # Two distinct Direct runs on one problem: P(first wrong, second right) = c(n-c)/(n(n-1)).
    flip = wins * (count - wins) / (count * (count - 1))

    def stat(num, den):
        return round(float(num.sum() / den.sum()), 4), interval(
            ratio(num[idx].sum(1), den[idx].sum(1))
        )

    fix, fix_ci = stat(flip, 1 - rate)
    brk, brk_ci = stat(flip, rate)
    strata = {
        key: "never" if w == 0 else "always" if w == c else "sometimes"
        for key, w, c in zip(problems, wins, count, strict=True)
    }
    return {
        "rate": rate,
        "strata": strata,
        "summary": {
            "runs": int(count.sum()),
            "wins": int(wins.sum()),
            "success": round(float(wins.sum() / count.sum()), 4),
            "ci": interval(wins[idx].sum(1) / count[idx].sum(1)),
            "tokens_mean": round(sum(r["tokens"] for r in runs) / len(runs)),
            "runs_per_problem": sorted({int(c) for c in count}),
            "problems": {s: sum(v == s for v in strata.values()) for s in STRATA},
            "resample": {"fix": fix, "fix_ci": fix_ci, "break": brk, "break_ci": brk_ci},
        },
    }


def experiment_summary(exp_id, rows_by_artifact, calls):
    spec = EXPERIMENTS[exp_id]
    rows = rows_by_artifact[spec["artifact"]]
    direct_rows = [
        r
        for r in rows_by_artifact[EXPERIMENTS[DIRECT_SOURCE[exp_id]]["artifact"]]
        if r["arm"] == "direct"
    ]
    for r in rows:
        r["calls_key"] = (exp_id, *r["key"], r["arm"], r["seed"])
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    scopes = {}
    for scope in SCOPES:
        in_scope = (lambda key: True) if scope == "all" else (lambda key, s=scope: key[0] == s)
        problems = sorted({r["key"] for r in rows if in_scope(r["key"])})
        idx = resample_index(problems, seed=len(scopes))
        weights = np.stack([np.bincount(row, minlength=len(problems)) for row in idx]).astype(float)
        direct = direct_reference([r for r in direct_rows if in_scope(r["key"])], problems, idx)
        arms = {}
        for arm, _, _ in spec["arms"]:
            runs = [r for r in by_arm.get(arm, []) if in_scope(r["key"])]
            if runs:
                arms[arm] = {"family": family(arm)} | arm_stats(
                    runs, problems, idx, weights, direct, direct["strata"], calls
                )
        # Calibration series: each value-tool arm, or, past three arms, arms pooled by
        # evidence type.
        value_arms = [arm for arm in arms if family(arm) == "value_tool"]
        labels = {arm: label for arm, label, _ in spec["arms"]}
        groups = (
            [(arm, labels[arm], [arm]) for arm in value_arms]
            if len(value_arms) <= 3
            else [
                (mode, label, [a for a in value_arms if a.startswith(f"attempt_{mode}_")])
                for mode, label in MODES
            ]
        )
        series = []
        for group, label, members in groups:
            runs = [r for arm in members for r in by_arm[arm] if in_scope(r["key"])]
            stats = calibration(runs, problems, weights)
            if stats:
                series.append({"id": group, "label": label, "arms": members} | stats)
        scopes[scope] = {
            "problems": len(problems),
            "direct": {"source": DIRECT_SOURCE[exp_id]} | direct["summary"],
            "arms": arms,
            "calibration": series,
        }
    return {"artifact": spec["artifact"], "rows": len(rows), "scopes": scopes}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/checkpoint/fort/anikaitsingh/value_as_tool_rollout_viewer/data"),
    )
    parser.add_argument(
        "--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS)
    )
    args = parser.parse_args()
    data = args.data.resolve()
    calls = load_calls(data)
    print(f"{len(calls)} rollout call counts from {data / 'index'}", flush=True)
    needed = {EXPERIMENTS[e]["artifact"] for e in args.experiments}
    needed |= {EXPERIMENTS[DIRECT_SOURCE[e]]["artifact"] for e in args.experiments}
    rows_by_artifact = {}
    for artifact in sorted(needed):
        rows_by_artifact[artifact] = load_rows(artifact)
        print(f"  {artifact}: {len(rows_by_artifact[artifact])} rows", flush=True)
    summary = {
        "resamples": RESAMPLES,
        "calibration_bins": {"max": CALIBRATION_BINS, "min_runs": MIN_BIN_RUNS},
        "status_codes": STATUS_CODES,
        "experiments": {
            exp_id: experiment_summary(exp_id, rows_by_artifact, calls)
            for exp_id in args.experiments
        },
    }
    out = data / "index" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)", flush=True)


if __name__ == "__main__":
    main()
