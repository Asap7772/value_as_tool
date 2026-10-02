"""Launch branched GVR collection on ArXivMath: pilot, gate, full run, export.

``launch`` snapshots the source, prepares the experiment offline and selects a
pilot stratified by Qwen3.6 pass rate; with ``--submit`` (or a later
``submit-pilot``) it submits the pilot's solve jobs and then one node-judge job.
``pilot-report`` measures the finished pilot against its gate. ``full`` refuses
to run until a human writes ``approve-full.json`` in the run root, then submits
sharded solve and node-judge arrays over the whole schedule (pilot trees are
already complete and are skipped) plus a finalize job. ``finalize`` exports the
flat dataset and writes completion.json. As in submit_qwen35_large_budget.py,
every sbatch intent is recorded before submission, so an ambiguous submission
is never silently repeated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from submit_qwen35_large_budget import (
    accounting,
    manifest_lock,
    now,
    record_command,
    snapshot_source,
    submit_job,
    write_json,
)

REPO = Path(__file__).absolute().parents[1]
SOLVER = ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a")
JUDGE = ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee")
# Data-collection harnesses this launcher runs: worst-case budget constant and job prefix.
HARNESS_PROFILES = {
    "value_as_tool.harnesses.gvr_branched:AgentHarness": ("WORST_CASE_TOKENS", "gvrt"),
    "value_as_tool.harnesses.gvr_replan:JointPlanHarness": ("JOINT_WORST_CASE_TOKENS", "gvrj"),
    "value_as_tool.harnesses.gvr_replan:IndependentPlanHarness": (
        "INDEPENDENT_WORST_CASE_TOKENS",
        "gvri",
    ),
}
EXPECTED_TREES = 1_649
NODES_PER_TREE = 41
SIBLINGS_PER_TREE = 40
QOS = {"high": "g3_scientific-reasoning_high", "shared": "g3_core_shared"}
PILOT_TREES = {"arxivmath_train": 20, "arxivmath_eval": 10}
BUCKETS = ("zero", "low", "high", "one")
GATE = {"complete_share": 0.95, "contained_failure_share": 0.02, "planner_failure_share": 0.02}
FULL_SOLVE_ARRAYS = {"high": "0-191%192", "shared": "192-255%64"}
JUDGE_SHARDS = 16


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _prefix(manifest: dict[str, Any]) -> str:
    return str(manifest.get("job_prefix", "gvrt"))


def _pilot_problems(artifact_root: Path, run_ids: list[str]) -> list[str]:
    schedule = json.loads((artifact_root / "schedule.json").read_text())
    problem_of = {item["run_id"]: item["problem_id"] for item in schedule["items"]}
    return sorted(problem_of[run_id] for run_id in run_ids)


def _sbatch(manifest: dict[str, Any], key: str, qos: str) -> list[str]:
    exports = manifest["exports"]
    if any("," in name or "," in value or "\n" in value for name, value in exports.items()):
        raise ValueError("Slurm export values must not contain commas or newlines")
    logs = Path(manifest["run_root"]) / "logs"
    return [
        "sbatch", "--parsable", "--partition=g3", "--account=scientific-reasoning",
        "--segment=1", f"--qos={qos}", f"--chdir={manifest['source']}",
        "--export=" + ",".join(f"{name}={value}" for name, value in exports.items()),
        f"--job-name={_prefix(manifest)}-{key}", f"--output={logs / (key + '-%A_%a.out')}",
        f"--error={logs / (key + '-%A_%a.err')}",
    ]


def _gpu_job(
    manifest: dict[str, Any],
    key: str,
    qos: str,
    array: str,
    stage: str,
    arguments: list[str],
    dependencies: tuple[str, ...] = (),
) -> list[str]:
    command = _sbatch(manifest, key, qos) + [
        "--gpus-per-node=1", "--cpus-per-task=24", "--mem=96G", "--time=2-00:00:00",
        f"--array={array}",
    ]
    if dependencies:
        command.append("--dependency=afterany:" + ":".join(dependencies))
    return command + [
        str(Path(manifest["source"]) / "scripts/slurm/run_gpu.sbatch"),
        manifest["config"],
        stage,
        *arguments,
    ]


def _run_id_arguments(run_ids: list[str]) -> list[str]:
    return [argument for run_id in run_ids for argument in ("--run-id", run_id)]


def _bucket(rate: float) -> str:
    if rate == 0:
        return "zero"
    if rate == 1:
        return "one"
    return "low" if rate < 0.5 else "high"


def select_pilot(artifact_root: Path) -> list[str]:
    """A deterministic pilot, balanced across Qwen3.6 pass-rate buckets."""

    schedule = json.loads((artifact_root / "schedule.json").read_text())
    chosen: list[str] = []
    for benchmark, quota in PILOT_TREES.items():
        rows = {
            row["item_id"]: row
            for row in map(
                json.loads,
                (artifact_root / "prepared/benchmarks" / f"{benchmark}.jsonl")
                .read_text()
                .splitlines(),
            )
        }
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in schedule["items"]:
            if item["benchmark"] == benchmark:
                buckets[_bucket(rows[item["problem_id"]]["qwen36_pass_rate"])].append(item)
        for items in buckets.values():
            items.sort(key=lambda item: hashlib.sha256(item["problem_id"].encode()).hexdigest())
        picked: list[str] = []
        while len(picked) < quota and any(buckets[name] for name in BUCKETS):
            for name in BUCKETS:
                if buckets[name] and len(picked) < quota:
                    picked.append(buckets[name].pop(0)["run_id"])
        chosen.extend(picked)
    return chosen


def submit_pilot(path: Path, manifest: dict[str, Any]) -> None:
    run_ids = manifest["pilot_run_ids"]
    gpus = manifest["pilot_gpus"]
    solve_ids = []
    for index in range(gpus):
        chunk = run_ids[index::gpus]
        if not chunk:
            continue
        key = f"pilot-solve-{index}"
        solve_ids.append(submit_job(
            path, manifest, key,
            _gpu_job(manifest, key, QOS["high"], "0", "solve", _run_id_arguments(chunk)),
            stage="solve", array="0",
        ))
    key = "pilot-judge-nodes"
    submit_job(
        path, manifest, key,
        _gpu_job(
            manifest, key, QOS["high"], "0", "judge-nodes",
            [*_run_id_arguments(run_ids), "--node-concurrency", "8"],
            tuple(solve_ids),
        ),
        stage="judge-nodes", array="0",
    )
    manifest.update(state="pilot_submitted", pilot_submitted_at=now())
    write_json(path, manifest)


def launch(args: argparse.Namespace) -> Path:
    from value_as_tool.pipeline import load_context

    config = args.config.absolute()
    settings = load_context(config, environment={}).config
    harnesses = tuple(settings.evaluation.harnesses or ())
    if len(harnesses) != 1 or harnesses[0] not in HARNESS_PROFILES:
        raise ValueError(f"expected exactly one of {sorted(HARNESS_PROFILES)}, got {harnesses}")
    harness = harnesses[0]
    attribute, prefix = HARNESS_PROFILES[harness]
    worst_case = int(getattr(importlib.import_module(harness.partition(":")[0]), attribute))
    checks = {
        "solver": (settings.models.solver.name, settings.models.solver.revision) == SOLVER,
        "judge": (settings.models.judge.name, settings.models.judge.revision) == JUDGE,
        "benchmarks": tuple(settings.evaluation.benchmarks) == tuple(PILOT_TREES),
        "seeds": tuple(settings.evaluation.seeds) == (0,),
        "backend": settings.runtime.solver_backend == "sglang"
        and settings.runtime.solver_tensor_parallel_size == 1,
        "budget": settings.budget.generated_tokens >= worst_case,
        "thinking": settings.sampling.enable_thinking
        and settings.sampling.thinking_content_reserve_tokens == 0,
        "shards": settings.runtime.solve_shards == 256
        and settings.runtime.judge_shards == JUDGE_SHARDS,
    }
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        raise ValueError(f"configuration differs from the agreed collection profile: {failed}")
    run_root, artifact_root = args.run_root.absolute(), args.artifact_root.absolute()
    for root in (run_root, artifact_root):
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"refusing nonempty directory: {root}")
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir()
    path = run_root / "submission.json"
    with manifest_lock(path):
        source = run_root / "source"
        digests = snapshot_source(REPO, config, source)
        write_json(run_root / "snapshot-files.json", digests)
        exports = {
            "VALUE_AS_TOOL_REPO_ROOT": str(source),
            "VALUE_AS_TOOL_ENV_FILE": str(REPO / ".env"),
            "VALUE_AS_TOOL_PYTHON": str(REPO / ".venv/bin/python"),
            "VALUE_AS_TOOL_ARTIFACT_ROOT": str(artifact_root),
            "VALUE_AS_TOOL_ASSET_ROOT": str(REPO / "artifacts/assets"),
            "VALUE_AS_TOOL_SGLANG_BIN": str(REPO / ".venv-sglang/bin/sglang"),
            "VALUE_AS_TOOL_VLLM_BIN": "/storage/home/anikaitsingh/.conda/envs/myenv/bin/vllm",
            "TIKTOKEN_ENCODINGS_BASE": str(REPO / "artifacts/assets/tiktoken"),
            # Trees in flight per GPU; not part of the config fingerprint.
            "VALUE_AS_TOOL_MAX_CONCURRENCY": str(args.lanes),
        }
        for name in ("VALUE_AS_TOOL_PYTHON", "VALUE_AS_TOOL_SGLANG_BIN", "VALUE_AS_TOOL_VLLM_BIN"):
            if not os.access(exports[name], os.X_OK):
                raise FileNotFoundError(f"missing executable: {exports[name]}")
        manifest: dict[str, Any] = {
            "schema_version": 1, "created_at": now(), "state": "snapshot_created",
            "run_root": str(run_root), "manifest": str(path), "source": str(source),
            "config": str(source / "experiment.source.yaml"),
            "artifact_root": str(artifact_root), "exports": exports,
            "pilot_gpus": args.pilot_gpus, "lanes": args.lanes,
            "harness": harness, "job_prefix": prefix, "worst_case_tokens": worst_case,
            "source_sha256": hashlib.sha256(
                json.dumps(digests, sort_keys=True).encode()
            ).hexdigest(),
            "jobs": {}, "credentials": "not copied; only the live .env path is exported",
        }
        write_json(path, manifest)
        prepare = record_command(path, manifest, "prepare", [
            "bash", "-c",
            'source "$VALUE_AS_TOOL_REPO_ROOT/scripts/slurm/common.sh" "$1" prepare; '
            'exec "$PYTHON_BIN" "$REPO_ROOT/scripts/prepare_qwen35_large_budget.py" '
            '--config "$EXPERIMENT_CONFIG"',
            f"{prefix}-prepare", manifest["config"],
        ])
        if prepare.returncode:
            raise RuntimeError("offline preparation failed; see logs/prepare.stderr")
        prepared = json.loads(prepare.stdout)
        if prepared["scheduled"] != EXPECTED_TREES:
            raise ValueError(f"unexpected schedule size: {prepared['scheduled']}")
        pilot = select_pilot(artifact_root)
        problems = _pilot_problems(artifact_root, pilot)
        if args.baseline_manifest is not None:
            # Variants are compared problem by problem, so their pilots must coincide.
            baseline = _load_manifest(args.baseline_manifest.absolute())
            expected = _pilot_problems(Path(baseline["artifact_root"]), baseline["pilot_run_ids"])
            if problems != expected:
                raise ValueError("pilot problems differ from the baseline launch's pilot")
            manifest["baseline_manifest"] = str(args.baseline_manifest.absolute())
        manifest.update(
            state="prepared",
            config_fingerprint=prepared["config_fingerprint"],
            schedule_fingerprint=prepared["schedule_fingerprint"],
            prepared=prepared,
            pilot_run_ids=pilot,
            pilot_problem_ids=problems,
        )
        write_json(path, manifest)
        if args.submit:
            submit_pilot(path, manifest)
    print(json.dumps({key: manifest[key] for key in ("state", "run_root", "artifact_root")}))
    return path


def _events(run_dir: Path, attempt: int) -> list[dict[str, Any]]:
    path = run_dir / "attempts" / f"{attempt:06d}" / "events.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row for row in rows if row.get("event") == "request_completed"]


def _seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "mean": round(statistics.fmean(ordered), 3),
        "p90": round(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], 3),
        "max": round(ordered[-1], 3),
    }


def _boxed(text: str) -> str | None:
    from value_as_tool.judging import find_last_boxed_content

    return find_last_boxed_content(text)


def server_throughput(artifact_root: Path, job_ids: list[str]) -> dict[str, Any]:
    """SGLang decode-batch statistics for the given solve jobs' servers."""

    running, generated = [], []
    by_running: dict[int, list[float]] = {}
    for job_id in job_ids:
        for log in (artifact_root / "logs").glob(f"server-{job_id}-*-solve.log"):
            for line in log.read_text(errors="ignore").splitlines():
                if "Decode batch" not in line:
                    continue
                requests = re.search(r"#running-req: (\d+)", line)
                speed = re.search(r"gen throughput \(token/s\): ([\d.]+)", line)
                if requests and speed:
                    running.append(float(requests[1]))
                    generated.append(float(speed[1]))
                    by_running.setdefault(int(requests[1]), []).append(float(speed[1]))
    # Aggregate and per-request decode speed as a function of batch size; the
    # lanes choice for the full run extrapolates from this curve.
    curve = {
        str(count): {
            "samples": len(speeds),
            "tokens_per_second": round(statistics.fmean(speeds), 1),
            "tokens_per_second_per_request": round(statistics.fmean(speeds) / count, 1),
        }
        for count, speeds in sorted(by_running.items())
        if count > 0
    }
    return {
        "running_requests": _summary(running),
        "generated_tokens_per_second_per_gpu": _summary(generated),
        "by_running_requests": curve,
    }


def pilot_report(args: argparse.Namespace) -> dict[str, Any]:
    path = args.manifest.absolute()
    manifest = _load_manifest(path)
    artifact_root = Path(manifest["artifact_root"])
    pilot_jobs = {key: job for key, job in manifest["jobs"].items() if key.startswith("pilot-")}
    rows = {
        row["item_id"]: row
        for name in PILOT_TREES
        for row in map(
            json.loads,
            (artifact_root / "prepared/benchmarks" / f"{name}.jsonl").read_text().splitlines(),
        )
    }
    schedule = {
        item["run_id"]: item
        for item in json.loads((artifact_root / "schedule.json").read_text())["items"]
    }
    trees, tokens, walls, speeds = [], [], [], []
    statuses, failures, recoveries, lengths = Counter(), Counter(), Counter(), Counter()
    first_by_bucket: dict[str, list[bool]] = defaultdict(list)
    spine_by_round: dict[int, list[bool]] = defaultdict(list)
    verdicts_by_point: dict[int, Counter[str]] = defaultdict(Counter)
    recheck_changes: list[bool] = []
    tree_mix: dict[str, Counter[str]] = defaultdict(Counter)
    verdicts_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    label_changes: dict[str, Counter[str]] = defaultdict(Counter)
    answer_changes: dict[str, list[bool]] = defaultdict(list)
    planned: list[dict[str, Any]] = []
    judged_nodes, judgments, node_status = 0, 0, Counter()
    checkpoint_bytes = []
    for run_id in manifest["pilot_run_ids"]:
        run_dir = artifact_root / "solve/runs" / run_id
        result_path = run_dir / "result.json"
        if not result_path.exists():
            statuses["missing"] += 1
            continue
        result = json.loads(result_path.read_text())
        statuses[result["status"]] += 1
        calls = result.get("calls", [])
        tokens.append(result["usage"]["completion_tokens"])
        for call in calls:
            label = call["label"]
            if label.endswith(".t1") or label.endswith(".recovery"):
                recoveries[label.rsplit(".", 2)[-2]] += 1
            if (call.get("response") or {}).get("finish_reason") == "length":
                lengths[call["role"]] += 1
        failures["contained"] += sum(
            transition["action"] == "failed" for transition in result.get("transitions", [])
        )
        planned += [
            json.loads(transition["detail"])
            for transition in result.get("transitions", [])
            if transition["action"] == "plan"
        ]
        events = _events(run_dir, int(result.get("attempt", 1)))
        starts = [event["started_at"] for event in events if event.get("started_at")]
        if starts:
            walls.append(_seconds(min(starts), max(event["at"] for event in events)) or 0)
        for event in events:
            elapsed = _seconds(event.get("started_at"), event.get("at"))
            if elapsed and elapsed > 0:
                speeds.append(event["usage"]["completion_tokens"] / elapsed)
        attempt_dir = run_dir / "attempts" / f"{int(result.get('attempt', 1)):06d}"
        checkpoint = attempt_dir / "checkpoint.json"
        if checkpoint.exists():
            sequence = json.loads(checkpoint.read_text()).get("sequence", 0)
            checkpoint_bytes.append(sequence * result_path.stat().st_size / 2)
        judged_path = artifact_root / "node_judge/runs" / run_id / "result.json"
        labels = {}
        if judged_path.exists():
            judged = json.loads(judged_path.read_text())
            judgments += len(judged.get("judgments", {}))
            for node in judged.get("nodes", []):
                judged_nodes += 1
                node_status[node["judge_status"]] += 1
                labels[node["call_index"]] = node["correct"]
        candidates = {c["call_index"]: c for c in result.get("candidates", [])}
        spine = [c for c in candidates.values() if c["cycle"] == 1 and c.get("branch") is None]
        promoted = {
            (t["cycle"], int(t["source"].removeprefix("branch_")))
            for t in result.get("transitions", [])
            if t["action"] == "promote"
        }
        spine += [c for c in candidates.values() if (c["cycle"], c.get("branch")) in promoted]
        for node in spine:
            if node["call_index"] in labels:
                spine_by_round[node["cycle"]].append(labels[node["call_index"]])
        bucket = _bucket(rows[schedule[run_id]["problem_id"]]["qwen36_pass_rate"])
        if spine and spine[0]["call_index"] in labels:
            first_by_bucket[bucket].append(labels[spine[0]["call_index"]])
        tree_labels = [labels[index] for index in candidates if index in labels]
        if tree_labels:
            tree_mix[bucket][
                "mixed" if 0 < sum(tree_labels) < len(tree_labels)
                else "all_correct" if all(tree_labels) else "all_incorrect"
            ] += 1
        for verdict in result.get("verdicts", []):
            verdicts_by_point[verdict["cycle"]][verdict["verdict"]] += 1
            if verdict.get("parent_call_index") in labels:
                judged = "correct" if labels[verdict["parent_call_index"]] else "incorrect"
                verdicts_by_label[judged][verdict["verdict"]] += 1
        labels_by_call = {call["index"]: call["label"] for call in calls}
        for candidate in candidates.values():
            parent = candidates.get(candidate.get("parent_call_index"))
            label = labels_by_call.get(candidate["call_index"], "")
            if parent and ".recheck." in label:
                recheck_changes.append(_boxed(candidate["content"]) != _boxed(parent["content"]))
            if parent:
                answer_changes[label.split(".")[2]].append(
                    _boxed(candidate["content"]) != _boxed(parent["content"])
                )
            if parent and {candidate["call_index"], parent["call_index"]} <= labels.keys():
                # r{k}.b{j}.{recheck|revise|regenerate}.t{n}: judged parent -> judged child.
                before, after = labels[parent["call_index"]], labels[candidate["call_index"]]
                label_changes[label.split(".")[2]][
                    f"{'C' if before else 'I'}->{'C' if after else 'I'}"
                ] += 1
        trees.append(run_id)
    solved = len(trees)
    complete = statuses.get("cycle_limit", 0)
    siblings = solved * SIBLINGS_PER_TREE
    gate = {
        "all_pilot_jobs_completed": accounting(pilot_jobs)["ready"] if pilot_jobs else False,
        "no_fatal_statuses": set(statuses) <= {"cycle_limit"},
        "complete_share": round(complete / max(1, len(manifest["pilot_run_ids"])), 4),
        "contained_failure_share": round(failures["contained"] / max(1, siblings), 4),
        "judge_coverage": round(judged_nodes / max(1, sum(
            len(json.loads((artifact_root / "solve/runs" / run_id / "result.json").read_text())
                .get("candidates", []))
            for run_id in trees
        )), 4),
    }
    replan = manifest.get("harness", "").startswith("value_as_tool.harnesses.gvr_replan:")
    if replan:
        # A branch without a recorded plan lost its planner, or its tree died first.
        gate["planner_failure_share"] = round(1 - len(planned) / max(1, siblings), 4)
    gate["passed"] = (
        gate["all_pilot_jobs_completed"]
        and gate["no_fatal_statuses"]
        and gate["complete_share"] >= GATE["complete_share"]
        and gate["contained_failure_share"] <= GATE["contained_failure_share"]
        and gate["judge_coverage"] == 1.0
        and gate.get("planner_failure_share", 0.0) <= GATE["planner_failure_share"]
    )
    mean_tokens = statistics.fmean(tokens) if tokens else 0.0
    probabilities = [plan["p"] for plan in planned if plan["p"] is not None]
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "pilot_trees": len(manifest["pilot_run_ids"]),
        "solved_trees": solved,
        "statuses": dict(statuses),
        "generated_tokens_per_tree": _summary([float(value) for value in tokens]),
        "projected_generated_tokens": round(mean_tokens * EXPECTED_TREES),
        "tree_wall_seconds": _summary(walls),
        "per_request_decode_tokens_per_second": _summary(speeds),
        "servers": server_throughput(
            artifact_root,
            [job["job_id"] for key, job in pilot_jobs.items() if key.startswith("pilot-solve")],
        ),
        "recoveries": dict(recoveries),
        "length_finishes_by_role": dict(lengths),
        "contained_failures": failures["contained"],
        "estimated_checkpoint_bytes_written_per_tree": _summary(checkpoint_bytes),
        "node_judge": {
            "nodes": judged_nodes,
            "judgments": judgments,
            "dedup_ratio": round(judgments / max(1, judged_nodes), 4),
            "statuses": dict(node_status),
        },
        "first_candidate_accuracy_by_qwen36_bucket": {
            name: {"trees": len(values), "accuracy": round(statistics.fmean(values), 4)}
            for name, values in sorted(first_by_bucket.items())
            if values
        },
        "spine_accuracy_by_round": {
            str(cycle): round(statistics.fmean(values), 4)
            for cycle, values in sorted(spine_by_round.items())
            if values
        },
        "verdicts_by_point": {str(k): dict(v) for k, v in sorted(verdicts_by_point.items())},
        "answer_change_share_by_mode": {
            mode: round(statistics.fmean(values), 4)
            for mode, values in sorted(answer_changes.items())
        },
        "plans": {
            "planned_branches": len(planned),
            "recovered_share": round(statistics.fmean(p["recovered"] for p in planned), 4),
            "show_current_solution_share": round(statistics.fmean(p["show"] for p in planned), 4),
            "missing_probability_share": round(
                statistics.fmean(p["p"] is None for p in planned), 4
            ),
            "mean_success_probability": round(statistics.fmean(probabilities), 4)
            if probabilities
            else None,
        }
        if planned
        else None,
        "recheck_changes_answer_share": round(statistics.fmean(recheck_changes), 4)
        if recheck_changes
        else None,
        # Whether a tree carries within-tree label contrast, how each revision
        # mode moves the judge label, and how verdicts track the judge label.
        "tree_label_mix_by_qwen36_bucket": {
            name: dict(counts) for name, counts in sorted(tree_mix.items())
        },
        "judge_label_changes_by_mode": {
            mode: dict(sorted(counts.items())) for mode, counts in sorted(label_changes.items())
        },
        "verdicts_by_judge_label": {
            name: dict(counts) for name, counts in sorted(verdicts_by_label.items())
        },
        "gate": gate,
        "gate_thresholds": GATE,
    }
    write_json(Path(manifest["run_root"]) / "pilot-report.json", report)
    print(json.dumps(report, indent=1))
    return report


def full(args: argparse.Namespace) -> None:
    path = args.manifest.absolute()
    with manifest_lock(path):
        manifest = _load_manifest(path)
        approval = Path(manifest["run_root"]) / "approve-full.json"
        if not approval.exists():
            raise SystemExit(f"write {approval} to approve the full run")
        if args.lanes is not None:
            manifest["exports"]["VALUE_AS_TOOL_MAX_CONCURRENCY"] = str(args.lanes)
            manifest["lanes"] = args.lanes
        solve_ids = []
        for pool, array in FULL_SOLVE_ARRAYS.items():
            key = f"solve-{pool}"
            solve_ids.append(submit_job(
                path, manifest, key,
                _gpu_job(manifest, key, QOS[pool], array, "solve", ["--shard-count", "256"]),
                stage="solve", array=array,
            ))
        key = "judge-nodes"
        array = f"0-{JUDGE_SHARDS - 1}%{JUDGE_SHARDS}"
        judge_id = submit_job(
            path, manifest, key,
            _gpu_job(
                manifest, key, QOS["high"], array, "judge-nodes",
                ["--shard-count", str(JUDGE_SHARDS), "--node-concurrency", "8"],
                tuple(solve_ids),
            ),
            stage="judge-nodes", array=array,
        )
        submit_job(
            path, manifest, "finalize",
            _sbatch(manifest, "finalize", QOS["high"]) + [
                "--cpus-per-task=8", "--mem=64G", "--time=1-00:00:00",
                f"--dependency=afterany:{judge_id}",
                str(Path(manifest["source"]) / "scripts/slurm/run_tree_controller.sbatch"),
                manifest["config"], "finalize", str(path),
            ],
            stage="controller",
        )
        manifest.update(state="full_submitted", full_submitted_at=now())
        write_json(path, manifest)


def finalize(args: argparse.Namespace) -> None:
    path = args.manifest.absolute()
    manifest = _load_manifest(path)
    export = record_command(path, manifest, "export", [
        manifest["exports"]["VALUE_AS_TOOL_PYTHON"],
        str(Path(manifest["source"]) / "scripts/export_gvr_tree_dataset.py"),
        "--artifact-root", manifest["artifact_root"],
        "--output", str(Path(manifest["artifact_root"]) / "export"),
    ])
    full_jobs = {
        key: job for key, job in manifest["jobs"].items()
        if key in {"solve-high", "solve-shared", "judge-nodes"}
    }
    jobs = accounting(full_jobs)
    counts = json.loads(export.stdout)["counts"] if export.returncode == 0 else {}
    complete = (
        export.returncode == 0
        and jobs["ready"]
        and counts.get("trees") == EXPECTED_TREES
        and not counts.get("unjudged_trees")
    )
    completion = {
        "completed_at": now(), "complete": complete, "export_counts": counts,
        "config_fingerprint": manifest.get("config_fingerprint"),
        "schedule_fingerprint": manifest.get("schedule_fingerprint"),
        "slurm_accounting": jobs,
        "export": str(Path(manifest["artifact_root"]) / "export"),
    }
    write_json(Path(manifest["run_root"]) / "completion.json", completion)
    manifest.update(state="complete" if complete else "incomplete", completion=completion)
    write_json(path, manifest)
    if not complete:
        raise SystemExit("collection is incomplete; see completion.json")


def status(args: argparse.Namespace) -> None:
    manifest = _load_manifest(args.manifest.absolute())
    jobs = {key: job for key, job in manifest["jobs"].items() if job.get("job_id")}
    squeue = subprocess.run(
        ["squeue", "-h", "-u", os.environ.get("USER", ""), "-n",
         ",".join(f"{_prefix(manifest)}-{key}" for key in jobs), "-o", "%j %T"],
        capture_output=True, text=True, check=False,
    )
    artifact_root = Path(manifest["artifact_root"])
    print(json.dumps({
        "state": manifest.get("state"),
        "jobs": {key: job["job_id"] for key, job in jobs.items()},
        "queue": Counter(line.split()[-1] for line in squeue.stdout.splitlines() if line.strip()),
        "solved_trees": len(list((artifact_root / "solve/runs").glob("*/result.json"))),
        "node_judged_trees": len(list((artifact_root / "node_judge/runs").glob("*/result.json"))),
    }, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("launch", help="snapshot, prepare and plan the pilot")
    start.add_argument("--config", type=Path, required=True)
    start.add_argument("--run-root", type=Path, required=True)
    start.add_argument("--artifact-root", type=Path, required=True)
    start.add_argument("--pilot-gpus", type=int, default=8)
    start.add_argument("--lanes", type=int, default=2, help="trees in flight per GPU")
    start.add_argument("--submit", action="store_true", help="also submit the pilot")
    start.add_argument(
        "--baseline-manifest", type=Path, default=None,
        help="an earlier launch whose pilot problems this pilot must match",
    )
    for name in ("submit-pilot", "pilot-report", "finalize", "status"):
        commands.add_parser(name).add_argument("--manifest", type=Path, required=True)
    grow = commands.add_parser("full", help="submit the full run after approve-full.json")
    grow.add_argument("--manifest", type=Path, required=True)
    grow.add_argument("--lanes", type=int, default=None, help="retune trees in flight per GPU")
    args = parser.parse_args()
    if args.command == "launch":
        launch(args)
    elif args.command == "submit-pilot":
        path = args.manifest.absolute()
        with manifest_lock(path):
            manifest = _load_manifest(path)
            if manifest.get("state") != "prepared":
                raise SystemExit(f"pilot is not awaiting submission: {manifest.get('state')}")
            submit_pilot(path, manifest)
    else:
        {"pilot-report": pilot_report, "full": full, "finalize": finalize, "status": status}[
            args.command
        ](args)


if __name__ == "__main__":
    main()
