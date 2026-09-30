"""Bounded conditioning launch, persistent CPU supervision, and final reporting.

``launch --dry-run`` prepares reviewable inputs and commands without submitting
anything. Only ``launch --submit`` submits the CPU supervisor. It admits GPU
workers for the current stage against the user's other same-QoS requests, and
never pre-submits dependent GPU arrays. Ambiguous sbatch intents fail closed.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from attempt_conditioning_queue import TaskQueue
from attempt_conditioning_resources import LIMITS, QOS, admission_plan, query_user_jobs

STAGES = ("preprocess", "pilot_solve", "pilot_judge", "solve", "judge")
COPY_ITEMS = ("src", "scripts", "prompts", "pyproject.toml", "uv.lock")
SOLVE_OUTCOMES = {
    "accepted",
    "completed",
    "cycle_limit",
    "protocol_error",
    "context_exhausted",
    "budget_exhausted",
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@contextmanager
def manifest_lock(path: Path):
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def source_files(source: Path) -> dict[str, str]:
    result = {}
    for name in (
        *COPY_ITEMS,
        *(p.name for p in source.glob("*.yaml")),
        *(p.name for p in source.glob("*.yml")),
    ):
        root = source / name
        if not root.exists():
            raise FileNotFoundError(root)
        candidates = root.rglob("*") if root.is_dir() else [root]
        for path in candidates:
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.name == ".env" or path.name.startswith(".env."):
                raise ValueError("credentials must not enter the source snapshot")
            if path.is_symlink():
                raise ValueError(f"source snapshot contains a mutable symlink: {path}")
            if path.is_file():
                result[str(path.relative_to(source))] = digest(path)
    return dict(sorted(result.items()))


def verify_source(manifest: dict[str, Any]) -> None:
    source = Path(manifest["source"])
    if source_files(source) != manifest["source_files"]:
        raise RuntimeError("immutable source snapshot changed")
    for key in ("preprocess_config", "config"):
        if manifest.get(key + "_sha256"):
            if digest(Path(manifest[key])) != manifest[key + "_sha256"]:
                raise RuntimeError(f"pinned {key} changed")
    if digest(Path(manifest["bank_root"]) / "bank.json") != manifest["bank_sha256"]:
        raise RuntimeError("pinned attempt bank changed")


def execution_environment(manifest: dict[str, Any]) -> dict[str, str]:
    return {
        **os.environ,
        **manifest["exports"],
        "PYTHONPATH": str(Path(manifest["source"]) / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def base_sbatch(manifest: dict[str, Any], key: str, pool: str) -> list[str]:
    exports = manifest["exports"]
    if any("," in v or "\n" in v for v in exports.values()):
        raise ValueError("Slurm export paths must not contain commas or newlines")
    logs = Path(manifest["control_root"]) / "logs"
    return [
        "sbatch",
        "--parsable",
        "--partition=g3",
        "--account=scientific-reasoning",
        "--segment=1",
        f"--qos={QOS[pool]}",
        f"--chdir={manifest['source']}",
        "--export=" + ",".join(f"{k}={v}" for k, v in exports.items()),
        f"--job-name=attempt-{key}",
        f"--output={logs / (key + '-%A_%a.out')}",
        f"--error={logs / (key + '-%A_%a.err')}",
    ]


def worker_command(
    manifest: dict[str, Any], stage: str, pool: str, workers: int, key: str
) -> list[str]:
    if stage != manifest["stage"] or stage not in STAGES:
        raise ValueError("only the current stage may submit GPU workers")
    if not 1 <= workers <= LIMITS[pool]:
        raise ValueError("invalid worker count")
    return base_sbatch(manifest, key, pool) + [
        "--nodes=1",
        "--ntasks=1",
        "--gpus-per-node=1",
        "--cpus-per-task=24",
        "--mem=96G",
        "--time=7-00:00:00",
        f"--array=0-{workers - 1}",
        str(Path(manifest["source"]) / "scripts/slurm/run_attempt_worker.sbatch"),
        manifest["preprocess_config"] if stage == "preprocess" else manifest["config"],
        stage,
        manifest["manifest"],
        manifest["queues"][stage]["path"],
    ]


def controller_command(manifest: dict[str, Any]) -> list[str]:
    return base_sbatch(manifest, "controller", "high") + [
        "--cpus-per-task=8",
        "--mem=32G",
        "--time=7-00:00:00",
        str(Path(manifest["source"]) / "scripts/slurm/run_attempt_controller.sbatch"),
        manifest["preprocess_config"],
        manifest["manifest"],
    ]


def submit_job(
    path: Path,
    manifest: dict[str, Any],
    key: str,
    command: list[str],
    *,
    stage: str,
    pool: str | None = None,
    workers: int = 0,
) -> str:
    existing = manifest["jobs"].get(key)
    if existing:
        if existing["command"] == command and existing.get("job_id"):
            return existing["job_id"]
        raise RuntimeError(f"ambiguous or changed submission {key}; inspect Slurm before recovery")
    entry = {
        "stage": stage,
        "pool": pool,
        "workers": workers,
        "command": command,
        "intent_at": now(),
    }
    manifest["jobs"][key] = entry
    write_json(path, manifest)
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60)
    job_id = result.stdout.strip().split(";", 1)[0]
    entry.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    if result.returncode or not re.fullmatch(r"\d+", job_id):
        entry["state"] = "submission_uncertain"
        write_json(path, manifest)
        raise RuntimeError(f"sbatch did not confirm {key}")
    entry.update(job_id=job_id, state="submitted", submitted_at=now())
    write_json(path, manifest)
    return job_id


def observe_owned_jobs(manifest: dict[str, Any]) -> dict[str, dict[str, str]]:
    jobs = [
        j for j in manifest["jobs"].values() if j.get("pool") and not j.get("terminal_confirmed")
    ]
    if any(not j.get("job_id") for j in jobs):
        raise RuntimeError("unresolved GPU submission intent")
    if not jobs:
        return {}
    command = [
        "sacct",
        "-X",
        "-n",
        "-P",
        "-j",
        ",".join(j["job_id"] for j in jobs),
        "--format=JobID%64,State%40,ExitCode",
    ]
    output = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    records = {}
    for line in output.stdout.splitlines():
        job_id, state, exit_code = line.split("|")[:3]
        if job_id in records:
            raise RuntimeError(f"ambiguous current accounting for {job_id}")
        records[job_id] = {"state": state.split()[0], "exit_code": exit_code}
    terminal = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}
    for job in jobs:
        expected = [f"{job['job_id']}_{i}" for i in range(job["workers"])]
        selected = {key: records[key] for key in expected if key in records}
        job["accounting"] = selected
        job["observed_at"] = now()
        if len(selected) == len(expected) and all(
            r["state"] in terminal for r in selected.values()
        ):
            job["terminal_confirmed"] = True
            job["clean_exit"] = all(
                r == {"state": "COMPLETED", "exit_code": "0:0"} for r in selected.values()
            )
    return records


def queue_tasks(manifest: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    if stage == "preprocess":
        from value_as_tool.conditioning import enumerate_summary_tasks

        return enumerate_summary_tasks(manifest["bank_root"])
    schedule = json.loads((Path(manifest["artifact_root"]) / "schedule.json").read_text())
    pilot = set(manifest["pilot_run_ids"])
    selected = [
        i for i in schedule["items"] if (i["run_id"] in pilot) == stage.startswith("pilot_")
    ]
    return [
        {
            "task_id": i["run_id"],
            "run_id": i["run_id"],
            "benchmark": i["benchmark"],
            "problem_id": i["problem_id"],
            "method": i["harness_id"],
            "seed": i["seed"],
            "dependencies": [],
        }
        for i in selected
    ]


def initialize_stage(manifest: dict[str, Any], stage: str) -> TaskQueue:
    tasks = queue_tasks(manifest, stage)
    path = Path(manifest["control_root"]) / "queues" / f"{stage}.sqlite3"
    identity = {
        "stage": stage,
        "source_sha256": manifest["source_sha256"],
        "bank_sha256": manifest["bank_sha256"],
        "config_sha256": manifest["preprocess_config_sha256"]
        if stage == "preprocess"
        else manifest["config_sha256"],
        "schedule_fingerprint": manifest.get("schedule_fingerprint"),
    }
    queue = TaskQueue(path)
    queue.initialize(tasks, identity)
    manifest["queues"][stage] = {"path": str(path), "tasks": len(tasks), "identity": identity}
    manifest["stage"] = stage
    return queue


def choose_pilot(
    schedule: list[dict[str, Any]], bank: dict[str, Any], requested: dict[str, str]
) -> tuple[list[str], dict[str, str]]:
    selected_problems = {}
    for benchmark, problems in sorted(bank["problems"].items()):
        ranked = sorted(
            problems,
            key=lambda p: (problems[p].get("source_usage", {}).get("completion_tokens", 0), p),
        )
        problem = requested.get(benchmark, ranked[len(ranked) // 2])
        if problem not in problems:
            raise ValueError(f"pilot problem is absent from bank: {benchmark}/{problem}")
        selected_problems[benchmark] = problem
    selected = [
        i
        for i in schedule
        if i["seed"] == 8 and selected_problems.get(i["benchmark"]) == i["problem_id"]
    ]
    if len(selected) != 48 or len({(i["benchmark"], i["harness_id"]) for i in selected}) != 48:
        raise ValueError("pilot must cover all24arms in both benchmarks at seed8")
    return [i["run_id"] for i in selected], selected_problems


def bootstrap_preprocess_models(context: Any) -> dict[str, Any]:
    """Reuse the prior pinned model manifest, without creating a schedule early."""
    from value_as_tool.assets import model_manifest_path

    prior = Path(context.config.conditioning.source_artifact_root)
    if not prior.is_absolute():
        prior = Path(context.config_path).parent / prior
    source = prior / "prepared/models/manifest.json"
    models = json.loads(source.read_text())
    for role in ("solver", "judge"):
        expected = getattr(context.config.models, role)
        entry = models["models"][role]
        if (entry["name"] != expected.name or entry["revision"] != expected.revision
                or entry.get("tokenizer_only")):
            raise ValueError(f"prior {role} model manifest differs from pinned launch")
        root = Path(entry["path"])
        for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            if not (root / filename).is_file() or not (root / filename).stat().st_size:
                raise FileNotFoundError(root / filename)
        index = root / "model.safetensors.index.json"
        if index.exists():
            weight_index = json.loads(index.read_text())
            shards = set(weight_index["weight_map"].values())
            expected_bytes = int(weight_index.get("metadata", {}).get("total_size", 1))
        else:
            shards, expected_bytes = {"model.safetensors"}, 1
        if not shards or any(Path(name).name != name for name in shards):
            raise ValueError("invalid pinned weight index")
        if any(not (root / name).is_file() or not (root / name).stat().st_size for name in shards):
            raise FileNotFoundError(f"incomplete pinned {role} weights")
        if sum((root / name).stat().st_size for name in shards) < expected_bytes:
            raise ValueError(f"truncated pinned {role} weights")
    destination = model_manifest_path(context.config)
    if destination.exists() and json.loads(destination.read_text()) != models:
        raise ValueError("refusing to replace different experiment model assets")
    if not destination.exists():
        write_json(destination, models)
    return {"source": str(source), "source_sha256": digest(source),
            "path": str(destination), "sha256": digest(destination)}


def freeze_and_prepare(manifest: dict[str, Any]) -> None:
    from prepare_qwen35_large_budget import prepare_local

    from value_as_tool.conditioning import freeze_bank

    frozen = freeze_bank(manifest["bank_root"], expected_bank_sha256=manifest["bank_sha256"])
    payload = yaml.safe_load(Path(manifest["preprocess_config"]).read_text())
    payload["conditioning"]["manifest_sha256"] = frozen["manifest_sha256"]
    path = Path(manifest["control_root"]) / "experiment.frozen.yaml"
    text = yaml.safe_dump(payload, sort_keys=False)
    if path.exists() and path.read_text() != text:
        raise RuntimeError("refusing to replace a different frozen experiment config")
    if not path.exists():
        with path.open("x") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        path.chmod(0o444)
    prepared = prepare_local(str(path), Path(manifest["source_artifact_root"]))
    if prepared["scheduled"] != manifest["expected_runs"]:
        raise RuntimeError("prepared schedule size differs from bank × arms × fresh seeds")
    schedule = json.loads(Path(prepared["schedule_path"]).read_text())
    bank = json.loads((Path(manifest["bank_root"]) / "bank.json").read_text())
    pilot, problems = choose_pilot(schedule["items"], bank, manifest["requested_pilot_problems"])
    manifest.update(
        config=str(path),
        config_sha256=digest(path),
        prepared=prepared,
        config_fingerprint=prepared["config_fingerprint"],
        schedule_fingerprint=prepared["schedule_fingerprint"],
        conditioning_manifest_sha256=frozen["manifest_sha256"],
        pilot_run_ids=pilot,
        pilot_problems=problems,
        pilot_selection="requested problem or median prior Direct generated cost",
    )


def pilot_result_issues(solve: dict[str, Any], judge: dict[str, Any] | None = None) -> list[str]:
    issues = []
    if solve.get("status") not in {"accepted", "completed", "cycle_limit"}:
        issues.append("pilot solve protocol or infrastructure failure")
    if (
        not solve.get("calls")
        or solve.get("error")
        or any(c.get("error") or not c.get("response") for c in solve.get("calls", []))
    ):
        issues.append("pilot solve request error")
    if solve.get("usage_exact") is False:
        issues.append("pilot solve usage is inexact")
    if judge is not None and judge.get("judge_status") != "completed":
        issues.append("pilot judge failed or could not parse")
    return issues


def pilot_gate(manifest: dict[str, Any], *, with_judges: bool) -> dict[str, Any]:
    from audit_budget_run import _restore_runtime_messages, _restore_runtime_tools

    from value_as_tool.pipeline import _model_entry, load_context
    from value_as_tool.tokenization import HuggingFaceTokenCounter

    artifact = Path(manifest["artifact_root"])
    issues = []
    context = load_context(manifest["config"])
    counter = (
        None
        if with_judges
        else HuggingFaceTokenCounter(
            _model_entry(context, "solver")["path"],
            enable_thinking=context.config.sampling.enable_thinking,
        )
    )
    if len(manifest["pilot_run_ids"]) != 48 or len(set(manifest["pilot_run_ids"])) != 48:
        issues.append({"issues": ["pilot does not contain48unique cells"]})
    for run_id in manifest["pilot_run_ids"]:
        solve_path = artifact / "solve" / "runs" / run_id / "result.json"
        judge_path = artifact / "judge" / "runs" / run_id / "result.json"
        if not solve_path.exists() or (with_judges and not judge_path.exists()):
            issues.append({"run_id": run_id, "issues": ["missing pilot result"]})
            continue
        solved = json.loads(solve_path.read_text())
        found = pilot_result_issues(
            solved,
            json.loads(judge_path.read_text()) if with_judges else None,
        )
        if counter is not None:
            for call in solved.get("calls", []):
                prompt_tokens = counter.count_messages(
                    _restore_runtime_messages(call["messages"]),
                    _restore_runtime_tools(call.get("tools")),
                )
                allowance = max(
                    0,
                    context.config.budget.context_tokens
                    - context.config.budget.context_headroom_tokens
                    - prompt_tokens,
                )
                if call["max_tokens"] != allowance:
                    found.append(f"call{call['index']} did not use exact native context allowance")
        if found:
            issues.append({"run_id": run_id, "issues": found})
    stage_names = {"pilot_solve", "pilot_judge"} if with_judges else {"pilot_solve"}
    if any(not j.get("clean_exit") for j in manifest["jobs"].values() if j["stage"] in stage_names):
        issues.append({"issues": ["pilot worker accounting not COMPLETED0:0"]})
    result = {
        "checked_at": now(),
        "pilot_cells": len(manifest["pilot_run_ids"]),
        "judges_checked": with_judges,
        "ready": not issues,
        "issues": issues,
    }
    write_json(
        Path(manifest["control_root"])
        / ("pilot-gate.json" if with_judges else "pilot-solve-gate.json"),
        result,
    )
    return result


def finalize(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest["stage"] != "judge":
        raise RuntimeError("finalization requires the judge stage")
    for stage in STAGES:
        counts = TaskQueue(manifest["queues"][stage]["path"]).counts()
        if counts["failed"] or counts["pending"] or counts["running"]:
            raise RuntimeError(f"cannot finalize unfinished {stage} queue")
    if any(j.get("pool") and not j.get("terminal_confirmed") for j in manifest["jobs"].values()):
        raise RuntimeError("cannot finalize while GPU workers remain active or unobserved")
    from audit_budget_run import audit

    from value_as_tool.pipeline import load_context, report

    context = load_context(manifest["config"])
    report_result = report(context)
    if report_result.get("scheduled_cells") != manifest["expected_runs"]:
        raise RuntimeError("final report does not cover the expected schedule")
    audit_result = audit(manifest["config"], check_context=True)
    write_json(Path(manifest["control_root"]) / "final-budget-audit.json", audit_result)
    if not audit_result["ready"] or audit_result["selected"] != manifest["expected_runs"]:
        raise RuntimeError("final coverage/context audit failed; preserved outcomes require review")
    completion = {
        "completed_at": now(),
        "execution_complete": True,
        "expected_runs": manifest["expected_runs"],
        "config_fingerprint": manifest["config_fingerprint"],
        "schedule_fingerprint": manifest["schedule_fingerprint"],
        "conditioning_manifest_sha256": manifest["conditioning_manifest_sha256"],
        "source_sha256": manifest["source_sha256"],
        "model_statuses": audit_result["solve_statuses"],
        "judge_statuses": audit_result["judge_statuses"],
        "report": str(context.artifact_root / "report/report.json"),
        "accounting": {
            key: j.get("accounting") for key, j in manifest["jobs"].items() if j.get("pool")
        },
        "note": "Execution coverage is distinct from model correctness and protocol outcomes.",
    }
    write_json(Path(manifest["control_root"]) / "completion.json", completion)
    return completion


def tick(path: Path, manifest: dict[str, Any], *, allow_submit: bool = True) -> bool:
    """One controller transition/admission; False means the controller has stopped."""
    verify_source(manifest)
    observe_owned_jobs(manifest)
    stage = manifest["stage"]
    queue = TaskQueue(manifest["queues"][stage]["path"])
    recovered = queue.recover_abandoned()
    if recovered:
        manifest.setdefault("claim_recoveries", []).append(
            {"at": now(), "stage": stage, "tasks": recovered}
        )
    counts = queue.counts()
    manifest["stage_counts"] = counts
    manifest["updated_at"] = now()
    if counts["failed"]:
        manifest.update(state="blocked", blocked_reason=f"{stage} has failed durable tasks")
        write_json(path, manifest)
        return False
    if not counts["pending"] + counts["running"]:
        active = [
            j
            for j in manifest["jobs"].values()
            if j["stage"] == stage and not j.get("terminal_confirmed")
        ]
        if active:
            write_json(path, manifest)
            return True
        if stage == "preprocess":
            freeze_and_prepare(manifest)
            initialize_stage(manifest, "pilot_solve")
        elif stage in {"pilot_solve", "pilot_judge"}:
            gate = pilot_gate(manifest, with_judges=stage == "pilot_judge")
            if not gate["ready"]:
                manifest.update(
                    state="blocked", blocked_reason="pilot validation failed", pilot_gate=gate
                )
                write_json(path, manifest)
                return False
            initialize_stage(manifest, "pilot_judge" if stage == "pilot_solve" else "solve")
        elif stage == "solve":
            initialize_stage(manifest, "judge")
        else:
            manifest.update(state="complete", completion=finalize(manifest))
            write_json(path, manifest)
            return False
        write_json(path, manifest)
        return True
    rows = query_user_jobs()
    desired = min(sum(LIMITS.values()), math.ceil((counts["ready"] + counts["running"]) / 2))
    plan = admission_plan(rows, manifest["jobs"], desired)
    manifest["admission"] = {"at": now(), **plan}
    write_json(path, manifest)
    if allow_submit:
        for pool in LIMITS:
            # Recheck after each submission; reserve submitted-but-not-yet-visible arrays.
            plan = admission_plan(query_user_jobs(), manifest["jobs"], desired)
            workers = plan["add_workers"][pool]
            if workers:
                serial = sum(j.get("stage") == stage for j in manifest["jobs"].values())
                key = f"{stage}-{pool}-{serial:04d}"
                submit_job(
                    path,
                    manifest,
                    key,
                    worker_command(manifest, stage, pool, workers, key),
                    stage=stage,
                    pool=pool,
                    workers=workers,
                )
    return True


def prepare_launch(args: argparse.Namespace) -> Path:
    from value_as_tool.harnesses import ATTEMPT_CONDITIONED_HARNESSES
    from value_as_tool.pipeline import load_context

    repo = Path(__file__).resolve().parents[1]
    context = load_context(args.config, environment={})
    config = context.config
    if (
        config.models.solver.name != "Qwen/Qwen3.5-9B"
        or config.models.solver.revision != "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        or list(config.evaluation.seeds) != list(range(8, 16))
        or set(config.evaluation.harnesses or ()) != set(ATTEMPT_CONDITIONED_HARNESSES)
        or config.runtime.max_concurrency != 2
        or config.runtime.solver_tensor_parallel_size != 1
        or config.runtime.solver_backend != "sglang"
        or not config.sampling.enable_thinking
        or config.sampling.thinking_content_reserve_tokens != 0
        or config.budget.context_tokens != 262144
        or config.budget.context_headroom_tokens != 1024
        or config.budget.generated_tokens != 8388608
    ):
        raise ValueError("configuration differs from the 24-arm/pinned-9B/fresh-8-seed profile")
    if config.conditioning is None or not config.conditioning.bank_sha256:
        raise ValueError("build and pin the attempt bank before preparing launch")
    control = args.control_root.resolve()
    path = control / "submission.json"
    if path.exists():
        raise FileExistsError("manifest exists; use --manifest to review or submit it")
    control.mkdir(parents=True, exist_ok=True)
    (control / "logs").mkdir(exist_ok=True)
    source = args.source.resolve() if args.source else control / "source"
    if not args.source:
        source.mkdir()
        for name in COPY_ITEMS:
            current = repo / name
            if current.is_dir():
                shutil.copytree(
                    current,
                    source / name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".env", ".env.*"),
                )
            else:
                shutil.copy2(current, source / name)
        shutil.copy2(args.config, source / "experiment.source.yaml")
        for current in sorted(source.rglob("*"), reverse=True):
            current.chmod(current.stat().st_mode & ~0o222)
        source.chmod(source.stat().st_mode & ~0o222)
    files = source_files(source)
    preprocess = control / "experiment.preprocess.yaml"
    # A copied relative path would resolve against the control directory on
    # workers. Pin all resolved runtime paths before creating the snapshot;
    # the frozen evaluation config inherits these same paths.
    preprocess.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    preprocess.chmod(0o444)
    bank_root = Path(config.conditioning.bank_root)
    bank = json.loads((bank_root / "bank.json").read_text())
    if digest(bank_root / "bank.json") != config.conditioning.bank_sha256:
        raise ValueError("attempt bank does not match pinned config digest")
    if bank["source_seeds"] != list(range(8)) or set(bank["problems"]) != set(
        config.evaluation.benchmarks
    ):
        raise ValueError("attempt bank seed/benchmark coverage differs from the approved launch")
    problem_count = sum(len(problems) for problems in bank["problems"].values())
    expected = problem_count * len(config.evaluation.harnesses) * len(config.evaluation.seeds)
    requested = dict(value.split("=", 1) for value in args.pilot_problem)
    exports = {
        "VALUE_AS_TOOL_REPO_ROOT": str(source),
        "VALUE_AS_TOOL_ENV_FILE": str(repo / ".env"),
        "VALUE_AS_TOOL_PYTHON": str(repo / ".venv/bin/python"),
        "VALUE_AS_TOOL_ARTIFACT_ROOT": str(context.artifact_root),
        "VALUE_AS_TOOL_ASSET_ROOT": str(context.asset_root),
        "VALUE_AS_TOOL_SGLANG_BIN": str(repo / ".venv-sglang/bin/sglang"),
        "VALUE_AS_TOOL_VLLM_BIN": "/storage/home/anikaitsingh/.conda/envs/myenv/bin/vllm",
        "TIKTOKEN_ENCODINGS_BASE": str(context.asset_root / "tiktoken"),
    }
    for key in exports:
        if os.environ.get(key) and key not in {
            "VALUE_AS_TOOL_REPO_ROOT",
            "VALUE_AS_TOOL_ARTIFACT_ROOT",
            "VALUE_AS_TOOL_ASSET_ROOT",
        }:
            exports[key] = os.environ[key]
    manifest = {
        "schema_version": 1,
        "created_at": now(),
        "state": "prepared",
        "stage": "preprocess",
        "manifest": str(path),
        "control_root": str(control),
        "source": str(source),
        "source_files": files,
        "source_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "preprocess_config": str(preprocess),
        "preprocess_config_sha256": digest(preprocess),
        "artifact_root": str(context.artifact_root),
        "source_artifact_root": config.conditioning.source_artifact_root,
        "bank_root": str(bank_root),
        "bank_sha256": config.conditioning.bank_sha256,
        "expected_runs": expected,
        "problems": problem_count,
        "arms": 24,
        "fresh_seeds": list(range(8, 16)),
        "pilot_cells": 48,
        "requested_pilot_problems": requested,
        "gpu_limits": LIMITS,
        "segment": 1,
        "request_concurrency": 2,
        "tensor_parallel_size": 1,
        "exports": exports,
        "jobs": {},
        "queues": {},
        "credentials": "not copied; known credentials read from external.env at runtime",
    }
    manifest["preprocess_models"] = bootstrap_preprocess_models(context)
    initialize_stage(manifest, "preprocess")
    verify_source(manifest)
    write_json(path, manifest)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("launch", "supervise", "status"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--control-root", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--pilot-problem", action="append", default=[], metavar="BENCHMARK=PROBLEM")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--submit", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.command != "launch" and (args.dry_run or args.submit):
        parser.error("--dry-run and --submit are supported only for launch")
    if args.manifest is None:
        if args.command != "launch" or args.config is None or args.control_root is None:
            parser.error("supply --manifest, or launch with --config and --control-root")
        args.manifest = prepare_launch(args)
    path = args.manifest.resolve()
    if args.command == "status":
        manifest = json.loads(path.read_text())
        print(
            json.dumps(
                {
                    "state": manifest["state"],
                    "stage": manifest["stage"],
                    "queue": TaskQueue(manifest["queues"][manifest["stage"]]["path"]).counts(),
                }
            )
        )
        return
    with manifest_lock(path):
        manifest = json.loads(path.read_text())
        verify_source(manifest)
        if args.command == "launch":
            if not args.submit:
                counts = TaskQueue(manifest["queues"][manifest["stage"]]["path"]).counts()
                desired = min(276, math.ceil((counts["ready"] + counts["running"]) / 2))
                plan = admission_plan(query_user_jobs(), manifest["jobs"], desired)
                print(
                    json.dumps(
                        {
                            "manifest": str(path),
                            "dry_run": True,
                            "expected_runs": manifest["expected_runs"],
                            "admission": plan,
                            "controller_command": controller_command(manifest),
                            "current_stage_worker_commands": [
                                worker_command(
                                    manifest, manifest["stage"], pool, count, f"preview-{pool}"
                                )
                                for pool, count in plan["add_workers"].items()
                                if count
                            ],
                        },
                        indent=2,
                    )
                )
                return
            if manifest["state"] not in {"prepared", "running"}:
                raise RuntimeError("blocked/completed manifests cannot be launched automatically")
            submit_job(
                path, manifest, "controller", controller_command(manifest), stage="controller"
            )
            manifest["state"] = "running"
            write_json(path, manifest)
            return
        if manifest["state"] != "running":
            raise RuntimeError("supervisor requires an explicitly submitted running launch")
        os.environ.update(manifest["exports"])
        while True:
            try:
                if not tick(path, manifest):
                    break
            except Exception as exc:
                manifest.update(
                    state="blocked", blocked_at=now(), blocked_reason=f"{type(exc).__name__}: {exc}"
                )
                write_json(path, manifest)
                raise
            time.sleep(max(1, args.poll_seconds))


if __name__ == "__main__":
    main()
