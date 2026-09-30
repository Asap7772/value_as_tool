"""Submit and supervise the pinned Qwen3.5-9B large-budget experiment.

Launch prepares an immutable source snapshot and three pilot shards. A CPU
controller checks their accounting and budget audit before submitting the full
solve/judge graph; a final CPU job writes reports and an audited completion.
Submission intents precede sbatch, so an interrupted, ambiguous submission is
never silently dispatched a second time.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
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

SHARDS = 384
EXPECTED_RUNS = 14_760
PILOT = ((0, "high"), (272, "shared"), (280, "shared"))
POOLS = {
    "high": ("g3_scientific-reasoning_high", "0-255%128"),
    "shared": ("g3_core_shared", "256-383%64"),
}
COPY_ITEMS = ("src", "scripts", "prompts", "tests", "pyproject.toml", "README.md", "uv.lock")


def now() -> str:
    return datetime.now(UTC).isoformat()


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
    with path.with_suffix(".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def execution_environment(manifest: dict[str, Any]) -> dict[str, str]:
    environment = {**os.environ, **manifest["exports"]}
    environment["PYTHONPATH"] = str(Path(manifest["source"]) / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def record_command(
    path: Path, manifest: dict[str, Any], label: str, command: list[str]
) -> subprocess.CompletedProcess[str]:
    entry = {"label": label, "command": command, "started_at": now()}
    manifest.setdefault("commands", []).append(entry)
    write_json(path, manifest)
    result = subprocess.run(
        command, cwd=manifest["source"], env=execution_environment(manifest),
        capture_output=True, text=True, check=False,
    )
    logs = Path(manifest["run_root"]) / "logs"
    (logs / f"{label}.stdout").write_text(result.stdout)
    (logs / f"{label}.stderr").write_text(result.stderr)
    entry.update(completed_at=now(), returncode=result.returncode)
    write_json(path, manifest)
    return result


def base_sbatch(manifest: dict[str, Any], key: str, qos: str) -> list[str]:
    exports = manifest["exports"]
    if any("," in key or "," in value or "\n" in value for key, value in exports.items()):
        raise ValueError("Slurm export paths must not contain commas or newlines")
    logs = Path(manifest["run_root"]) / "logs"
    return [
        "sbatch", "--parsable", "--partition=g3", "--account=scientific-reasoning",
        "--segment=1", f"--qos={qos}", f"--chdir={manifest['source']}",
        "--export=" + ",".join(f"{name}={value}" for name, value in exports.items()),
        f"--job-name=q35-{key}", f"--output={logs / (key + '-%A_%a.out')}",
        f"--error={logs / (key + '-%A_%a.err')}",
    ]


def gpu_command(
    manifest: dict[str, Any], key: str, stage: str, pool: str, array: str,
    dependencies: tuple[str, ...] = (),
) -> list[str]:
    command = base_sbatch(manifest, key, POOLS[pool][0])
    command += ["--gpus-per-node=1", "--cpus-per-task=24", "--mem=96G", "--time=7-00:00:00"]
    command.append(f"--array={array}")
    if dependencies:
        command.append("--dependency=afterany:" + ":".join(dependencies))
    return command + [
        str(Path(manifest["source"]) / "scripts/slurm/run_gpu.sbatch"),
        manifest["config"], stage, "--shard-count", str(SHARDS),
    ]


def controller_command(
    manifest: dict[str, Any], mode: str, dependencies: tuple[str, ...]
) -> list[str]:
    return base_sbatch(manifest, mode, POOLS["high"][0]) + [
        "--cpus-per-task=8", "--mem=32G", "--time=1-00:00:00",
        "--dependency=afterany:" + ":".join(dependencies),
        str(Path(manifest["source"]) / "scripts/slurm/run_qwen35_controller.sbatch"),
        manifest["config"], mode, manifest["manifest"],
    ]


def submit_job(
    path: Path, manifest: dict[str, Any], key: str, command: list[str],
    *, stage: str, array: str | None = None,
) -> str:
    jobs = manifest.setdefault("jobs", {})
    if key in jobs:
        previous = jobs[key]
        if previous["command"] != command:
            raise RuntimeError(f"submission command changed for {key}")
        if previous.get("job_id"):
            return previous["job_id"]
        raise RuntimeError(f"ambiguous prior submission for {key}; inspect Slurm before recovery")
    entry = {"stage": stage, "array": array, "command": command, "intent_at": now()}
    jobs[key] = entry
    write_json(path, manifest)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    entry.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    job_id = result.stdout.strip().split(";", 1)[0]
    if result.returncode or not re.fullmatch(r"[0-9]+", job_id):
        entry["state"] = "submission_uncertain"
        write_json(path, manifest)
        raise RuntimeError(f"sbatch did not return a confirmed job ID for {key}")
    entry.update(job_id=job_id, state="submitted", submitted_at=now())
    write_json(path, manifest)
    return job_id


def array_indices(expression: str) -> list[int]:
    bounds = expression.split("%", 1)[0].split("-", 1)
    return list(range(int(bounds[0]), int(bounds[-1]) + 1))


def accounting(jobs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected = {
        f"{job['job_id']}_{index}"
        for job in jobs.values()
        for index in array_indices(job["array"])
    }
    command = [
        "sacct", "-X", "--noheader", "--parsable2",
        "--jobs=" + ",".join(job["job_id"] for job in jobs.values()),
        "--format=JobID%64,State%32,ExitCode",
    ]
    records = {}
    for attempt in range(12):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(f"sacct failed: {result.stderr.strip()}")
        records = {}
        for line in result.stdout.splitlines():
            fields = line.strip().split("|")
            if len(fields) >= 3 and fields[0] in expected:
                records[fields[0]] = {"state": fields[1], "exit_code": fields[2]}
        if expected <= records.keys():
            break
        if attempt < 11:
            time.sleep(5)
    missing = sorted(expected - records.keys())
    failed = {
        key: value for key, value in records.items()
        if value != {"state": "COMPLETED", "exit_code": "0:0"}
    }
    return {"command": command, "ready": not missing and not failed,
            "expected": len(expected), "records": records, "missing": missing, "failed": failed}


def audit_command(manifest: dict[str, Any], *, pilot: bool) -> list[str]:
    output = Path(manifest["run_root"]) / ("pilot-audit.json" if pilot else "full-audit.json")
    command = [
        manifest["exports"]["VALUE_AS_TOOL_PYTHON"],
        str(Path(manifest["source"]) / "scripts/audit_budget_run.py"),
        "--config", manifest["config"], "--shard-count", str(SHARDS),
        "--check-context", "--require-ready", "--output", str(output),
    ]
    if pilot:
        for shard, _ in PILOT:
            command += ["--shard-index", str(shard)]
    return command


def advance(path: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("state") in {"full_submitted", "complete"}:
        return
    pilot_jobs = {key: job for key, job in manifest["jobs"].items() if key.startswith("pilot-")}
    if len(pilot_jobs) != 6 or any(not job.get("job_id") for job in pilot_jobs.values()):
        raise RuntimeError("all six pilot submissions must have confirmed job IDs")
    manifest["pilot_accounting"] = accounting(pilot_jobs)
    write_json(path, manifest)
    audit = record_command(path, manifest, "pilot-audit", audit_command(manifest, pilot=True))
    if not manifest["pilot_accounting"]["ready"] or audit.returncode:
        manifest.update(state="pilot_failed", stopped_at=now())
        write_json(path, manifest)
        raise RuntimeError("pilot accounting or budget audit failed; full run was not launched")
    manifest.update(state="pilot_validated", pilot_validated_at=now())
    write_json(path, manifest)
    solve_ids = []
    for pool, (_, array) in POOLS.items():
        key = f"solve-{pool}"
        solve_ids.append(submit_job(
            path, manifest, key, gpu_command(manifest, key, "solve", pool, array),
            stage="solve", array=array,
        ))
    judge_ids = []
    for pool, (_, array) in POOLS.items():
        key = f"judge-{pool}"
        judge_ids.append(submit_job(
            path, manifest, key,
            gpu_command(manifest, key, "judge", pool, array, tuple(solve_ids)),
            stage="judge", array=array,
        ))
    submit_job(
        path, manifest, "finalize", controller_command(manifest, "finalize", tuple(judge_ids)),
        stage="controller",
    )
    manifest.update(state="full_submitted", full_submitted_at=now())
    write_json(path, manifest)


def finalize(path: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("state") == "complete":
        return
    report = record_command(path, manifest, "report", [
        manifest["exports"]["VALUE_AS_TOOL_PYTHON"], "-m", "value_as_tool",
        "--config", manifest["config"], "report",
    ])
    audit = record_command(path, manifest, "full-audit", audit_command(manifest, pilot=False))
    full_jobs = {key: job for key, job in manifest["jobs"].items()
                 if key in {"solve-high", "solve-shared", "judge-high", "judge-shared"}}
    if len(full_jobs) != 4:
        raise RuntimeError("all four full-run arrays must be recorded before finalization")
    completed = accounting(full_jobs)
    complete = report.returncode == 0 and audit.returncode == 0 and completed["ready"]
    completion = {
        "completed_at": now(), "complete": complete, "scheduled": EXPECTED_RUNS,
        "config_fingerprint": manifest["config_fingerprint"],
        "schedule_fingerprint": manifest["schedule_fingerprint"],
        "report_returncode": report.returncode, "audit_returncode": audit.returncode,
        "slurm_accounting": completed,
        "report": str(Path(manifest["artifact_root"]) / "report/report.md"),
        "audit": str(Path(manifest["run_root"]) / "full-audit.json"),
    }
    write_json(Path(manifest["run_root"]) / "completion.json", completion)
    manifest.update(state="complete" if complete else "incomplete", completion=completion)
    write_json(path, manifest)
    if not complete:
        raise RuntimeError("collection checks remain incomplete; see completion.json")


def snapshot_source(repo: Path, config: Path, destination: Path) -> dict[str, str]:
    destination.mkdir()
    for name in COPY_ITEMS:
        source = repo / name
        if source.is_dir():
            shutil.copytree(source, destination / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".env", ".env.*"))
        else:
            shutil.copy2(source, destination / name)
    shutil.copy2(config, destination / "experiment.source.yaml")
    digests = {
        str(file.relative_to(destination)): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(destination.rglob("*")) if file.is_file()
    }
    for file in sorted(destination.rglob("*"), reverse=True):
        file.chmod(file.stat().st_mode & ~0o222)
    destination.chmod(destination.stat().st_mode & ~0o222)
    return digests


def launch(args: argparse.Namespace) -> Path:
    from value_as_tool.pipeline import load_context

    repo = Path(__file__).absolute().parents[1]
    config = args.config.absolute()
    context = load_context(config, environment={})
    settings = context.config
    if (settings.models.solver.name != "Qwen/Qwen3.5-9B"
            or settings.models.solver.revision != "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
            or tuple(settings.evaluation.seeds) != tuple(range(8))
            or len(settings.evaluation.harnesses or ()) != 9
            or settings.runtime.solve_shards != SHARDS or settings.runtime.judge_shards != SHARDS
            or settings.runtime.solver_tensor_parallel_size != 1
            or settings.runtime.solver_backend != "sglang" or settings.runtime.max_concurrency != 2
            or settings.models.judge.name != "openai/gpt-oss-20b"
            or settings.models.judge.revision != "6cee5e81ee83917806bbde320786a8fb61efebee"
            or settings.budget.generated_tokens != 8_388_608
            or settings.budget.context_tokens != 262_144
            or settings.budget.cch_stage_tokens != 262_144
            or not settings.sampling.enable_thinking
            or settings.sampling.thinking_content_reserve_tokens != 0):
        raise ValueError("configuration differs from the agreed pinned 9B/eight-sample run")
    run_root = args.run_root.absolute()
    artifact_root = context.artifact_root
    for root in (run_root, artifact_root):
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"refusing nonempty run directory: {root}")
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir()
    path = run_root / "submission.json"
    with manifest_lock(path):
        source = run_root / "source"
        digests = snapshot_source(repo, config, source)
        write_json(run_root / "snapshot-files.json", digests)
        exports = {
            "VALUE_AS_TOOL_REPO_ROOT": str(source),
            "VALUE_AS_TOOL_ENV_FILE": str(repo / ".env"),
            "VALUE_AS_TOOL_PYTHON": str(repo / ".venv/bin/python"),
            "VALUE_AS_TOOL_ARTIFACT_ROOT": str(artifact_root),
            "VALUE_AS_TOOL_ASSET_ROOT": str(context.asset_root),
            "VALUE_AS_TOOL_SGLANG_BIN": str(repo / ".venv-sglang/bin/sglang"),
            "VALUE_AS_TOOL_VLLM_BIN": "/storage/home/anikaitsingh/.conda/envs/myenv/bin/vllm",
            "TIKTOKEN_ENCODINGS_BASE": str(context.asset_root / "tiktoken"),
        }
        for name in ("VALUE_AS_TOOL_PYTHON", "VALUE_AS_TOOL_SGLANG_BIN", "VALUE_AS_TOOL_VLLM_BIN"):
            if not os.access(exports[name], os.X_OK):
                raise FileNotFoundError(f"missing executable: {exports[name]}")
        manifest = {
            "schema_version": 1, "created_at": now(), "state": "snapshot_created",
            "run_root": str(run_root), "manifest": str(path), "source": str(source),
            "config": str(source / "experiment.source.yaml"), "artifact_root": str(artifact_root),
            "exports": exports, "segment": 1, "shards": SHARDS, "scheduled": EXPECTED_RUNS,
            "max_concurrent_gpus": {"high": 128, "shared": 64, "combined": 192},
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
            '--config "$EXPERIMENT_CONFIG" --source-artifact-root "$2"',
            "qwen35-prepare", manifest["config"], str(repo / "artifacts/default"),
        ])
        if prepare.returncode:
            raise RuntimeError("offline preparation failed; see logs/prepare.stderr")
        prepared = json.loads(prepare.stdout)
        if prepared["scheduled"] != EXPECTED_RUNS:
            raise ValueError(f"unexpected schedule size: {prepared['scheduled']}")
        manifest.update(config_fingerprint=prepared["config_fingerprint"],
                        schedule_fingerprint=prepared["schedule_fingerprint"], prepared=prepared)
        write_json(path, manifest)
        judge_ids = []
        for shard, pool in PILOT:
            solve_key = f"pilot-solve-{shard}"
            solve_id = submit_job(
                path, manifest, solve_key,
                gpu_command(manifest, solve_key, "solve", pool, str(shard)),
                stage="solve", array=str(shard),
            )
            judge_key = f"pilot-judge-{shard}"
            judge_ids.append(submit_job(
                path, manifest, judge_key,
                gpu_command(manifest, judge_key, "judge", pool, str(shard), (solve_id,)),
                stage="judge", array=str(shard),
            ))
        submit_job(
            path, manifest, "advance", controller_command(manifest, "advance", tuple(judge_ids)),
            stage="controller",
        )
        manifest.update(state="pilot_submitted", pilot_submitted_at=now())
        write_json(path, manifest)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    launcher = commands.add_parser("launch")
    launcher.add_argument("--config", type=Path,
                          default=Path("experiment_qwen35_9b_harness_large_budget.yaml"))
    launcher.add_argument("--run-root", type=Path, required=True)
    for mode in ("advance", "finalize"):
        commands.add_parser(mode).add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.mode == "launch":
        print(launch(args))
    else:
        with manifest_lock(args.manifest):
            manifest = json.loads(args.manifest.read_text())
            globals()[args.mode](args.manifest, manifest)


if __name__ == "__main__":
    main()
