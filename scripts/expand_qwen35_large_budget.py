"""Expand a running pilot without duplicating its active shards.

Only the submission manifest and Slurm graph change. Workers and finalization
continue to use the original immutable source and experiment configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any


def plan_expansion(
    manifest: dict[str, Any], high_cap: int, shared_cap: int,
) -> dict[str, Any]:
    if not 2 <= high_cap <= 256 or not 3 <= shared_cap <= 128:
        raise ValueError("caps must reserve one high and two shared pilot GPUs")
    jobs = manifest["jobs"]
    for shard in (0, 272, 280):
        for stage in ("solve", "judge"):
            job = jobs[f"pilot-{stage}-{shard}"]
            if not job.get("job_id") or job["array"] != str(shard):
                raise ValueError("all pilot submissions must have confirmed IDs and shards")
    return {
        "solve_arrays": {
            "high": f"0-255%{high_cap - 1}",
            "shared": f"256-383%{shared_cap - 2}",
        },
        "judge_arrays": {
            "high": f"0-255%{high_cap}",
            "shared": f"256-383%{shared_cap}",
        },
        "pilot_dependencies": {
            "high": {"0": jobs["pilot-solve-0"]["job_id"]},
            "shared": {
                str(shard): jobs[f"pilot-solve-{shard}"]["job_id"]
                for shard in (272, 280)
            },
        },
        "judge_pilot_dependencies": [
            jobs[f"pilot-judge-{shard}"]["job_id"] for shard in (0, 272, 280)
        ],
        "max_concurrent_gpus": {
            "high": high_cap, "shared": shared_cap, "combined": high_cap + shared_cap,
        },
    }


def load_launcher(manifest: dict[str, Any]):
    source = Path(manifest["source"])
    hashes = json.loads((Path(manifest["run_root"]) / "snapshot-files.json").read_text())
    for name, expected in hashes.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"immutable source changed: {name}")
    spec = importlib.util.spec_from_file_location(
        "pinned_qwen35_launcher", source / "scripts/submit_qwen35_large_budget.py",
    )
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    return launcher


def expand(path: Path, manifest: dict[str, Any], launcher, plan: dict[str, Any]) -> None:
    expansion = manifest.setdefault("immediate_expansion", {})
    if expansion.get("complete"):
        if expansion["plan"] != plan:
            raise RuntimeError("this expansion was already completed with a different plan")
        return
    if manifest["state"] not in {"pilot_submitted", "expanding", "full_submitted"}:
        raise RuntimeError(f"cannot expand state {manifest['state']}")
    if expansion.get("plan", plan) != plan:
        raise RuntimeError("cannot change a partially submitted expansion")
    expansion.update(
        plan=plan,
        reason="User requested immediate parallel collection with the recorded GPU caps",
        pilot_gate_overridden=True,
        source_unchanged=True,
        controller=str(Path(__file__).resolve()),
        controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    operations = expansion.setdefault("operations", {})

    def operation(label: str, command: list[str]) -> None:
        if operations.get(label, {}).get("returncode") == 0:
            return
        result = launcher.record_command(path, manifest, label, command)
        operations[label] = {"command": command, "returncode": result.returncode}
        launcher.write_json(path, manifest)
        if result.returncode:
            raise RuntimeError(f"{label} failed: {result.stderr.strip()}")

    advance_id = manifest["jobs"]["advance"]["job_id"]
    operation("expansion-hold-advance", ["scontrol", "hold", advance_id])
    manifest.update(state="expanding", max_concurrent_gpus=plan["max_concurrent_gpus"])
    launcher.write_json(path, manifest)

    solve_ids = []
    for pool, array in plan["solve_arrays"].items():
        key = f"solve-{pool}"
        command = launcher.gpu_command(manifest, key, "solve", pool, array)
        command.insert(1, "--hold")
        job_id = launcher.submit_job(
            path, manifest, key, command, stage="solve", array=array,
        )
        solve_ids.append(job_id)
        for shard, pilot_id in plan["pilot_dependencies"][pool].items():
            label = f"expansion-dependency-{pool}-{shard}"
            operation(label, [
                "scontrol", "update", f"JobId={job_id}_{shard}",
                f"Dependency=afterany:{pilot_id}",
            ])
            # Confirm the split array element is still held and has its own
            # dependency before any part of the array can start.
            if not operations.get(f"expansion-release-{pool}"):
                observed = subprocess.run(
                    ["scontrol", "show", "job", f"{job_id}_{shard}", "--oneliner"],
                    capture_output=True, text=True, check=True,
                ).stdout
                if (f"Dependency=afterany:{pilot_id}" not in observed
                        or "JobState=PENDING" not in observed or "Priority=0 " not in observed):
                    raise RuntimeError(f"held dependency was not verified for {job_id}_{shard}")
                expansion.setdefault("dependency_verification", {})[f"{job_id}_{shard}"] = observed
                launcher.write_json(path, manifest)

    judge_ids = []
    dependencies = tuple(solve_ids + plan["judge_pilot_dependencies"])
    for pool, array in plan["judge_arrays"].items():
        key = f"judge-{pool}"
        judge_ids.append(launcher.submit_job(
            path, manifest, key,
            launcher.gpu_command(manifest, key, "judge", pool, array, dependencies),
            stage="judge", array=array,
        ))
    launcher.submit_job(
        path, manifest, "finalize",
        launcher.controller_command(manifest, "finalize", tuple(judge_ids)),
        stage="controller",
    )
    # The original advance controller now becomes a harmless no-op. The full
    # immutable finalizer still requires every canonical array cell and artifact.
    manifest.update(state="full_submitted", full_submitted_at=launcher.now())
    launcher.write_json(path, manifest)
    for pool, job_id in zip(plan["solve_arrays"], solve_ids, strict=True):
        operation(f"expansion-release-{pool}", ["scontrol", "release", job_id])
    operation("expansion-release-advance", ["scontrol", "release", advance_id])
    expansion.update(complete=True, completed_at=launcher.now())
    launcher.write_json(path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--high-cap", type=int, default=212)
    parser.add_argument("--shared-cap", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    launcher = load_launcher(manifest)
    with launcher.manifest_lock(args.manifest):
        manifest = json.loads(args.manifest.read_text())
        plan = plan_expansion(manifest, args.high_cap, args.shared_cap)
        if not args.dry_run:
            expand(args.manifest, manifest, launcher, plan)
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
