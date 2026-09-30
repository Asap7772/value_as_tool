"""Observe an attempt-conditioning launch without modifying its control state.

The observer reads Slurm and SQLite, retaining progress samples separately from
the immutable source and controller manifest. Rates use completed tasks, not GPU
allocations or claimed/in-flight requests. Incomplete full-solve stages expose a
raw work projection only: problem order and varying costs prevent a reliable ETA.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from attempt_conditioning_resources import query_user_jobs


def queue_progress(path: str, now: datetime, *, stage: str | None = None) -> dict[str, Any]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30) as db:
        db.row_factory = sqlite3.Row
        rows = list(
            db.execute("SELECT task_id,status,payload,started_at,completed_at,result FROM tasks")
        )
    counts = Counter(row["status"] for row in rows)
    completed = [
        datetime.fromisoformat(row["completed_at"]) for row in rows if row["status"] == "complete"
    ]
    rates = {
        str(minutes): sum(t >= now - timedelta(minutes=minutes) for t in completed) / minutes
        for minutes in (5, 15)
    }
    remaining = counts["pending"] + counts["running"]
    breakdown: dict[str, Counter[str]] = {}
    for row in rows:
        payload = json.loads(row["payload"])
        group = "/".join(str(payload[k]) for k in ("kind", "mode", "method") if payload.get(k))
        breakdown.setdefault(group or "trajectories", Counter())[row["status"]] += 1
    active = sorted(
        (
            {
                "task_id": row["task_id"],
                "age_minutes": (now - datetime.fromisoformat(row["started_at"])).total_seconds()
                / 60,
            }
            for row in rows
            if row["status"] == "running" and row["started_at"]
        ),
        key=lambda row: row["age_minutes"],
        reverse=True,
    )
    projection = remaining / rates["5"] if rates["5"] else None
    long_tail = bool(
        active and projection is not None and active[0]["age_minutes"] > max(10, 2 * projection)
    )
    caveat = (
        "Failed tasks require recovery."
        if counts["failed"]
        else (
            "Problem costs vary and solve queue order is not representative of remaining work; "
            "the raw completion-rate projection is not an ETA."
        )
        if stage == "solve" and remaining
        else "Long active requests make the bulk completion-rate projection unreliable."
        if long_tail
        else "No recent completed tasks to measure throughput."
        if projection is None and remaining
        else None
    )
    return {
        "total": len(rows),
        "counts": {key: counts[key] for key in ("complete", "running", "pending", "failed")},
        "completed_percent": 100 * counts["complete"] / len(rows) if rows else 100,
        "completed_per_minute": rates,
        "eta_minutes": (0 if not remaining else projection) if caveat is None else None,
        "bulk_work_projection_minutes": projection,
        "eta_caveat": caveat,
        "eta_basis": "remaining tasks / last 5 minutes completion rate; stage only",
        "first_completed_at": min(completed).isoformat() if completed else None,
        "last_completed_at": max(completed).isoformat() if completed else None,
        "breakdown": {key: dict(value) for key, value in sorted(breakdown.items())},
        "outcomes": dict(
            Counter(
                json.loads(row["result"] or "{}").get("outcome", "error")
                for row in rows
                if row["status"] in {"complete", "failed"}
            )
        ),
        "oldest_active": active[:5],
        "failed": [
            {"task_id": row["task_id"], "result": json.loads(row["result"] or "{}")}
            for row in rows
            if row["status"] == "failed"
        ][:10],
    }


def observe(manifest_path: Path, *, decode: bool = False) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    now = datetime.now(UTC)
    live = query_user_jobs()
    owned = {job["job_id"]: job for job in manifest["jobs"].values() if job.get("job_id")}
    live = [row for row in live if row["parent"] in owned]
    controllers = [job["job_id"] for job in owned.values() if job["stage"] == "controller"]
    controller = [row for row in live if row["parent"] in controllers]
    accounting = None
    if controllers and not controller:
        result = subprocess.run(
            ["sacct", "-n", "-P", "-j", ",".join(controllers), "--format=JobIDRaw,State,ExitCode"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        accounting = result.stdout.strip().splitlines()
    queues = {
        stage: queue_progress(entry["path"], now, stage=stage)
        for stage, entry in manifest["queues"].items()
    }
    snapshot = {
        "observed_at": now.isoformat(),
        "manifest": str(manifest_path.resolve()),
        "state": manifest["state"],
        "stage": manifest["stage"],
        "controller_updated_at": manifest.get("updated_at"),
        "blocked_reason": manifest.get("blocked_reason"),
        "controller_live": controller,
        "controller_accounting": accounting,
        "gpu_states": {
            state: sum(row["gpus"] for row in live if state in row["states"])
            for state in sorted({state for row in live for state in row["states"]})
        },
        "queues": queues,
        "expected_evaluation_trajectories": manifest["expected_runs"],
        "completion": manifest.get("completion"),
    }
    if decode:
        from sample_attempt_decode import sample

        try:
            diagnostics = sample(manifest_path)
            if diagnostics["stage"] != snapshot["stage"]:
                raise ValueError("stage changed during decode observation")
            snapshot["decode"] = diagnostics
        except (
            OSError, ValueError, RuntimeError, sqlite3.Error, subprocess.SubprocessError
        ) as error:
            snapshot["decode_observation_error"] = str(error)
    return snapshot


def persist(root: Path, snapshot: dict[str, Any]) -> None:
    root.mkdir(exist_ok=True)
    temporary = root / "latest.json.tmp"
    temporary.write_text(json.dumps(snapshot, indent=2) + "\n")
    temporary.replace(root / "latest.json")
    with (root / "history.jsonl").open("a") as stream:
        stream.write(json.dumps(snapshot) + "\n")
    stage = snapshot["stage"]
    progress = snapshot["queues"][stage]
    eta = progress["eta_minutes"]
    summary = {
        "utc": snapshot["observed_at"],
        "state": snapshot["state"],
        "stage": stage,
        "counts": progress["counts"],
        "total": progress["total"],
        "tasks_per_minute_5m": round(progress["completed_per_minute"]["5"], 2),
        "stage_eta_minutes": round(eta, 1) if eta is not None else None,
        "eta_caveat": progress["eta_caveat"],
        "gpu_states": snapshot["gpu_states"],
        "controller_live": bool(snapshot["controller_live"]),
        "blocked_reason": snapshot["blocked_reason"],
    }
    if "decode" in snapshot:
        diagnostics = snapshot["decode"]
        summary["sampled_decode_workers"] = diagnostics["sampled_workers"]
        summary["decode_tokens_per_second"] = diagnostics["aggregate_decode_tokens_per_second"]
    if "decode_observation_error" in snapshot:
        summary["decode_observation_error"] = snapshot["decode_observation_error"]
    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--decode", action="store_true", help="include live decode diagnostics")
    args = parser.parse_args()
    if args.interval < 10:
        parser.error("interval must be at least 10 seconds")
    root = args.manifest.resolve().parent / "monitor"
    while True:
        try:
            snapshot = observe(args.manifest, decode=args.decode)
            persist(root, snapshot)
            if not args.watch or snapshot["state"] == "complete":
                break
        except (
            OSError, ValueError, RuntimeError, sqlite3.Error, subprocess.SubprocessError
        ) as error:
            # A failed observation is not evidence that a job stopped. Leave
            # the controller untouched and retry the same launch next time.
            print(json.dumps({"observation_error": str(error)}), flush=True)
            if not args.watch:
                raise
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
