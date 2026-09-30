"""Retire idle preprocessing allocations without interrupting claimed model work.

Dry-run is the default. With --retire, the existing queue flock is held from
claim inspection until Slurm confirms every cancellation is terminal. After a
cancellation is attempted, a confirmation timeout retains that lock and keeps
polling; SIGINT/SIGTERM are deferred until it is safe to release. No launch
manifest, queue contents, or immutable source files are changed.
"""

from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import os
import re
import signal
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TERMINAL = {
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
    "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT",
}
ELIGIBLE = {"RUNNING", "PENDING"}
QOS = {"high": "g3_scientific-reasoning_high", "shared": "g3_core_shared"}


def now() -> str:
    return datetime.now(UTC).isoformat()


def emit(value: Any) -> None:
    """Closed monitoring output must never release an in-flight cancellation lock."""
    try:
        print(value, flush=True)
    except OSError:
        pass


def write_record(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def number(value: Any) -> int | None:
    if isinstance(value, dict):
        if not value.get("set") or value.get("infinite"):
            return None
        value = value.get("number")
    if value is None:
        return None
    if isinstance(value, bool) or not re.fullmatch(r"\d+", str(value)):
        raise ValueError("invalid scheduler job identifier")
    return int(value)


def states(row: dict[str, Any]) -> set[str]:
    value = row.get("job_state")
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value or any(not isinstance(v, str) for v in value):
        raise ValueError("scheduler omitted job state")
    return set(value)


def owned_jobs(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if manifest.get("stage") != "preprocess" or manifest.get("state") != "running":
        raise ValueError("retirement requires a running preprocessing stage")
    result = {}
    for key, job in manifest["jobs"].items():
        if job.get("stage") != "preprocess" or not job.get("pool"):
            continue
        parent = number(job.get("job_id"))
        if not parent or parent in result or job["pool"] not in QOS:
            raise ValueError("ambiguous preprocessing submission ownership")
        if not isinstance(job.get("workers"), int) or job["workers"] < 1:
            raise ValueError("invalid preprocessing array size")
        result[parent] = {**job, "key": key}
    if not result:
        raise ValueError("no owned preprocessing worker arrays")
    return result


def query_jobs(parents: list[int]) -> list[dict[str, Any]]:
    # Query by user so a purged cancelled parent does not make squeue reject
    # the whole request with "invalid job id" during terminal confirmation.
    result = subprocess.run(
        ["squeue", "--array", "--states=all", "--json", "--user", getpass.getuser()],
        text=True, capture_output=True, check=True, timeout=30,
    )
    payload = json.loads(result.stdout)
    if payload.get("errors") or not isinstance(payload.get("jobs"), list):
        raise RuntimeError("scheduler did not return an authoritative job listing")
    return [row for row in payload["jobs"] if number(row.get("array_job_id")) in parents]


def queue_snapshot(queue: Path) -> dict[str, Any]:
    # The caller already holds the queue's ordinary flock. Do not open a second
    # TaskQueue transaction, which would deadlock against our own lock.
    with sqlite3.connect(f"file:{queue}?mode=ro", uri=True) as db:
        identity = json.loads(db.execute(
            "SELECT value FROM metadata WHERE key='identity'"
        ).fetchone()[0])
        counts = dict(db.execute("SELECT status,count(*) FROM tasks GROUP BY status"))
        active = []
        for task_id, worker in db.execute(
            "SELECT task_id,worker FROM tasks WHERE status='running' ORDER BY task_id"
        ):
            if not isinstance(worker, str) or not re.fullmatch(r"\d+:[^:]+:[01]", worker):
                raise ValueError(f"cannot identify active claim owner: {task_id}")
            active.append({
                "task_id": task_id, "worker": worker, "job_id": int(worker.split(":")[0]),
            })
    return {"identity": identity, "counts": counts, "active_claims": active}


def allocation_rows(
    raw: list[dict[str, Any]], manifest: dict[str, Any], owned: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = {}
    expected_command = str(Path(manifest["source"]) / "scripts/slurm/run_attempt_worker.sbatch")
    for row in raw:
        parent = number(row.get("array_job_id"))
        if parent not in owned:
            continue
        job = owned[parent]
        index = number(row.get("array_task_id"))
        actual = number(row.get("job_id"))
        if index is None or not 0 <= index < job["workers"] or actual is None:
            raise ValueError("scheduler did not expand an owned array into explicit elements")
        if row.get("array_task_string"):
            raise ValueError("unexpanded array expression is unsafe to cancel")
        if (
            row.get("user_name") != getpass.getuser()
            or row.get("name") != f"attempt-{job['key']}"
            or row.get("qos") != QOS[job["pool"]]
            or row.get("command") != expected_command
        ):
            raise ValueError("live allocation identity differs from manifest ownership")
        selector = f"{parent}_{index}"
        item = {
            "selector": selector, "job_id": actual, "parent": parent, "index": index,
            "states": sorted(states(row)), "pool": job["pool"],
        }
        if selector in rows and rows[selector] != item:
            raise ValueError("conflicting allocation records")
        rows[selector] = item
    return list(rows.values())


def retirement_plan(
    manifest: dict[str, Any], snapshot: dict[str, Any], raw: list[dict[str, Any]], keep_idle: int
) -> dict[str, Any]:
    owned = owned_jobs(manifest)
    if snapshot["identity"] != manifest["queues"]["preprocess"]["identity"]:
        raise ValueError("queue identity differs from launch manifest")
    if snapshot["counts"].get("failed"):
        raise ValueError("failed preprocessing tasks require separate recovery")
    rows = allocation_rows(raw, manifest, owned)
    active_ids = {claim["job_id"] for claim in snapshot["active_claims"]}
    if active_ids - {row["job_id"] for row in rows}:
        raise ValueError("an active claim cannot be mapped to an owned live allocation")
    active = [row for row in rows if row["job_id"] in active_ids]
    idle = sorted(
        (row for row in rows if row["job_id"] not in active_ids and set(row["states"]) <= ELIGIBLE),
        key=lambda row: (row["states"] != ["RUNNING"], row["parent"], row["index"]),
    )
    return {
        "active_claims": snapshot["active_claims"], "queue_counts": snapshot["counts"],
        "protected_allocations": active, "kept_idle": idle[:keep_idle],
        "retire": idle[keep_idle:], "keep_idle": keep_idle,
    }


def confirmed_terminal(
    selected: list[dict[str, Any]], raw: list[dict[str, Any]], accounting: dict[str, str]
) -> tuple[list[str], list[str]]:
    visible: dict[str, set[str]] = {}
    parents = {int(row["selector"].split("_")[0]) for row in selected}
    for row in raw:
        parent, index = number(row.get("array_job_id")), number(row.get("array_task_id"))
        if parent in parents and (index is None or row.get("array_task_string")):
            raise ValueError("cannot confirm cancellation from unexpanded live array records")
        if parent is not None and index is not None:
            selector = f"{parent}_{index}"
            visible.setdefault(selector, set()).update(states(row))
    confirmed, unresolved = [], []
    for row in selected:
        selector = row["selector"]
        current = visible.get(selector)
        # Absence alone is insufficient. Exact accounting is the fallback only
        # after a job disappears from the all-states live controller listing.
        terminal = accounting.get(selector) in TERMINAL and (
            not current or current <= TERMINAL
        )
        (confirmed if terminal else unresolved).append(selector)
    return confirmed, unresolved


def query_accounting(selectors: list[str]) -> dict[str, str]:
    result = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-j", ",".join(selectors), "--format=JobID%64,State%40"],
        text=True, capture_output=True, check=True, timeout=30,
    )
    records = {}
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            raise ValueError("invalid scheduler accounting record")
        selector, state = parts[:2]
        if selector not in selectors:
            continue
        state = state.split()[0]  # CANCELLED may include the requesting UID.
        if selector in records and records[selector] != state:
            raise ValueError("ambiguous scheduler accounting record")
        records[selector] = state
    return records


@contextmanager
def queue_lock(path: Path, timeout_seconds: float):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("bounded queue lock acquisition requires the main thread")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0] > 0:
        raise RuntimeError("an active SIGALRM timer prevents safe retirement lock acquisition")

    def expired(_signum, _frame):
        # Raising prevents Python from automatically retrying interrupted flock.
        raise TimeoutError("queue lock acquisition timed out; no cancellation attempted")

    # NFS emulates flock with fcntl byte-range locks and requires a writable
    # descriptor for LOCK_EX. r+ preserves the existing lock inode and contents.
    with path.with_suffix(path.suffix + ".lock").open("r+") as handle:
        signal.signal(signal.SIGALRM, expired)
        try:
            signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
            # Join the filesystem's blocking waiter queue. Nonblocking retries
            # can starve behind hundreds of normal blocking worker transactions.
            fcntl.flock(handle, fcntl.LOCK_EX)
        finally:
            # The deadline applies solely to acquisition, never to cancellation
            # or terminal confirmation while this lock protects active work.
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def defer_signals():
    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[sig] = signal.signal(sig, lambda *_: None)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def retire_idle(
    manifest_path: Path, *, retire: bool = False, keep_idle: int = 4,
    lock_timeout: float = 120, confirmation_timeout: float = 180, poll_seconds: float = 3,
) -> dict[str, Any]:
    if keep_idle < 4 or min(lock_timeout, confirmation_timeout, poll_seconds) <= 0:
        raise ValueError("keep at least four idle workers and use positive timeouts")
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    owned_jobs(manifest)
    queue = Path(manifest["queues"]["preprocess"]["path"])
    record_path = Path(manifest["control_root"]) / "monitor" / (
        f"retire-idle-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}.json"
    )
    record: dict[str, Any] = {
        "created_at": now(), "manifest": str(manifest_path), "record": str(record_path),
        "dry_run": not retire, "state": "waiting_for_queue_lock", "actions": [],
    }
    write_record(record_path, record)
    try:
        with queue_lock(queue, lock_timeout):
            # Stage changes and worker admissions are checked after lock acquisition.
            manifest = json.loads(manifest_path.read_text())
            owned = owned_jobs(manifest)
            if Path(manifest["queues"]["preprocess"]["path"]) != queue:
                raise ValueError("launch queue changed while acquiring lock")
            raw = query_jobs(sorted(owned))
            record["plan"] = retirement_plan(manifest, queue_snapshot(queue), raw, keep_idle)
            record.update(state="planned", planned_at=now())
            write_record(record_path, record)
            selected = record["plan"]["retire"]
            if not retire or not selected:
                record.update(
                    state="dry_run" if not retire else "no_idle_workers", completed_at=now(),
                )
                write_record(record_path, record)
                return record
            selectors = [row["selector"] for row in selected]
            command = ["scancel", "--ctld", "--full", *selectors]
            record["actions"].append({"at": now(), "command": command, "state": "intent"})
            write_record(record_path, record)
            # Once cancellation may have reached Slurm, do not let an interrupt,
            # scheduler timeout, or log-write failure release workers into an
            # outstanding asynchronous cancellation.
            with defer_signals():
                try:
                    result = subprocess.run(
                        command, text=True, capture_output=True, check=False, timeout=30,
                    )
                    record["actions"][-1].update(
                        returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                    )
                except Exception as exc:
                    record["actions"][-1]["error"] = f"{type(exc).__name__}: {exc}"
                started = time.monotonic()
                unresolved = selectors
                while unresolved:
                    try:
                        raw = query_jobs(sorted(owned))
                        confirmed, unresolved = confirmed_terminal(
                            selected, raw, query_accounting(selectors)
                        )
                        record.update(confirmed=confirmed, unresolved=unresolved)
                    except Exception as exc:
                        record["confirmation_error"] = f"{type(exc).__name__}: {exc}"
                    record.update(
                        state="confirmed" if not unresolved else "waiting_for_confirmation",
                        updated_at=now(),
                        confirmation_timeout_exceeded=(
                            time.monotonic() - started > confirmation_timeout
                        ),
                    )
                    try:
                        write_record(record_path, record)
                    except OSError as exc:
                        emit(f"Retirement record write failed; retaining queue lock: {exc}")
                    if unresolved:
                        emit(json.dumps({
                            "state": record["state"], "unresolved": len(unresolved),
                            "queue_lock_held": True,
                            "confirmation_timeout_exceeded": (
                                record["confirmation_timeout_exceeded"]
                            ),
                        }))
                        time.sleep(poll_seconds)
                record["completed_at"] = now()
                write_record(record_path, record)
            return record
    except Exception as exc:
        record.update(state="failed_before_retirement", error=f"{type(exc).__name__}: {exc}")
        write_record(record_path, record)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--retire", action="store_true")
    parser.add_argument("--keep-idle", type=int, default=4)
    parser.add_argument("--lock-timeout", type=float, default=120)
    parser.add_argument("--confirmation-timeout", type=float, default=180)
    args = parser.parse_args()
    print(json.dumps(retire_idle(
        args.manifest, retire=args.retire, keep_idle=args.keep_idle,
        lock_timeout=args.lock_timeout, confirmation_timeout=args.confirmation_timeout,
    ), indent=2))


if __name__ == "__main__":
    main()
