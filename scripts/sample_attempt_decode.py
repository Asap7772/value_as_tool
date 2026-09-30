"""Sample fresh SGLang/vLLM generation diagnostics from active, owned Slurm workers.

The default is read-only. ``--write-sidecar`` writes only the monitoring file
``monitor/decode-latest.json``. Decode rates are server diagnostics, not durable
completed throughput, generated-token totals, or an estimate of completion time.
"""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import re
import sqlite3
import subprocess
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from attempt_conditioning_resources import QOS, number

MAX_TAIL_BYTES = 65_536
MAX_DECODE_AGE_SECONDS = 120
TIMESTAMP = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)\]")
VLLM_TIMESTAMP = re.compile(r"\bINFO\s+(\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)\s")


def query_jobs() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["squeue", "--array", "--states=all", "--json", "--user", getpass.getuser()],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    payload = json.loads(result.stdout)
    if payload.get("errors") or not isinstance(payload.get("jobs"), list):
        raise RuntimeError("scheduler did not return an authoritative expanded job listing")
    return payload["jobs"]


def live_workers(manifest: dict[str, Any], raw: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    owned = {}
    for key, job in manifest["jobs"].items():
        if job.get("stage") != manifest["stage"] or not job.get("pool"):
            continue
        parent = number(job.get("job_id"))
        if not parent or parent in owned or job["pool"] not in QOS:
            raise ValueError("ambiguous manifest-owned worker array")
        owned[parent] = {**job, "key": key}
    workers = {}
    selectors = {}
    expected_command = str(Path(manifest["source"]) / "scripts/slurm/run_attempt_worker.sbatch")
    for row in raw:
        parent = number(row.get("array_job_id"))
        if parent not in owned:
            continue
        states = row.get("job_state")
        if isinstance(states, str):
            states = [states]
        if not isinstance(states, list) or not states:
            raise ValueError("scheduler omitted an owned worker's state")
        if set(states) != {"RUNNING"}:
            continue
        job = owned[parent]
        actual, index = number(row.get("job_id")), number(row.get("array_task_id"))
        if not actual or index is None or not 0 <= index < job["workers"]:
            raise ValueError("running owned arrays must have explicit actual job IDs and indices")
        if row.get("array_task_string"):
            raise ValueError("scheduler returned an unexpanded running array")
        if (
            row.get("user_name") != getpass.getuser()
            or row.get("name") != f"attempt-{job['key']}"
            or row.get("qos") != QOS[job["pool"]]
            or row.get("command") != expected_command
        ):
            raise ValueError("running worker identity differs from manifest ownership")
        selector = f"{parent}_{index}"
        item = {
            "actual_job_id": actual,
            "parent_job_id": parent,
            "array_index": index,
            "selector": selector,
            "pool": job["pool"],
        }
        if (actual in workers and workers[actual] != item) or (
            selector in selectors and selectors[selector] != actual
        ):
            raise ValueError("conflicting actual-job/array-element mapping")
        workers[actual] = item
        selectors[selector] = actual
    return workers


def active_claims(manifest: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    entry = manifest["queues"][manifest["stage"]]
    path = Path(entry["path"]).resolve()
    claims: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30) as db:
        identity = db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if not identity or json.loads(identity[0]) != entry["identity"]:
            raise ValueError("queue identity differs from the launch manifest")
        for task_id, worker, started in db.execute(
            "SELECT task_id,worker,started_at FROM tasks WHERE status='running' ORDER BY task_id"
        ):
            if not isinstance(worker, str) or not re.fullmatch(r"[1-9]\d*:[^:]+:[01]", worker):
                raise ValueError("active queue claim has no identifiable Slurm worker")
            when = datetime.fromisoformat(started)
            if when.tzinfo is None:
                raise ValueError("active queue claim has no timestamp timezone")
            claims[int(worker.split(":", 1)[0])].append(
                {"task_id": task_id, "worker": worker, "started_at": when.isoformat()}
            )
    return dict(claims)


def read_tail(path: Path, limit: int = MAX_TAIL_BYTES) -> tuple[str, dict[str, Any]]:
    """Read at most limit bytes, discarding a potentially partial first line."""
    if not 1 <= limit <= MAX_TAIL_BYTES:
        raise ValueError(f"tail limit must be between 1 and {MAX_TAIL_BYTES}")
    with path.open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        start = max(0, size - limit)
        stream.seek(start)
        data = stream.read(limit)
        modified = os.fstat(stream.fileno()).st_mtime
    read_bytes = len(data)
    if start:
        data = data.split(b"\n", 1)[1] if b"\n" in data else b""
    return data.decode("utf-8", errors="replace"), {
        "log_size_bytes": size,
        "log_bytes_read": read_bytes,
        "log_modified_at": datetime.fromtimestamp(modified, UTC).isoformat(),
    }


def _vllm_timestamp(stamp: str, observed: datetime) -> datetime:
    """Infer the nearest UTC year; normal freshness checks still apply afterward."""
    observed = observed.astimezone(UTC)
    candidates = []
    for year in (observed.year - 1, observed.year, observed.year + 1):
        try:
            candidates.append(datetime.fromisoformat(f"{year:04d}-{stamp}").replace(tzinfo=UTC))
        except ValueError:
            continue
    if not candidates:
        raise ValueError("latest vLLM throughput timestamp is invalid")
    return min(candidates, key=lambda when: abs((observed - when).total_seconds()))


def latest_decode(text: str, *, now: datetime | None = None) -> dict[str, Any] | None:
    observed = now or datetime.now(UTC)
    if observed.tzinfo is None:
        raise ValueError("observation time requires a timezone")
    for line in reversed(text.splitlines()):
        if "Decode batch" in line:
            timestamp = TIMESTAMP.search(line)
            requests = re.search(r"#running-req:\s*(\d+)", line)
            full = re.search(r"#full token:\s*(\d+)", line)
            rate = re.search(r"gen throughput \(token/s\):\s*([^,\s]+)", line)
            if not all((timestamp, requests, full, rate)):
                raise ValueError("latest Decode batch line is malformed")
            assert timestamp and full
            when = datetime.fromisoformat(timestamp[1]).replace(tzinfo=UTC)
            full_tokens = int(full[1])
            backend, log_format = "sglang", "sglang_decode_batch"
        elif "Avg generation throughput" in line:
            timestamp = VLLM_TIMESTAMP.search(line)
            requests = re.search(r"\bRunning(?: requests)?:\s*(\d+)", line)
            rate = re.search(r"Avg generation throughput:\s*([^,\s]+)\s+tokens/s", line)
            if not all((timestamp, requests, rate)):
                raise ValueError("latest vLLM throughput line is malformed")
            assert timestamp
            when = _vllm_timestamp(timestamp[1], observed)
            full_tokens = None
            backend, log_format = "vllm", "vllm_avg_generation_throughput"
        else:
            continue
        assert timestamp and requests and rate
        throughput = float(rate[1])
        if not math.isfinite(throughput) or throughput < 0:
            raise ValueError("latest generation throughput is invalid")
        return {
            "backend": backend,
            "log_format": log_format,
            "log_timestamp": timestamp[1],
            "log_timestamp_year_inferred": backend == "vllm",
            "decode_observed_at": when.isoformat(),
            "decode_tokens_per_second": throughput,
            "decode_running_requests": int(requests[1]),
            "full_tokens": full_tokens,
        }
    return None


def sample(manifest_path: Path, *, now: datetime | None = None) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    # A failed scheduler observation must never produce a success-shaped sample.
    workers = live_workers(manifest, query_jobs())
    claims = active_claims(manifest)
    observed = now or datetime.now(UTC)
    if observed.tzinfo is None:
        raise ValueError("observation time requires a timezone")
    role = "judge" if manifest["stage"].endswith("judge") else "solve"
    included, excluded = [], []
    for actual in sorted(set(workers) | set(claims)):
        if actual not in workers:
            excluded.append({"actual_job_id": actual, "reason": "no_owned_running_worker"})
            continue
        worker = workers[actual]
        if actual not in claims:
            excluded.append({**worker, "reason": "no_active_claim"})
            continue
        path = (
            Path(manifest["artifact_root"])
            / "logs"
            / (f"server-{actual}-{worker['array_index']}-{role}.log")
        )
        entry = {**worker, "active_claims": claims[actual], "log_path": str(path)}
        try:
            tail, details = read_tail(path)
            entry.update(details)
            decode = latest_decode(tail, now=observed)
        except (OSError, ValueError) as exc:
            excluded.append({**entry, "reason": "log_unavailable_or_invalid", "error": str(exc)})
            continue
        if decode is None:
            excluded.append({**entry, "reason": "no_decode_record_in_bounded_tail"})
            continue
        when = datetime.fromisoformat(decode["decode_observed_at"])
        modified = datetime.fromisoformat(entry["log_modified_at"])
        age = (observed - when).total_seconds()
        entry.update(decode, decode_age_seconds=age)
        started = min(datetime.fromisoformat(c["started_at"]) for c in claims[actual])
        reason = (
            "stale_decode"
            if age > MAX_DECODE_AGE_SECONDS
            else "future_decode_timestamp"
            if age < -5
            else "inferred_timestamp_after_log_modification"
            if decode["log_timestamp_year_inferred"] and (when - modified).total_seconds() > 5
            else "no_decoding_requests"
            if decode["decode_running_requests"] == 0
            else "decode_predates_active_claims"
            if (started - when).total_seconds() > 1
            else None
        )
        if reason:
            excluded.append({**entry, "reason": reason})
        else:
            included.append(entry)
    return {
        "schema_version": 1,
        "observed_at": observed.isoformat(),
        "manifest": str(manifest_path.resolve()),
        "state": manifest["state"],
        "stage": manifest["stage"],
        "log_timezone_assumption": "UTC",
        "max_decode_age_seconds": MAX_DECODE_AGE_SECONDS,
        "max_log_tail_bytes": MAX_TAIL_BYTES,
        "active_claims": sum(len(value) for value in claims.values()),
        "live_owned_workers": len(workers),
        "sampled_workers": len(included),
        "aggregate_decode_tokens_per_second": (
            sum(row["decode_tokens_per_second"] for row in included) if included else None
        ),
        "aggregate_decode_running_requests": sum(
            row["decode_running_requests"] for row in included
        ),
        "workers": included,
        "excluded": excluded,
        "interpretation": (
            "Sum of each active live worker's newest fresh generation-rate diagnostic: "
            "SGLang Decode batch or vLLM Avg generation throughput (an interval average). "
            "Each server is counted once, regardless of the number of active request lanes. "
            "Samples are asynchronous server diagnostics, not durable completed throughput. "
            "Full tokens are server batch/context counters, not generated-token totals; "
            "they are unavailable for vLLM. vLLM timestamp years are inferred from the "
            "nearest UTC date to observation time, then checked for freshness and against "
            "log modification time. "
            "No completion ETA is inferred."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--write-sidecar", action="store_true")
    args = parser.parse_args()
    snapshot = sample(args.manifest)
    if args.write_sidecar:
        manifest = json.loads(args.manifest.read_text())
        root = Path(manifest["control_root"]) / "monitor"
        root.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=root, delete=False) as stream:
            json.dump(snapshot, stream, indent=2)
            stream.write("\n")
            temporary = Path(stream.name)
        temporary.replace(root / "decode-latest.json")
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
