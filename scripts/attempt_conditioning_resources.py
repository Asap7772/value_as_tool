"""Fail-closed accounting for a bounded, current-stage Slurm worker pool."""

from __future__ import annotations

import getpass
import json
import math
import re
import subprocess
from collections import Counter
from typing import Any

QOS = {"high": "g3_scientific-reasoning_high", "shared": "g3_core_shared"}
LIMITS = {"high": 212, "shared": 64}
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}


def number(value: Any) -> int | None:
    if isinstance(value, dict):
        if not value.get("set") or value.get("infinite"):
            return None
        value = value.get("number")
    if value is None:
        return None
    if isinstance(value, bool) or not re.fullmatch(r"\d+", str(value)):
        raise ValueError(f"invalid Slurm integer: {value!r}")
    return int(value)


def array_indices(expression: str) -> list[int]:
    expression = expression.strip("[]").split("%", 1)[0]
    result: set[int] = set()
    for part in expression.split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+)(?::(\d+))?)?", part)
        if not match:
            raise ValueError(f"unrecognized array expression: {expression!r}")
        first, last, step = (int(match[1]), int(match[2] or match[1]), int(match[3] or 1))
        if last < first or step < 1 or (last - first) // step > 100_000:
            raise ValueError("array bounds require explicit review")
        result.update(range(first, last + 1, step))
    return sorted(result)


def requested_gpus(job: dict[str, Any]) -> int:
    """ReqTRES is a job total; generic GPU totals include typed GPU entries."""
    raw = job.get("tres_req_str")
    if not isinstance(raw, str) or not raw:
        raise ValueError("Slurm omitted requested TRES; cannot establish GPU headroom")
    gpu: dict[str, int] = {}
    for item in raw.split(","):
        key, separator, value = item.partition("=")
        if key.startswith("gres/gpu"):
            if not separator or not re.fullmatch(r"\d+", value) or key in gpu:
                raise ValueError(f"ambiguous GPU request: {raw!r}")
            gpu[key] = int(value)
    if "gres/gpu" in gpu:
        total = gpu.pop("gres/gpu")
        if sum(gpu.values()) > total:
            raise ValueError(f"typed GPU requests exceed generic total: {raw!r}")
        return total
    if gpu:
        return sum(gpu.values())
    if any("gpu" in str(job.get(k, "")) for k in ("tres_per_node", "tres_per_job")):
        raise ValueError("GPU request is absent from requested TRES")
    return 0


def expand_jobs(raw_jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand pending arrays and deduplicate overlapping parent/element records."""
    expanded: dict[str, dict[str, Any]] = {}
    for job in raw_jobs:
        state = job["job_state"]
        states = set(state if isinstance(state, list) else [state])
        if states <= TERMINAL:
            continue
        count = requested_gpus(job)
        parent = number(job.get("array_job_id")) or number(job["job_id"])
        task = number(job.get("array_task_id"))
        expression = job.get("array_task_string") or ""
        indices = [task] if task is not None else array_indices(expression) if expression else []
        ids = [f"{parent}_{index}" for index in indices] or [str(job["job_id"])]
        for job_id in ids:
            row = {
                "job_id": job_id,
                "parent": str(parent),
                "qos": job["qos"],
                "gpus": count,
                "states": sorted(states),
            }
            previous = expanded.get(job_id)
            if previous and (previous["qos"], previous["gpus"]) != (row["qos"], count):
                raise ValueError(f"conflicting resource records for {job_id}")
            expanded[job_id] = row
    return list(expanded.values())


def query_user_jobs() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["squeue", "--json", "--user", getpass.getuser()],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    payload = json.loads(result.stdout)
    if payload.get("errors"):
        raise RuntimeError(f"squeue reported errors: {payload['errors']}")
    return expand_jobs(payload["jobs"])


def admission_plan(
    rows: list[dict[str, Any]],
    jobs: dict[str, dict[str, Any]],
    desired_workers: int,
) -> dict[str, Any]:
    """Reserve every manifest-owned nonterminal submission, even before squeue sees it."""
    owned = {str(j["job_id"]): j for j in jobs.values() if j.get("job_id") and j.get("pool")}
    other: Counter[str] = Counter()
    own: Counter[str] = Counter()
    visible_owned: dict[str, set[str]] = {parent: set() for parent in owned}
    for row in rows:
        pool = next((p for p, qos in QOS.items() if qos == row["qos"]), None)
        if pool and row["parent"] not in owned:
            other[pool] += row["gpus"]
        elif row["parent"] in owned:
            if pool != owned[row["parent"]]["pool"] or row["gpus"] != 1:
                raise ValueError("manifest-owned worker has unexpected QoS or GPU resources")
            visible_owned[row["parent"]].add(row["job_id"])
    for parent, job in owned.items():
        expected = {f"{parent}_{i}" for i in range(int(job["workers"]))}
        if not visible_owned[parent] <= expected:
            raise ValueError("unexpected element in manifest-owned worker array")
        terminal = {
            key for key, row in job.get("accounting", {}).items() if row["state"] in TERMINAL
        }
        # A current live row wins over older terminal accounting. Missing rows
        # remain reserved until exact per-element terminal evidence exists.
        reserved = visible_owned[parent] | (expected - terminal)
        if job.get("terminal_confirmed") and not job.get("accounting"):
            reserved = visible_owned[parent]
        own[job["pool"]] += len(reserved)
    capacity = {p: max(0, cap - other[p]) for p, cap in LIMITS.items()}
    target = min(desired_workers, sum(capacity.values()))
    high = min(
        capacity["high"], math.ceil(target * capacity["high"] / max(1, sum(capacity.values())))
    )
    if target >= 2 and all(capacity.values()):
        high = min(high, target - 1)
    targets = {"high": high, "shared": target - high}
    remaining = max(0, desired_workers - sum(own.values()))
    additions = {}
    for pool in LIMITS:
        additions[pool] = min(
            remaining, max(0, capacity[pool] - own[pool]), max(0, targets[pool] - own[pool])
        )
        remaining -= additions[pool]
    for pool in LIMITS:
        extra = min(remaining, max(0, capacity[pool] - own[pool] - additions[pool]))
        additions[pool] += extra
        remaining -= extra
    return {
        "limits": LIMITS,
        "other_requested": dict(other),
        "own_reserved": dict(own),
        "add_workers": additions,
        "desired_workers": desired_workers,
        "over_limit": [p for p, cap in LIMITS.items() if other[p] + own[p] > cap],
    }
