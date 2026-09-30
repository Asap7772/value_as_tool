from __future__ import annotations

import getpass
import importlib
import json
import os
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

OBSERVED = datetime(2026, 9, 29, 0, 20, tzinfo=UTC)
# Exact pilot-judge record from server-1620761-0-judge.log.
VLLM_JUDGE_RECORD = (
    "(APIServer pid=1742104) INFO 09-29 01:34:15 [loggers.py:310] Engine 000: "
    "Avg prompt throughput: 891.8 tokens/s, Avg generation throughput: 107.7 tokens/s, "
    "Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.1%, Prefix cache hit rate: 0.0%\n"
)


@pytest.fixture
def sampler(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("sample_attempt_decode")


@pytest.fixture
def launch(tmp_path, sampler):
    queue = tmp_path / "queue.sqlite3"
    identity = {"stage": "preprocess", "bank_sha256": "b" * 64}
    with sqlite3.connect(queue) as db:
        db.execute("CREATE TABLE metadata (key TEXT, value TEXT)")
        db.execute("INSERT INTO metadata VALUES ('identity', ?)", (json.dumps(identity),))
        db.execute("CREATE TABLE tasks (task_id TEXT, status TEXT, worker TEXT, started_at TEXT)")
    manifest = {
        "state": "running",
        "stage": "preprocess",
        "source": str(tmp_path / "source"),
        "artifact_root": str(tmp_path / "artifact"),
        "control_root": str(tmp_path),
        "queues": {"preprocess": {"path": str(queue), "identity": identity}},
        "jobs": {
            "preprocess-high-0000": {
                "job_id": "100",
                "stage": "preprocess",
                "pool": "high",
                "workers": 8,
            }
        },
    }
    path = tmp_path / "submission.json"
    path.write_text(json.dumps(manifest))
    logs = Path(manifest["artifact_root"]) / "logs"
    logs.mkdir(parents=True)
    return SimpleNamespace(path=path, manifest=manifest, queue=queue, logs=logs)


def claim(launch, actual=700, *, lane=0, age=300):
    with sqlite3.connect(launch.queue) as db:
        db.execute(
            "INSERT INTO tasks VALUES (?, 'running', ?, ?)",
            (
                f"task-{actual}-{lane}",
                f"{actual}:worker-session:{lane}",
                (OBSERVED - timedelta(seconds=age)).isoformat(),
            ),
        )


def job(launch, actual=700, *, index=3, state="RUNNING"):
    return {
        "job_id": actual,
        "array_job_id": {"set": True, "number": 100},
        "array_task_id": {"set": True, "number": index},
        "array_task_string": "",
        "job_state": [state],
        "user_name": getpass.getuser(),
        "name": "attempt-preprocess-high-0000",
        "qos": "g3_scientific-reasoning_high",
        "command": str(Path(launch.manifest["source"]) / "scripts/slurm/run_attempt_worker.sbatch"),
    }


def line(*, age=10, requests=2, full=170000, rate=58.4):
    stamp = (OBSERVED - timedelta(seconds=age)).strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"[{stamp}] Decode batch, #running-req: {requests}, #full token: {full}, "
        f"full token usage: 0.05, gen throughput (token/s): {rate}, #queue-req: 0\n"
    )


def vllm_line(*, age=10, requests=2, rate=58.4, stamp=None):
    stamp = stamp or (OBSERVED - timedelta(seconds=age)).strftime("%m-%d %H:%M:%S")
    return (
        f"(APIServer pid=1742104) INFO {stamp} [loggers.py:310] "
        f"Engine 000: Avg prompt throughput: 0.0 tokens/s, "
        f"Avg generation throughput: {rate} tokens/s, Running: {requests} reqs, "
        "Waiting: 0 reqs, GPU KV cache usage: 0.1%, Prefix cache hit rate: 0.0%\n"
    )


@pytest.mark.parametrize("make_line", [line, vllm_line], ids=["sglang", "vllm"])
def test_actual_job_mapping_counts_one_server_once_with_two_active_requests(
    sampler,
    launch,
    monkeypatch,
    make_line,
):
    claim(launch)
    claim(launch, lane=1)
    (launch.logs / "server-700-3-solve.log").write_text(make_line(rate=40) + make_line(rate=58.4))
    (launch.logs / "server-100-3-solve.log").write_text(line(rate=999))
    monkeypatch.setattr(sampler, "query_jobs", lambda: [job(launch), job(launch)])
    result = sampler.sample(launch.path, now=OBSERVED)
    assert result["active_claims"] == 2 and result["sampled_workers"] == 1
    assert result["aggregate_decode_tokens_per_second"] == 58.4
    assert result["aggregate_decode_running_requests"] == 2
    worker = result["workers"][0]
    assert worker["actual_job_id"] == 700 and worker["selector"] == "100_3"
    assert worker["full_tokens"] == (170000 if make_line is line else None)
    assert worker["backend"] == ("sglang" if make_line is line else "vllm")
    assert worker["log_timestamp_year_inferred"] == (make_line is vllm_line)
    assert worker["decode_observed_at"] == "2026-09-29T00:19:50+00:00"
    assert worker["decode_age_seconds"] == 10
    assert worker["log_path"].endswith("server-700-3-solve.log")
    assert not any(key.startswith("eta") or "generated_tokens" in key for key in result)


def test_judge_stage_aggregates_only_exact_role_logs(sampler, launch, monkeypatch):
    manifest = launch.manifest
    manifest["stage"] = "judge"
    manifest["queues"]["judge"] = manifest["queues"].pop("preprocess")
    owned = manifest["jobs"].pop("preprocess-high-0000")
    owned["stage"] = "judge"
    manifest["jobs"]["judge-high-0000"] = owned
    launch.path.write_text(json.dumps(manifest))
    claim(launch, 700)
    claim(launch, 701)
    (launch.logs / "server-700-3-judge.log").write_text(vllm_line(rate=21.5, requests=1))
    (launch.logs / "server-701-4-judge.log").write_text(vllm_line(rate=12.5, requests=1))
    (launch.logs / "server-700-3-solve.log").write_text(line(rate=999))
    rows = [job(launch, 700), job(launch, 701, index=4)]
    for row in rows:
        row["name"] = "attempt-judge-high-0000"
    monkeypatch.setattr(sampler, "query_jobs", lambda: rows)
    result = sampler.sample(launch.path, now=OBSERVED)
    assert result["sampled_workers"] == 2
    assert result["aggregate_decode_tokens_per_second"] == 34
    assert all(row["log_path"].endswith("-judge.log") for row in result["workers"])
    assert all(row["full_tokens"] is None for row in result["workers"])
    assert all(row["log_format"] == "vllm_avg_generation_throughput" for row in result["workers"])


def test_queue_identity_mismatch_blocks_all_log_reads(sampler, launch, monkeypatch):
    claim(launch)
    with sqlite3.connect(launch.queue) as db:
        db.execute("UPDATE metadata SET value='{}' WHERE key='identity'")
    monkeypatch.setattr(sampler, "query_jobs", lambda: [job(launch)])
    monkeypatch.setattr(sampler, "read_tail", lambda *_: pytest.fail("untrusted queue log read"))
    with pytest.raises(ValueError, match="queue identity"):
        sampler.sample(launch.path, now=OBSERVED)


@pytest.mark.parametrize(
    "age,requests,claim_age,reason",
    [
        (121, 1, 300, "stale_decode"),
        (-10, 1, 300, "future_decode_timestamp"),
        (10, 0, 300, "no_decoding_requests"),
        (30, 1, 10, "decode_predates_active_claims"),
    ],
)
@pytest.mark.parametrize("make_line", [line, vllm_line], ids=["sglang", "vllm"])
def test_stale_idle_and_old_claim_samples_are_excluded(
    sampler,
    launch,
    monkeypatch,
    age,
    requests,
    claim_age,
    reason,
    make_line,
):
    claim(launch, age=claim_age)
    (launch.logs / "server-700-3-solve.log").write_text(make_line(age=age, requests=requests))
    monkeypatch.setattr(sampler, "query_jobs", lambda: [job(launch)])
    result = sampler.sample(launch.path, now=OBSERVED)
    assert result["sampled_workers"] == 0
    assert result["aggregate_decode_tokens_per_second"] is None
    assert result["excluded"][0]["reason"] == reason


def test_idle_nonlive_and_unowned_workers_never_have_logs_read(sampler, launch, monkeypatch):
    claim(launch, 701)
    claim(launch, 702)
    unowned = {**job(launch, 702, index=2), "array_job_id": {"set": True, "number": 999}}
    monkeypatch.setattr(
        sampler,
        "query_jobs",
        lambda: [
            job(launch),
            job(launch, 701, index=1, state="COMPLETED"),
            unowned,
        ],
    )
    monkeypatch.setattr(sampler, "read_tail", lambda *_: pytest.fail("ineligible log was opened"))
    result = sampler.sample(launch.path, now=OBSERVED)
    assert result["workers"] == []
    assert {row["actual_job_id"]: row["reason"] for row in result["excluded"]} == {
        700: "no_active_claim",
        701: "no_owned_running_worker",
        702: "no_owned_running_worker",
    }


@pytest.mark.parametrize("fault", ["unexpanded", "wrong_command", "conflict", "missing_actual"])
def test_ambiguous_scheduler_mapping_fails_closed(sampler, launch, monkeypatch, fault):
    row = job(launch)
    rows = [row]
    if fault == "unexpanded":
        row["array_task_string"] = "0-7"
    elif fault == "wrong_command":
        row["command"] = "another-launch.sh"
    elif fault == "conflict":
        rows.append(job(launch, 701))
    else:
        row["job_id"] = None
    monkeypatch.setattr(sampler, "query_jobs", lambda: rows)
    with pytest.raises(ValueError):
        sampler.sample(launch.path, now=OBSERVED)


def test_log_reads_are_bounded_and_ignore_old_prefix(sampler, tmp_path, monkeypatch):
    path = tmp_path / "server.log"
    path.write_text(line(rate=999) + "prefix\n" * 100000 + line(rate=20))
    original_open = Path.open
    reads = []

    class Tracked:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.stream.close()

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, size=-1):
            assert 0 <= size <= sampler.MAX_TAIL_BYTES
            reads.append(size)
            return self.stream.read(size)

    monkeypatch.setattr(Path, "open", lambda self, *a, **k: Tracked(original_open(self, *a, **k)))
    tail, details = sampler.read_tail(path, 512)
    assert reads == [512]
    assert details["log_bytes_read"] == 512
    assert details["log_size_bytes"] > 512
    assert "999" not in tail
    assert sampler.latest_decode(tail)["decode_tokens_per_second"] == 20


def test_malformed_newest_decode_does_not_fall_back_to_older_rate(sampler):
    with pytest.raises(ValueError, match="malformed"):
        sampler.latest_decode(line() + "[2026-09-29 00:19:59] Decode batch, missing fields\n")


def test_actual_vllm_judge_log_fixture(sampler):
    observed = datetime(2026, 9, 29, 1, 34, 42, tzinfo=UTC)
    assert sampler.latest_decode(VLLM_JUDGE_RECORD, now=observed) == {
        "backend": "vllm",
        "log_format": "vllm_avg_generation_throughput",
        "log_timestamp": "09-29 01:34:15",
        "log_timestamp_year_inferred": True,
        "decode_observed_at": "2026-09-29T01:34:15+00:00",
        "decode_tokens_per_second": 107.7,
        "decode_running_requests": 2,
        "full_tokens": None,
    }


@pytest.mark.parametrize(
    "stamp,observed,expected",
    [
        ("12-31 23:59:55", "2027-01-01T00:00:05+00:00", "2026-12-31T23:59:55+00:00"),
        ("01-01 00:00:05", "2026-12-31T23:59:55+00:00", "2027-01-01T00:00:05+00:00"),
        ("02-29 23:59:55", "2024-03-01T00:00:05+00:00", "2024-02-29T23:59:55+00:00"),
        ("12-31 22:59:55", "2027-01-01T01:00:05+02:00", "2026-12-31T22:59:55+00:00"),
    ],
)
def test_vllm_year_is_inferred_nearest_to_observation_in_utc(sampler, stamp, observed, expected):
    parsed = sampler.latest_decode(vllm_line(stamp=stamp), now=datetime.fromisoformat(observed))
    assert parsed["decode_observed_at"] == expected
    assert parsed["log_timestamp_year_inferred"] is True


def test_old_yearless_log_cannot_appear_fresh_in_a_later_year(sampler, launch, monkeypatch):
    claim(launch)
    path = launch.logs / "server-700-3-solve.log"
    path.write_text(vllm_line())
    old = OBSERVED.replace(year=2025).timestamp()
    os.utime(path, (old, old))
    monkeypatch.setattr(sampler, "query_jobs", lambda: [job(launch)])
    result = sampler.sample(launch.path, now=OBSERVED)
    assert result["sampled_workers"] == 0
    assert result["excluded"][0]["reason"] == "inferred_timestamp_after_log_modification"


@pytest.mark.parametrize(
    "record,match",
    [
        (vllm_line().replace("Running: 2 reqs", "missing requests"), "malformed"),
        (vllm_line(stamp="02-30 00:19:50"), "timestamp is invalid"),
        (vllm_line(rate="nan"), "throughput is invalid"),
        (vllm_line(rate="inf"), "throughput is invalid"),
        (vllm_line(rate=-1), "throughput is invalid"),
    ],
)
def test_bad_latest_vllm_stat_never_reuses_an_older_rate(sampler, record, match):
    with pytest.raises(ValueError, match=match):
        sampler.latest_decode(line() + vllm_line() + record, now=OBSERVED)


def test_latest_generation_stat_wins_across_log_formats(sampler):
    assert sampler.latest_decode(line() + vllm_line(), now=OBSERVED)["backend"] == "vllm"
    assert sampler.latest_decode(vllm_line() + line(), now=OBSERVED)["backend"] == "sglang"


def test_scheduler_failure_does_not_write_a_sidecar(sampler, launch, monkeypatch):
    def fail():
        raise subprocess.CalledProcessError(1, ["squeue"])

    monkeypatch.setattr(sampler, "query_jobs", fail)
    monkeypatch.setattr(sys, "argv", ["sample", "--manifest", str(launch.path), "--write-sidecar"])
    with pytest.raises(subprocess.CalledProcessError):
        sampler.main()
    assert not (launch.path.parent / "monitor").exists()


def test_cli_default_is_read_only_and_explicit_write_is_only_a_sidecar(
    sampler,
    launch,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(sampler, "query_jobs", lambda: [])
    original_manifest = launch.path.read_bytes()
    original_queue = launch.queue.read_bytes()
    monkeypatch.setattr(sys, "argv", ["sample", "--manifest", str(launch.path)])
    sampler.main()
    assert json.loads(capsys.readouterr().out)["sampled_workers"] == 0
    monitor = launch.path.parent / "monitor"
    assert not monitor.exists()
    monkeypatch.setattr(sys, "argv", ["sample", "--manifest", str(launch.path), "--write-sidecar"])
    sampler.main()
    assert [path.name for path in monitor.iterdir()] == ["decode-latest.json"]
    assert json.loads((monitor / "decode-latest.json").read_text())["sampled_workers"] == 0
    assert launch.path.read_bytes() == original_manifest
    assert launch.queue.read_bytes() == original_queue
