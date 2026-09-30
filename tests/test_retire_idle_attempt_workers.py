from __future__ import annotations

import copy
import fcntl
import getpass
import importlib
import json
import os
import signal
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    retirement = importlib.import_module("retire_idle_attempt_workers")
    queue_module = importlib.import_module("attempt_conditioning_queue")
    queue = queue_module.TaskQueue(tmp_path / "queues/preprocess.sqlite3")
    identity = {"stage": "preprocess", "bank_sha256": "bank"}
    queue.initialize([{"task_id": "active", "dependencies": []}], identity)
    # Slurm can assign the parent numeric ID to an array element. Cancelling
    # bare '100' here would also kill this protected active sibling.
    claim = queue.claim("100:session:0")
    manifest = {
        "stage": "preprocess", "state": "running", "control_root": str(tmp_path),
        "source": str(tmp_path / "source"),
        "queues": {"preprocess": {"path": str(queue.path), "identity": identity}},
        "jobs": {
            "controller": {"job_id": "99", "stage": "controller", "pool": None},
            "preprocess-high-0000": {
                "job_id": "100", "stage": "preprocess", "pool": "high", "workers": 8,
            },
        },
    }
    path = tmp_path / "submission.json"
    path.write_text(json.dumps(manifest))
    rows = [
        {
            "job_id": 200 + i if i != 7 else 100,
            "array_job_id": {"set": True, "number": 100},
            "array_task_id": {"set": True, "number": i},
            "array_task_string": "", "job_state": ["RUNNING"],
            "user_name": getpass.getuser(), "name": "attempt-preprocess-high-0000",
            "qos": "g3_scientific-reasoning_high",
            "command": str(tmp_path / "source/scripts/slurm/run_attempt_worker.sbatch"),
        }
        for i in range(8)
    ]
    monkeypatch.setattr(retirement, "query_jobs", lambda _: copy.deepcopy(rows))
    monkeypatch.setattr(
        retirement, "query_accounting", lambda ids: {i: "CANCELLED" for i in ids}
    )
    monkeypatch.setattr(
        retirement.subprocess, "run", lambda *a, **k: pytest.fail("unexpected scheduler mutation")
    )
    yield SimpleNamespace(
        module=retirement, queue=queue, manifest=manifest, path=path, rows=rows,
        claim=claim, tmp_path=tmp_path,
    )
    claim.close()


def assert_queue_locked(queue):
    with queue.path.with_suffix(".sqlite3.lock").open("r+") as handle:
        with pytest.raises(BlockingIOError):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def assert_queue_unlocked(queue):
    with queue.path.with_suffix(".sqlite3.lock").open("r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(handle, fcntl.LOCK_UN)


def test_default_dry_run_protects_actual_parent_id_and_preserves_all_artifacts(setup):
    original_manifest = setup.path.read_bytes()
    original_queue = setup.queue.path.read_bytes()
    record = setup.module.retire_idle(setup.path)
    assert record["dry_run"] and record["state"] == "dry_run"
    assert [r["selector"] for r in record["plan"]["retire"]] == ["100_4", "100_5", "100_6"]
    assert [r["selector"] for r in record["plan"]["protected_allocations"]] == ["100_7"]
    assert len(record["plan"]["kept_idle"]) == 4
    assert record["actions"] == []
    assert setup.path.read_bytes() == original_manifest
    assert setup.queue.path.read_bytes() == original_queue
    assert json.loads(Path(record["record"]).read_text()) == record
    assert_queue_unlocked(setup.queue)


def test_retirement_holds_lock_until_all_explicit_elements_are_confirmed(setup, monkeypatch):
    module = setup.module
    queries = 0

    def jobs(_):
        nonlocal queries
        assert_queue_locked(setup.queue)
        queries += 1
        rows = copy.deepcopy(setup.rows)
        if queries > 1:
            for row in rows[4:7]:
                row["job_state"] = ["CANCELLED"]
            if queries == 2:
                rows[4]["job_state"] = ["CANCELLED", "COMPLETING"]
        return rows

    def cancel(command, **kwargs):
        assert_queue_locked(setup.queue)
        assert command == ["scancel", "--ctld", "--full", "100_4", "100_5", "100_6"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module, "query_jobs", jobs)
    monkeypatch.setattr(module.subprocess, "run", cancel)
    monkeypatch.setattr(module.time, "sleep", lambda _: assert_queue_locked(setup.queue))
    record = module.retire_idle(setup.path, retire=True)
    assert record["state"] == "confirmed"
    assert record["confirmed"] == ["100_4", "100_5", "100_6"]
    assert record["unresolved"] == [] and queries == 3
    assert_queue_unlocked(setup.queue)


def test_timeout_after_uncertain_cancel_keeps_lock_until_accounting_arrives(setup, monkeypatch):
    module = setup.module
    queries = 0
    accounting_calls = 0
    records = []
    original_write = module.write_record

    def jobs(_):
        nonlocal queries
        queries += 1
        if queries == 1:
            return setup.rows
        if queries == 2:
            raise TimeoutError("scheduler unavailable")
        return []

    def accounting(selectors):
        nonlocal accounting_calls
        accounting_calls += 1
        return {} if accounting_calls == 1 else {s: "CANCELLED" for s in selectors}

    def cancel(*args, **kwargs):
        assert_queue_locked(setup.queue)
        raise TimeoutError("cancellation RPC timed out")

    def save(path, record):
        if record["state"] == "waiting_for_confirmation":
            assert_queue_locked(setup.queue)
            records.append(copy.deepcopy(record))
        original_write(path, record)

    clock = iter([0, 2, 3, 4])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(module.time, "sleep", lambda _: assert_queue_locked(setup.queue))
    monkeypatch.setattr(module, "query_jobs", jobs)
    monkeypatch.setattr(module, "query_accounting", accounting)
    monkeypatch.setattr(module.subprocess, "run", cancel)
    monkeypatch.setattr(module, "write_record", save)
    record = module.retire_idle(setup.path, retire=True, confirmation_timeout=1)
    assert record["state"] == "confirmed"
    assert any(r["confirmation_timeout_exceeded"] for r in records)
    assert "TimeoutError" in record["actions"][0]["error"]
    assert_queue_unlocked(setup.queue)


@pytest.mark.parametrize("stage", ["pilot_solve", "pilot_judge", "solve", "judge"])
def test_non_preprocess_stages_are_never_retired(setup, stage):
    setup.manifest["stage"] = stage
    setup.path.write_text(json.dumps(setup.manifest))
    with pytest.raises(ValueError, match="preprocessing"):
        setup.module.retire_idle(setup.path, retire=True)


@pytest.mark.parametrize("field,value", [
    ("user_name", "someone_else"), ("name", "another_job"),
    ("array_task_string", "0-7"), ("command", "/different/worker.sbatch"),
])
def test_ambiguous_ownership_or_array_expansion_aborts_before_cancellation(setup, field, value):
    setup.rows[4][field] = value
    with pytest.raises(ValueError):
        setup.module.retire_idle(setup.path, retire=True)
    assert_queue_unlocked(setup.queue)


def test_controller_non_owned_rows_and_pending_worker_selection(setup):
    setup.rows[0]["job_state"] = ["PENDING"]
    setup.rows.append({"job_id": 99, "array_job_id": 0, "job_state": ["RUNNING"]})
    setup.rows.append({"job_id": 888, "array_job_id": 888, "job_state": ["RUNNING"]})
    record = setup.module.retire_idle(setup.path)
    retired = {r["selector"] for r in record["plan"]["retire"]}
    assert retired == {"100_0", "100_5", "100_6"}
    assert all(r["states"] == ["RUNNING"] for r in record["plan"]["kept_idle"])


def test_confirmation_requires_terminal_accounting_and_no_live_completing_state(setup):
    module = setup.module
    selected = [{"selector": "100_4"}]
    row = copy.deepcopy(setup.rows[4])
    row["job_state"] = ["CANCELLED", "COMPLETING"]
    assert module.confirmed_terminal(selected, [row], {"100_4": "CANCELLED"}) == ([], ["100_4"])
    row["job_state"] = ["CANCELLED"]
    assert module.confirmed_terminal(selected, [row], {}) == ([], ["100_4"])
    assert module.confirmed_terminal(selected, [], {}) == ([], ["100_4"])
    assert module.confirmed_terminal(selected, [], {"100_4": "CANCELLED"}) == (["100_4"], [])


def test_queue_identity_mismatch_and_unknown_active_owner_fail_closed(setup):
    module = setup.module
    snapshot = module.queue_snapshot(setup.queue.path)
    snapshot["identity"] = {"stage": "another"}
    with pytest.raises(ValueError, match="queue identity"):
        module.retirement_plan(setup.manifest, snapshot, setup.rows, 4)
    snapshot = module.queue_snapshot(setup.queue.path)
    snapshot["active_claims"][0]["job_id"] = 123456
    with pytest.raises(ValueError, match="active claim"):
        module.retirement_plan(setup.manifest, snapshot, setup.rows, 4)


def test_closed_stdout_cannot_unlock_while_cancellation_is_unconfirmed(setup, monkeypatch):
    module = setup.module
    queries = 0

    def jobs(_):
        nonlocal queries
        queries += 1
        assert_queue_locked(setup.queue)
        return setup.rows if queries <= 2 else []

    def closed_stdout(*args, **kwargs):
        raise BrokenPipeError("monitoring pipe closed")

    monkeypatch.setattr(module, "query_jobs", jobs)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout="", stderr=""
    ))
    monkeypatch.setattr("builtins.print", closed_stdout)
    monkeypatch.setattr(module.time, "sleep", lambda _: assert_queue_locked(setup.queue))
    record = module.retire_idle(setup.path, retire=True)
    assert record["state"] == "confirmed" and queries == 3
    assert_queue_unlocked(setup.queue)


def test_exclusive_queue_lock_uses_writable_descriptor_for_nfs(setup, monkeypatch):
    original = setup.module.fcntl.flock
    observed = []

    def nfs_flock(handle, operation):
        if operation & fcntl.LOCK_EX:
            mode = fcntl.fcntl(handle, fcntl.F_GETFL) & os.O_ACCMODE
            assert mode == os.O_RDWR
            observed.append(mode)
        return original(handle, operation)

    monkeypatch.setattr(setup.module.fcntl, "flock", nfs_flock)
    with setup.module.queue_lock(setup.queue.path, 1):
        pass
    assert observed == [os.O_RDWR]


@contextmanager
def competing_lock(queue):
    acquired = threading.Event()
    release = threading.Event()

    def hold_lock():
        with queue.path.with_suffix(".sqlite3.lock").open("r+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            acquired.set()
            release.wait(timeout=5)
            fcntl.flock(handle, fcntl.LOCK_UN)

    thread = threading.Thread(target=hold_lock)
    thread.start()
    assert acquired.wait(timeout=2)
    try:
        yield release
    finally:
        release.set()
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_blocking_acquisition_waits_for_real_contender_and_disarms_deadline(setup, monkeypatch):
    module = setup.module
    original = module.fcntl.flock
    operations = []

    def observe(handle, operation):
        if threading.current_thread() is threading.main_thread():
            operations.append(operation)
        return original(handle, operation)

    previous_handler = signal.getsignal(signal.SIGALRM)
    with competing_lock(setup.queue) as release:
        timer = threading.Timer(0.1, release.set)
        timer.start()
        monkeypatch.setattr(module.fcntl, "flock", observe)
        started = time.monotonic()
        with module.queue_lock(setup.queue.path, 1):
            assert time.monotonic() - started >= 0.05
            assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
            assert signal.getsignal(signal.SIGALRM) == previous_handler
        timer.join()
    assert operations[0] == fcntl.LOCK_EX
    assert all(not operation & fcntl.LOCK_NB for operation in operations)
    assert_queue_unlocked(setup.queue)


def test_blocking_lock_timeout_preserves_queue_manifest_and_signal_state(setup):
    original_manifest = setup.path.read_bytes()
    original_queue = setup.queue.path.read_bytes()
    previous_handler = signal.getsignal(signal.SIGALRM)
    with competing_lock(setup.queue):
        with pytest.raises(TimeoutError, match="no cancellation attempted"):
            setup.module.retire_idle(setup.path, retire=True, lock_timeout=0.05)
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
        assert signal.getsignal(signal.SIGALRM) == previous_handler
    assert setup.path.read_bytes() == original_manifest
    assert setup.queue.path.read_bytes() == original_queue
    records = list((setup.tmp_path / "monitor").glob("retire-idle-*.json"))
    record = json.loads(records[0].read_text())
    assert record["actions"] == [] and record["state"] == "failed_before_retirement"
    assert_queue_unlocked(setup.queue)


def test_existing_alarm_is_rejected_without_changing_it(setup):
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def earlier_alarm(*args):
        pytest.fail("existing alarm must not fire in this test")

    try:
        signal.signal(signal.SIGALRM, earlier_alarm)
        signal.setitimer(signal.ITIMER_REAL, 5, 2)
        with pytest.raises(RuntimeError, match="active SIGALRM"):
            with setup.module.queue_lock(setup.queue.path, 1):
                pytest.fail("must not acquire the lock")
        remaining, interval = signal.getitimer(signal.ITIMER_REAL)
        assert 0 < remaining <= 5 and interval == 2
        assert signal.getsignal(signal.SIGALRM) == earlier_alarm
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
