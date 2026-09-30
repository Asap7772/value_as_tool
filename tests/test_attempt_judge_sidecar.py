from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    sidecar = importlib.import_module("attempt_judge_sidecar")
    queue = importlib.import_module("attempt_conditioning_queue")
    return sidecar, queue


def make_queue(queue_module, tmp_path, count):
    queue = queue_module.TaskQueue(tmp_path / "queues/judge.sqlite3")
    tasks = [{"task_id": f"run-{i:03d}", "run_id": f"run-{i:03d}"} for i in range(count)]
    queue.initialize(tasks, {"stage": "judge"})
    return queue


def rows(queue):
    with queue.transaction() as db:
        return {
            row["task_id"]: dict(row)
            for row in db.execute("SELECT task_id,status,worker,result FROM tasks")
        }


def unlocked(queue, task_ids):
    locks = [queue._task_lock(task_id) for task_id in task_ids]
    for lock in locks:
        if lock is not None:
            lock.close()
    return all(lock is not None for lock in locks)


def test_batches_finish_and_refill_alongside_regular_workers(modules, tmp_path):
    sidecar_module, queue_module = modules
    queue = make_queue(queue_module, tmp_path, 30)
    regular = queue.claim("7:regular:0")

    async def judge(claim):
        await asyncio.sleep(0.01)
        return "completed"

    sidecar = sidecar_module.Sidecar(
        queue, judge, session="7:sidecar-x", target=4, interval=0, max_errors=3
    )
    summary = asyncio.run(sidecar.run())
    assert rows(queue)["run-000"]["status"] == "running"
    queue.finish(regular, {"outcome": "completed", "worker": "7:regular"})

    state = rows(queue)
    assert summary["claimed"] == 29
    assert summary["outcomes"] == {"completed": 29}
    assert {row["status"] for row in state.values()} == {"complete"}
    assert state["run-000"]["worker"] == "7:regular:0"
    assert {row["worker"] for key, row in state.items() if key != "run-000"} == {"7:sidecar-x:0"}
    assert json.loads(state["run-001"]["result"]) == {
        "outcome": "completed",
        "worker": "7:sidecar-x",
    }
    assert unlocked(queue, state)
    assert queue.recover_abandoned() == []


def test_limit_stops_claiming(modules, tmp_path):
    sidecar_module, queue_module = modules
    queue = make_queue(queue_module, tmp_path, 12)

    async def judge(claim):
        return "completed"

    sidecar = sidecar_module.Sidecar(
        queue, judge, session="7:sidecar-x", target=4, interval=0, max_errors=3, limit=5
    )
    summary = asyncio.run(sidecar.run())
    assert summary["claimed"] == 5
    assert queue.counts() == {"pending": 7, "running": 0, "complete": 5, "failed": 0, "ready": 7}


def test_unsuccessful_outcomes_release_without_retry_then_drain(modules, tmp_path):
    sidecar_module, queue_module = modules
    queue = make_queue(queue_module, tmp_path, 10)

    async def judge(claim):
        if claim.task_id == "run-001":
            raise RuntimeError("server went away")
        return "invalidated" if claim.task_id == "run-000" else "completed"

    sidecar = sidecar_module.Sidecar(
        queue, judge, session="7:sidecar-x", target=2, interval=0, max_errors=2
    )
    summary = asyncio.run(sidecar.run())
    state = rows(queue)
    assert summary["claimed"] == 2
    assert summary["errors"] == 2
    assert summary["released_without_retry"] == ["run-000", "run-001"]
    assert state["run-000"]["status"] == state["run-001"]["status"] == "pending"
    assert state["run-000"]["worker"] is None
    assert "failed" not in {row["status"] for row in state.values()}
    assert unlocked(queue, state)


def test_failed_task_blocks_new_sidecar_claims(modules, tmp_path):
    sidecar_module, queue_module = modules
    queue = make_queue(queue_module, tmp_path, 6)
    queue.finish(queue.claim("7:regular:0"), {"error": "boom"}, failed=True)

    async def judge(claim):
        return "completed"

    sidecar = sidecar_module.Sidecar(
        queue, judge, session="7:sidecar-x", target=4, interval=0, max_errors=3
    )
    assert asyncio.run(sidecar.run())["claimed"] == 0
    assert queue.counts()["pending"] == 5


def test_cancel_releases_active_and_in_transit_claims(modules, tmp_path):
    sidecar_module, queue_module = modules
    queue = make_queue(queue_module, tmp_path, 5)

    async def judge(claim):
        await asyncio.sleep(3600)

    async def scenario():
        sidecar = sidecar_module.Sidecar(
            queue, judge, session="7:sidecar-x", target=3, interval=0, max_errors=3
        )
        runner = asyncio.create_task(sidecar.run())
        while len(sidecar.tasks) < 3:
            await asyncio.sleep(0.01)
        assert queue.counts()["running"] == 3
        sidecar.cancel()
        return await asyncio.wait_for(runner, 10)

    summary = asyncio.run(scenario())
    state = rows(queue)
    assert summary["claimed"] == 3
    assert {(row["status"], row["worker"]) for row in state.values()} == {("pending", None)}
    assert unlocked(queue, state)


def test_batch_claims_ahead_but_bounds_concurrent_requests(modules, tmp_path):
    sidecar_module, queue_module = modules
    queue = make_queue(queue_module, tmp_path, 20)
    active = peak = 0

    async def judge(claim):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return "completed"

    def sidecar(session):
        return sidecar_module.Sidecar(
            queue, judge, session=session, target=3, batch=9, interval=0, max_errors=3
        )

    claims, stats = sidecar("7:sidecar-x").exchange()
    assert stats["claimed"] == stats["held"] == 9
    assert queue.counts()["running"] == 9
    for claim in claims:
        claim.close()
    assert len(queue.recover_abandoned()) == 9

    summary = asyncio.run(sidecar("7:sidecar-y").run())
    assert summary["claimed"] == 20
    assert peak == 3
    assert queue.counts()["complete"] == 20
