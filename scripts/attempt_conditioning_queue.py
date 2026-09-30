"""Balanced durable task claims, without result-tree scans or time-based leases.

SQLite uses rollback journaling, never WAL. A POSIX flock surrounds every short
transaction; a separate lock remains held for each in-progress model task.
Recovery requires acquiring that task lock, so long requests are not expired.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import sqlite3
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(UTC).isoformat()


def balanced_order(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        key = tuple(str(task.get(k, "")) for k in ("benchmark", "method", "mode", "kind"))
        groups[key].append(task)
    for group in groups.values():
        group.sort(
            key=lambda t: (
                -float(t.get("estimated_tokens", 0)),
                str(t.get("problem_id", "")),
                t.get("seed", 0),
                t["task_id"],
            )
        )
    ordered = []
    for index in range(max((len(g) for g in groups.values()), default=0)):
        for key in sorted(groups):
            if index < len(groups[key]):
                ordered.append(groups[key][index])
    return ordered


@dataclass
class Claim:
    task_id: str
    payload: dict[str, Any]
    worker: str
    lock: Any

    def close(self) -> None:
        self.lock.close()


class TaskQueue:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.locks = self.path.parent / (self.path.name + ".claims")
        self.locks.mkdir(exist_ok=True)

    @contextmanager
    def transaction(self):
        with self.path.with_suffix(self.path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            connection = sqlite3.connect(self.path, timeout=30)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA foreign_keys=ON")
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

    def initialize(self, tasks: list[dict[str, Any]], identity: dict[str, Any]) -> None:
        ordered = balanced_order(tasks)
        task_ids = {t["task_id"] for t in ordered}
        if len(task_ids) != len(ordered):
            raise ValueError("duplicate queue task IDs")
        for task in ordered:
            if not set(task.get("dependencies", [])) <= task_ids:
                raise ValueError(f"unknown dependency for {task['task_id']}")
        # Detect dependency cycles before creating a queue that can never drain.
        resolved: set[str] = set()
        while len(resolved) < len(ordered):
            ready = {
                t["task_id"] for t in ordered if set(t.get("dependencies", [])) <= resolved
            } - resolved
            if not ready:
                raise ValueError("cyclic queue dependencies")
            resolved.update(ready)
        signature = json.dumps({"identity": identity, "tasks": ordered}, sort_keys=True)
        digest = hashlib.sha256(signature.encode()).hexdigest()
        with self.transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT)")
            previous = db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
            if previous:
                if previous[0] != digest:
                    raise ValueError("queue identity or task list changed")
                return
            db.execute("""CREATE TABLE tasks (task_id TEXT PRIMARY KEY,priority INTEGER,
                payload TEXT,status TEXT NOT NULL DEFAULT 'pending',worker TEXT,
                started_at TEXT,completed_at TEXT,result TEXT)""")
            db.execute("CREATE INDEX ready_tasks ON tasks(status,priority)")
            db.execute("""CREATE TABLE dependencies (task_id TEXT,dependency TEXT,
                PRIMARY KEY(task_id,dependency),FOREIGN KEY(task_id) REFERENCES tasks(task_id),
                FOREIGN KEY(dependency) REFERENCES tasks(task_id))""")
            db.executemany(
                "INSERT INTO tasks(task_id,priority,payload) VALUES(?,?,?)",
                [(t["task_id"], i, json.dumps(t, sort_keys=True)) for i, t in enumerate(ordered)],
            )
            db.executemany(
                "INSERT INTO dependencies VALUES(?,?)",
                [
                    (t["task_id"], dependency)
                    for t in ordered
                    for dependency in t.get("dependencies", [])
                ],
            )
            db.execute("INSERT INTO metadata VALUES('signature',?)", (digest,))
            db.execute("INSERT INTO metadata VALUES('identity',?)", (json.dumps(identity),))

    def _task_lock(self, task_id: str):
        name = hashlib.sha256(task_id.encode()).hexdigest()
        handle = (self.locks / name).open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return None
        return handle

    def claim(self, worker: str) -> Claim | None:
        with self.transaction() as db:
            if db.execute("SELECT 1 FROM tasks WHERE status='failed' LIMIT 1").fetchone():
                return None
            row = db.execute("""SELECT t.* FROM tasks t WHERE t.status='pending'
                AND NOT EXISTS(SELECT 1 FROM dependencies d JOIN tasks p
                ON d.dependency=p.task_id WHERE d.task_id=t.task_id AND p.status!='complete')
                ORDER BY t.priority LIMIT 1""").fetchone()
            if row is None:
                return None
            lock = self._task_lock(row["task_id"])
            if lock is None:
                return None
            try:
                db.execute(
                    "UPDATE tasks SET status='running',worker=?,started_at=? WHERE task_id=?",
                    (worker, now(), row["task_id"]),
                )
            except BaseException:
                lock.close()
                raise
            return Claim(row["task_id"], json.loads(row["payload"]), worker, lock)

    def finish(self, claim: Claim, result: dict[str, Any], *, failed: bool = False) -> None:
        try:
            with self.transaction() as db:
                cursor = db.execute(
                    """UPDATE tasks SET status=?,result=?,completed_at=?
                    WHERE task_id=? AND status='running' AND worker=?""",
                    (
                        "failed" if failed else "complete",
                        json.dumps(result, sort_keys=True),
                        now(),
                        claim.task_id,
                        claim.worker,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("claim ownership changed")
        finally:
            claim.close()

    def release(self, claim: Claim) -> None:
        try:
            with self.transaction() as db:
                db.execute(
                    """UPDATE tasks SET status='pending',worker=NULL,started_at=NULL
                    WHERE task_id=? AND status='running' AND worker=?""",
                    (claim.task_id, claim.worker),
                )
        finally:
            claim.close()

    def recover_abandoned(self) -> list[str]:
        recovered = []
        with self.transaction() as db:
            for row in db.execute("SELECT task_id FROM tasks WHERE status='running'").fetchall():
                lock = self._task_lock(row[0])
                if lock is None:
                    continue
                try:
                    db.execute(
                        """UPDATE tasks SET status='pending',worker=NULL,started_at=NULL
                        WHERE task_id=?""",
                        (row[0],),
                    )
                    recovered.append(row[0])
                finally:
                    lock.close()
        return recovered

    def counts(self) -> dict[str, int]:
        with self.transaction() as db:
            result = dict(db.execute("SELECT status,count(*) FROM tasks GROUP BY status"))
            result["ready"] = db.execute("""SELECT count(*) FROM tasks t WHERE t.status='pending'
                AND NOT EXISTS(SELECT 1 FROM dependencies d JOIN tasks p ON d.dependency=p.task_id
                WHERE d.task_id=t.task_id AND p.status!='complete')""").fetchone()[0]
            return {
                k: int(result.get(k, 0))
                for k in ("pending", "running", "complete", "failed", "ready")
            }

    def results(self) -> list[dict[str, Any]]:
        with self.transaction() as db:
            return [
                {
                    "task_id": r["task_id"],
                    "status": r["status"],
                    "result": json.loads(r["result"]) if r["result"] else None,
                }
                for r in db.execute("SELECT task_id,status,result FROM tasks ORDER BY priority")
            ]
