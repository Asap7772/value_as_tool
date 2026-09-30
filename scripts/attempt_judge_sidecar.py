"""Batched sidecar judge inside a running attempt-conditioning judge allocation.

The queue's global lock admits roughly eleven transactions per second on NFS,
and a regular worker spends two of them on every task. A sidecar runs inside an
existing judge worker allocation (``srun --overlap --jobid=...``), adopts the
sibling worker's environment and local judge server, and keeps several requests
in flight. Each cycle is one transaction under the pinned TaskQueue lock, using
the pinned per-task claim locks and SQL, that finishes every completed task and
refills the in-flight set. Judging calls the pinned ``pipeline._judge_one``.

Unsuccessful outcomes release the task to the regular workers instead of
failing the queue, and the sidecar never reclaims a task it released. Creating
the drain file stops new claims; SIGTERM/SIGINT cancel and release active work.
Nothing in the launch manifest or immutable source changes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import sys
import threading
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ADOPTED = "VALUE_AS_TOOL_JUDGE_SIDECAR"
WORKER_SCRIPT = b"attempt_conditioning_worker.py"
RETRY_OUTCOMES = {"busy", "claimed_elsewhere"}
FAILED_OUTCOMES = {"failed", "invalidated", "blocked", "solve_missing"}


def log(**event: Any) -> None:
    print(json.dumps({"utc": datetime.now(UTC).isoformat(), **event}), flush=True)


def sibling_worker(job_id: str, manifest: Path) -> tuple[str, dict[str, str], str]:
    """Return the Python binary, environment, and cwd of this job's judge worker."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            if not any(arg.endswith(WORKER_SCRIPT) for arg in argv):
                continue
            environ = dict(
                os.fsdecode(item).split("=", 1)
                for item in (entry / "environ").read_bytes().split(b"\0")
                if b"=" in item
            )
            cwd = os.readlink(entry / "cwd")
        except OSError:
            continue
        args = [os.fsdecode(arg) for arg in argv]
        try:
            stage = args[args.index("--stage") + 1]
            launch = Path(args[args.index("--manifest") + 1]).resolve()
        except (ValueError, IndexError):
            continue
        if (
            environ.get("SLURM_JOB_ID") == job_id
            and environ.get("VALUE_AS_TOOL_JUDGE_BASE_URL")
            and stage == "judge"
            and launch == manifest.resolve()
        ):
            return args[0], environ, cwd
    raise SystemExit(f"no judge queue worker for this launch is running in job {job_id}")


def adopt(manifest: Path) -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise SystemExit("run inside a judge worker allocation: srun --overlap --jobid=...")
    python, environ, cwd = sibling_worker(job_id, manifest)
    environ[ADOPTED] = "1"
    os.chdir(cwd)
    os.execve(python, [python, str(Path(__file__).resolve()), *sys.argv[1:]], environ)


def import_pinned(source: Path) -> None:
    """Resolve the queue and controller modules from the immutable launch source."""
    here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != here]
    sys.path.insert(0, str(source / "scripts"))


class Sidecar:
    def __init__(
        self,
        queue: Any,
        judge: Callable[[Any], Awaitable[str]],
        *,
        session: str,
        target: int,
        interval: float,
        max_errors: int,
        batch: int | None = None,
        drain_file: Path | None = None,
        limit: int | None = None,
    ):
        self.queue = queue
        self.judge = judge
        self.session = session
        self.worker = f"{session}:0"
        self.target = target
        # Lock turns are scarce, so each one may claim more tasks than run at once.
        self.batch = max(batch or target, target)
        self.slots = asyncio.Semaphore(target)
        self.interval = interval
        self.max_errors = max_errors
        self.drain_file = drain_file
        self.limit = limit
        self.lock = threading.Lock()
        self.finished: list[tuple[Any, dict[str, Any]]] = []
        self.released: list[Any] = []
        self.avoid: set[str] = set()
        self.tasks: set[asyncio.Task] = set()
        self.held = 0
        self.claimed = 0
        self.errors = 0
        self.outcomes: Counter[str] = Counter()
        self.lost: list[str] = []
        self.draining = False
        self.exhausted = False
        self.cancelled = False

    def exchange(self) -> tuple[list[Any], dict[str, Any]]:
        """Finish, release, and refill in one pinned-protocol queue transaction."""
        from attempt_conditioning_queue import Claim, now

        started = time.monotonic()
        finished: list[tuple[Any, dict[str, Any]]] = []
        released: list[Any] = []
        claims: list[Any] = []
        try:
            with self.queue.transaction() as db:
                acquired = time.monotonic()
                # Take the batch only once the lock is held, so completions that
                # arrive while waiting for it are included.
                with self.lock:
                    finished, self.finished = self.finished, []
                    released, self.released = self.released, []
                remaining = self.held - len(finished) - len(released)
                # Errors may have arrived while this transaction waited for the lock.
                stop = self.draining or self.exhausted or self.errors >= self.max_errors
                capacity = 0 if stop else self.batch - remaining
                if self.limit is not None:
                    capacity = min(capacity, self.limit - self.claimed)
                lost = []
                for claim, result in finished:
                    cursor = db.execute(
                        """UPDATE tasks SET status=?,result=?,completed_at=?
                        WHERE task_id=? AND status='running' AND worker=?""",
                        (
                            "complete",
                            json.dumps(result, sort_keys=True),
                            now(),
                            claim.task_id,
                            claim.worker,
                        ),
                    )
                    if cursor.rowcount != 1:
                        lost.append(claim.task_id)
                for claim in released:
                    cursor = db.execute(
                        """UPDATE tasks SET status='pending',worker=NULL,started_at=NULL
                        WHERE task_id=? AND status='running' AND worker=?""",
                        (claim.task_id, claim.worker),
                    )
                    if cursor.rowcount != 1:
                        lost.append(claim.task_id)
                failed = db.execute("SELECT 1 FROM tasks WHERE status='failed' LIMIT 1").fetchone()
                if capacity > 0 and not lost and not failed:
                    rows = db.execute(
                        """SELECT t.* FROM tasks t WHERE t.status='pending'
                        AND NOT EXISTS(SELECT 1 FROM dependencies d JOIN tasks p
                        ON d.dependency=p.task_id WHERE d.task_id=t.task_id
                        AND p.status!='complete') ORDER BY t.priority LIMIT ?""",
                        (capacity + len(self.avoid),),
                    ).fetchall()
                    for row in rows:
                        if len(claims) == capacity:
                            break
                        if row["task_id"] in self.avoid:
                            continue
                        lock = self.queue._task_lock(row["task_id"])
                        if lock is None:
                            continue
                        claims.append(
                            Claim(row["task_id"], json.loads(row["payload"]), self.worker, lock)
                        )
                        db.execute(
                            """UPDATE tasks SET status='running',worker=?,started_at=?
                            WHERE task_id=?""",
                            (self.worker, now(), row["task_id"]),
                        )
                    if len(rows) < capacity + len(self.avoid):
                        # Too few ready tasks: leave the tail to the regular workers.
                        self.exhausted = True
                if failed or (self.limit is not None and self.claimed + len(claims) >= self.limit):
                    self.exhausted = True
        except BaseException:
            for claim in claims:
                claim.close()
            with self.lock:
                self.finished[:0] = finished
                self.released[:0] = released
            raise
        # Per-task locks close only after the commit, exactly like TaskQueue.finish.
        for claim, _ in finished:
            claim.close()
        for claim in released:
            claim.close()
        self.held = remaining + len(claims)
        self.claimed += len(claims)
        if lost:
            self.lost.extend(lost)
            self.draining = True
        return claims, {
            "finished": len(finished),
            "released": len(released),
            "claimed": len(claims),
            "held": self.held,
            "lost": lost,
            "failed_task_present": bool(failed),
            "lock_wait_seconds": round(acquired - started, 3),
            "hold_seconds": round(time.monotonic() - acquired, 3),
        }

    async def _judge(self, claim: Any) -> None:
        try:
            async with self.slots:
                outcome = await self.judge(claim)
        except asyncio.CancelledError:
            with self.lock:
                self.released.append(claim)
            raise
        except Exception as error:
            self.errors += 1
            self.avoid.add(claim.task_id)
            with self.lock:
                self.released.append(claim)
            log(event="error", task_id=claim.task_id, error=f"{type(error).__name__}: {error}")
            return
        self.outcomes[outcome] += 1
        if outcome in RETRY_OUTCOMES or outcome in FAILED_OUTCOMES:
            self.errors += outcome in FAILED_OUTCOMES
            self.avoid.add(claim.task_id)
            with self.lock:
                self.released.append(claim)
            log(event="released", task_id=claim.task_id, outcome=outcome)
            return
        with self.lock:
            self.finished.append((claim, {"outcome": outcome, "worker": self.session}))
        log(event="judged", task_id=claim.task_id, outcome=outcome)

    def cancel(self) -> None:
        self.cancelled = self.draining = True
        for task in self.tasks:
            task.cancel()

    async def run(self) -> dict[str, Any]:
        last = -float("inf")
        while True:
            self.tasks = {task for task in self.tasks if not task.done()}
            if self.errors >= self.max_errors or (self.drain_file and self.drain_file.exists()):
                self.draining = True
            pending_work = bool(self.finished or self.released)
            if not self.tasks and not pending_work and (self.draining or self.exhausted):
                break
            claiming = not (self.draining or self.exhausted)
            wait = self.interval - (time.monotonic() - last)
            # While claiming, stay queued for the lock: completions that arrive
            # during the wait join the batch taken once the lock is held.
            if (pending_work or claiming) and wait <= 0:
                try:
                    claims, stats = await asyncio.to_thread(self.exchange)
                except Exception as error:
                    log(event="exchange_error", error=f"{type(error).__name__}: {error}")
                    self.errors += 1
                    last = time.monotonic()
                    continue
                last = time.monotonic()
                if self.cancelled:
                    # A signal arrived during the transaction: return new claims at once.
                    with self.lock:
                        self.released.extend(claims)
                    claims = []
                for claim in claims:
                    self.tasks.add(asyncio.create_task(self._judge(claim)))
                log(event="exchange", **stats)
                continue
            timeout = max(1.0, wait)
            if self.tasks:
                await asyncio.wait(self.tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            else:
                await asyncio.sleep(timeout)
        return {
            "session": self.session,
            "claimed": self.claimed,
            "outcomes": dict(self.outcomes),
            "errors": self.errors,
            "lost": self.lost,
            "released_without_retry": sorted(self.avoid),
        }


async def pinned_judge(manifest: dict[str, Any]) -> tuple[Callable[[Any], Awaitable[str]], Any]:
    """Build the judge path exactly as attempt_conditioning_worker.run_worker does."""
    from value_as_tool import pipeline
    from value_as_tool.benchmarks import QEDPromptSet
    from value_as_tool.client import OpenAIChatClient
    from value_as_tool.judging import JudgeRunner
    from value_as_tool.tokenization import HuggingFaceTokenCounter

    context = pipeline.load_context(Path(manifest["config"]))
    model = context.config.models.judge
    client = OpenAIChatClient(
        context.config.models.operational_base_url("judge"),
        api_key=os.environ.get(model.api_key_env),
        timeout=context.config.runtime.request_timeout_seconds,
    )
    counter = HuggingFaceTokenCounter(
        pipeline._model_entry(context, "judge")["path"], enable_thinking=False
    )
    schedule = pipeline._load_schedule(context)
    items = {item.run_id: item for item in schedule}
    solve_store = pipeline._store(context, schedule, "solve", initialize=False)
    judge_store = pipeline._store(context, schedule, "judge")
    runner = JudgeRunner(
        QEDPromptSet(),
        max_tokens=context.config.evaluation.judge_output_tokens,
        token_counter=counter,
        reasoning_effort=context.config.evaluation.judge_reasoning_effort,
        model=model.name,
    )

    async def judge(claim: Any) -> str:
        return await pipeline._judge_one(
            items[claim.payload["run_id"]],
            context=context,
            solve_store=solve_store,
            judge_store=judge_store,
            client=client,
            runner=runner,
        )

    return judge, client


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text())
    source = Path(manifest["source"]).resolve()
    import_pinned(source)
    from attempt_conditioning_queue import TaskQueue

    live_queue = Path(manifest["queues"]["judge"]["path"]).resolve()
    client = None
    if args.dry_run_queue:
        if args.dry_run_queue.resolve() == live_queue:
            raise SystemExit("--dry-run-queue must be a copy, never the live judge queue")
        queue = TaskQueue(args.dry_run_queue)

        async def judge(claim: Any) -> str:
            await asyncio.sleep(random.uniform(0, args.fake_judge_seconds))
            return "completed"
    else:
        from submit_attempt_conditioning import verify_source

        verify_source(manifest)
        if manifest["state"] != "running" or manifest["stage"] != "judge":
            raise SystemExit("the launch is not running its judge stage")
        import attempt_conditioning_queue
        import submit_attempt_conditioning

        from value_as_tool import pipeline

        for module in (attempt_conditioning_queue, submit_attempt_conditioning, pipeline):
            if not Path(module.__file__).resolve().is_relative_to(source):
                raise SystemExit(f"{module.__name__} was not imported from the pinned source")
        queue = TaskQueue(live_queue)
        judge, client = await pinned_judge(manifest)
    session = f"{os.environ.get('SLURM_JOB_ID', 'local')}:sidecar-{uuid.uuid4().hex}"
    sidecar = Sidecar(
        queue,
        judge,
        session=session,
        target=args.target,
        batch=args.batch,
        interval=args.interval,
        max_errors=args.max_errors,
        drain_file=args.drain_file
        or args.manifest.resolve().parent / "monitor/judge-sidecar.drain",
        limit=args.limit,
    )
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, sidecar.cancel)
    log(
        event="start",
        session=session,
        target=args.target,
        batch=args.batch,
        base_url=os.environ.get("VALUE_AS_TOOL_JUDGE_BASE_URL"),
    )
    try:
        return await sidecar.run()
    finally:
        if client is not None:
            await client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", type=int, default=12, help="concurrent judge requests")
    parser.add_argument("--batch", type=int, default=36, help="claimed tasks held at once")
    parser.add_argument(
        "--interval", type=float, default=10, help="minimum seconds between transactions"
    )
    parser.add_argument("--max-errors", type=int, default=3)
    parser.add_argument("--drain-file", type=Path)
    parser.add_argument("--limit", type=int, help="stop claiming after this many tasks")
    parser.add_argument(
        "--dry-run-queue", type=Path, help="queue copy to exercise with a fake judge"
    )
    parser.add_argument("--fake-judge-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if not 1 <= args.target <= 14:
        parser.error("target must be between 1 and 14 (the judge server admits 16 sequences)")
    if not args.target <= args.batch <= 100:
        parser.error("batch must be between target and 100")
    if not args.dry_run_queue and os.environ.get(ADOPTED) != "1":
        adopt(args.manifest)
    log(event="exit", **asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
