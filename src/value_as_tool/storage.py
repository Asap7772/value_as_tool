"""Crash-safe experiment artifacts and per-trajectory checkpoints."""

from __future__ import annotations

import dataclasses
import enum
import fcntl
import hashlib
import json
import os
import re
import socket
import tempfile
import threading
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any


class ArtifactError(RuntimeError):
    pass


class ArtifactMismatchError(ArtifactError):
    """An existing immutable artifact belongs to another experiment."""


class RunClaimedError(ArtifactError):
    """Another live process owns the trajectory lock."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_write_bytes(path: str | Path, content: bytes) -> None:
    """Atomically replace a file and fsync both it and its parent directory."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(path: str | Path, content: str) -> None:
    atomic_write_bytes(path, content.encode("utf-8"))


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, default=_json_default, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one complete JSON record under an advisory file lock."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(record) + "\n").encode("utf-8")
    descriptor = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"expected a JSON object at {path}:{line_number}")
                records.append(value)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return records


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
_UNSET = object()


def _validate_run_id(run_id: str) -> str:
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError(
            "run_id must be 1-192 characters using only letters, digits, '.', '_' or '-'"
        )
    return run_id


def _active_requests(checkpoint: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Normalize current and early singular checkpoint request markers."""

    if not checkpoint:
        return {}
    current = checkpoint.get("in_flight_requests")
    if isinstance(current, Mapping):
        return {
            str(request_id): dict(request)
            for request_id, request in current.items()
            if isinstance(request, Mapping)
        }
    legacy = checkpoint.get("in_flight_request")
    if isinstance(legacy, Mapping):
        request_id = legacy.get("request_id")
        if isinstance(request_id, str) and request_id:
            return {request_id: dict(legacy)}
    return {}


class ClaimAction(enum.StrEnum):
    NEW = "new"
    RESUME = "resume"
    COMPLETE = "complete"


@dataclass(slots=True)
class RunClaim:
    action: ClaimAction
    run_id: str
    handle: AttemptHandle | None = None
    result: dict[str, Any] | None = None

    @property
    def should_run(self) -> bool:
        return self.handle is not None

    @property
    def checkpoint(self) -> dict[str, Any] | None:
        return self.handle.load_checkpoint() if self.handle is not None else None


class AttemptHandle:
    """Exclusive, resumable ownership of one trajectory attempt.

    Call ``begin_request`` immediately before an external model request and
    ``complete_request`` only after exact provider usage has been persisted.
    If the process dies between those writes, the next claimant invalidates
    this attempt because its generated-token usage is unknowable.
    """

    def __init__(
        self,
        store: ArtifactStore,
        run_id: str,
        attempt: int,
        lock_handle: IO[bytes],
    ) -> None:
        self.store = store
        self.run_id = run_id
        self.attempt = attempt
        self._lock_handle = lock_handle
        self._closed = False
        self._thread_lock = threading.RLock()

    @property
    def attempt_id(self) -> str:
        return f"{self.attempt:06d}"

    @property
    def path(self) -> Path:
        return self.store.run_dir(self.run_id) / "attempts" / self.attempt_id

    @property
    def checkpoint_path(self) -> Path:
        return self.path / "checkpoint.json"

    def _ensure_open(self) -> None:
        if self._closed:
            raise ArtifactError("attempt handle is closed")

    def load_checkpoint(self) -> dict[str, Any] | None:
        self._ensure_open()
        if not self.checkpoint_path.exists():
            return None
        return read_json(self.checkpoint_path)

    def save_checkpoint(
        self,
        state: str,
        payload: Mapping[str, Any],
        *,
        in_flight_request: Mapping[str, Any] | None | object = _UNSET,
        in_flight_requests: Mapping[str, Mapping[str, Any]] | object = _UNSET,
    ) -> dict[str, Any]:
        with self._thread_lock:
            self._ensure_open()
            previous = self.load_checkpoint()
            sequence = int(previous.get("sequence", 0)) + 1 if previous else 1
            active = _active_requests(previous)
            if in_flight_requests is not _UNSET:
                if not isinstance(in_flight_requests, Mapping):
                    raise TypeError("in_flight_requests must be a mapping")
                active = {
                    str(request_id): dict(request)
                    for request_id, request in in_flight_requests.items()
                }
            elif in_flight_request is not _UNSET:
                active = {}
                if in_flight_request is not None:
                    request = dict(in_flight_request)  # type: ignore[arg-type]
                    request_id = request.get("request_id")
                    if not isinstance(request_id, str) or not request_id:
                        raise ArtifactError("in_flight_request requires a non-empty request_id")
                    active[request_id] = request
            checkpoint = {
                "schema_version": 1,
                "run_id": self.run_id,
                "attempt": self.attempt,
                "config_fingerprint": self.store.config_fingerprint,
                "schedule_fingerprint": self.store.schedule_fingerprint,
                "sequence": sequence,
                "state": state,
                "payload": dict(payload),
                "in_flight_requests": active,
                # Singular compatibility field for early artifacts/readers.
                "in_flight_request": next(iter(active.values()), None),
                "updated_at": utc_now(),
            }
            atomic_write_json(self.checkpoint_path, checkpoint)
            self.store._write_status(
                self.run_id,
                {
                    "status": "active",
                    "attempt": self.attempt,
                    "state": state,
                    "in_flight": bool(active),
                    "in_flight_count": len(active),
                    "updated_at": checkpoint["updated_at"],
                },
            )
            return checkpoint

    checkpoint = save_checkpoint

    def begin_request(
        self,
        request_id: str,
        *,
        role: str,
        max_tokens: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._thread_lock:
            if max_tokens is None and metadata is not None:
                candidate_max = metadata.get("max_tokens")
                if isinstance(candidate_max, int) and not isinstance(candidate_max, bool):
                    max_tokens = candidate_max
            if max_tokens is not None and (
                isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0
            ):
                raise ValueError("max_tokens must be a positive integer when provided")
            previous = self.load_checkpoint()
            active = _active_requests(previous)
            if request_id in active:
                raise ArtifactError(f"request {request_id!r} is already in flight")
            active[request_id] = {
                "request_id": request_id,
                "role": role,
                "max_tokens": max_tokens,
                "started_at": utc_now(),
                "metadata": dict(metadata or {}),
            }
            return self.save_checkpoint(
                str(previous.get("state", "request")) if previous else "request",
                previous.get("payload", {}) if previous else {},
                in_flight_requests=active,
            )

    def complete_request(
        self,
        request_id: str,
        *,
        state: str,
        payload: Mapping[str, Any],
        usage: Mapping[str, Any],
    ) -> dict[str, Any]:
        with self._thread_lock:
            previous = self.load_checkpoint()
            active_requests = _active_requests(previous)
            active = active_requests.pop(request_id, None)
            if active is None:
                raise ArtifactError(f"request {request_id!r} is not active")
            completion_tokens = usage.get("completion_tokens")
            prompt_tokens = usage.get("prompt_tokens")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (completion_tokens, prompt_tokens)
            ):
                raise ArtifactError(
                    "exact non-negative prompt_tokens and completion_tokens are required"
                )
            event = {
                "event": "request_completed",
                "run_id": self.run_id,
                "attempt": self.attempt,
                "request_id": request_id,
                "role": active.get("role"),
                "usage": dict(usage),
                # Per-request latency and caps for throughput analysis.
                "started_at": active.get("started_at"),
                "max_tokens": active.get("max_tokens"),
                "metadata": active.get("metadata") or {},
                "at": utc_now(),
            }
            append_jsonl(self.path / "events.jsonl", event)
            return self.save_checkpoint(
                state,
                payload,
                in_flight_requests=active_requests,
            )

    def finalize(self, result: Mapping[str, Any]) -> dict[str, Any]:
        self._ensure_open()
        checkpoint = self.load_checkpoint()
        if _active_requests(checkpoint):
            raise ArtifactError("cannot finalize while requests are in flight")
        record = {
            **dict(result),
            "schema_version": 1,
            "run_id": self.run_id,
            "attempt": self.attempt,
            "config_fingerprint": self.store.config_fingerprint,
            "schedule_fingerprint": self.store.schedule_fingerprint,
            "completed_at": utc_now(),
        }
        atomic_write_json(self.path / "result.json", record)
        atomic_write_json(self.store.run_dir(self.run_id) / "result.json", record)
        self.store._write_status(
            self.run_id,
            {
                "status": "completed",
                "attempt": self.attempt,
                "updated_at": record["completed_at"],
            },
        )
        append_jsonl(
            self.path / "events.jsonl",
            {"event": "completed", "at": record["completed_at"], "run_id": self.run_id},
        )
        self.close()
        return record

    def invalidate(
        self,
        reason: str,
        *,
        unknown_usage: bool | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_open()
        record = self.store._invalidate_locked(
            self.run_id,
            self.attempt,
            reason,
            unknown_usage=unknown_usage,
            details=details,
        )
        self.close()
        return record

    def close(self) -> None:
        if self._closed:
            return
        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
        self._lock_handle.close()
        self._closed = True

    def __enter__(self) -> AttemptHandle:
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        # Leave a safe checkpoint resumable. An in-flight request is
        # automatically recognized and invalidated by the next claimant.
        self.close()


class ArtifactStore:
    """A filesystem-backed store safe for independent Slurm shards."""

    def __init__(
        self,
        root: str | Path,
        config_fingerprint: str | None = None,
        schedule_fingerprint: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.config_fingerprint = config_fingerprint
        self.schedule_fingerprint = schedule_fingerprint

    @property
    def runs_root(self) -> Path:
        return self.root / "runs"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def initialize(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create or validate the experiment manifest.

        The stored identity fields are immutable; resuming with a different
        config or schedule fails before any trajectory is touched.
        """

        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        if config is not None:
            derived = fingerprint(config)
            if self.config_fingerprint is None:
                self.config_fingerprint = derived
            elif self.config_fingerprint != derived:
                raise ArtifactMismatchError("supplied config does not match config_fingerprint")
        proposed = {
            "schema_version": 1,
            "config_fingerprint": self.config_fingerprint,
            "schedule_fingerprint": self.schedule_fingerprint,
            "config": dict(config) if config is not None else None,
            "metadata": dict(metadata or {}),
        }
        lock = self.root / ".manifest.lock"
        with lock.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            if self.manifest_path.exists():
                existing = read_json(self.manifest_path)
                for key in ("schema_version", "config_fingerprint", "schedule_fingerprint"):
                    expected = proposed[key]
                    if expected is not None and existing.get(key) != expected:
                        actual = existing.get(key)
                        raise ArtifactMismatchError(
                            f"artifact manifest {key} mismatch: {actual!r} != {expected!r}"
                        )
                return existing
            proposed["created_at"] = utc_now()
            atomic_write_json(self.manifest_path, proposed)
            return proposed

    def run_dir(self, run_id: str) -> Path:
        return self.runs_root / _validate_run_id(run_id)

    def _write_status(self, run_id: str, status: Mapping[str, Any]) -> None:
        record = {
            "schema_version": 1,
            "run_id": run_id,
            "config_fingerprint": self.config_fingerprint,
            "schedule_fingerprint": self.schedule_fingerprint,
            **dict(status),
        }
        atomic_write_json(self.run_dir(run_id) / "status.json", record)

    def _acquire_run_lock(self, run_id: str, *, blocking: bool) -> IO[bytes]:
        path = self.run_dir(run_id)
        path.mkdir(parents=True, exist_ok=True)
        handle = (path / ".lock").open("a+b")
        operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle.fileno(), operation)
        except BlockingIOError as error:
            handle.close()
            raise RunClaimedError(f"trajectory {run_id!r} is already claimed") from error
        return handle

    def claim(
        self,
        run_id: str,
        *,
        identity: Mapping[str, Any] | None = None,
        resume: bool = True,
        blocking: bool = False,
    ) -> RunClaim:
        """Claim a run, resume a safe checkpoint, or return its final result."""

        run_id = _validate_run_id(run_id)
        lock_handle = self._acquire_run_lock(run_id, blocking=blocking)
        try:
            run_path = self.run_dir(run_id)
            identity_path = run_path / "run.json"
            supplied_identity = dict(identity or {})
            identity_record = {
                "schema_version": 1,
                "run_id": run_id,
                "config_fingerprint": self.config_fingerprint,
                "schedule_fingerprint": self.schedule_fingerprint,
                "identity": supplied_identity,
                "identity_fingerprint": fingerprint(supplied_identity),
            }
            if identity_path.exists():
                existing_identity = read_json(identity_path)
                identity_keys = ["config_fingerprint", "schedule_fingerprint"]
                if identity is not None:
                    identity_keys.append("identity_fingerprint")
                for key in identity_keys:
                    expected = identity_record[key]
                    if expected is not None and existing_identity.get(key) != expected:
                        raise ArtifactMismatchError(f"run {run_id!r} {key} mismatch")
            else:
                identity_record["created_at"] = utc_now()
                atomic_write_json(identity_path, identity_record)

            result_path = run_path / "result.json"
            if result_path.exists():
                result = read_json(result_path)
                lock_handle.close()
                return RunClaim(ClaimAction.COMPLETE, run_id, result=result)

            attempts_root = run_path / "attempts"
            attempts_root.mkdir(parents=True, exist_ok=True)
            attempts = sorted(
                int(path.name)
                for path in attempts_root.iterdir()
                if path.is_dir() and path.name.isdigit()
            )
            if attempts:
                latest = attempts[-1]
                latest_path = attempts_root / f"{latest:06d}"
                invalid = (latest_path / "invalidation.json").exists()
                checkpoint_path = latest_path / "checkpoint.json"
                checkpoint = read_json(checkpoint_path) if checkpoint_path.exists() else None
                if not invalid and _active_requests(checkpoint):
                    self._invalidate_locked(
                        run_id,
                        latest,
                        "interrupted_request_with_unknown_usage",
                        unknown_usage=True,
                        details={"recovered_by": f"{socket.gethostname()}:{os.getpid()}"},
                    )
                    invalid = True
                if not invalid and resume:
                    handle = AttemptHandle(self, run_id, latest, lock_handle)
                    return RunClaim(ClaimAction.RESUME, run_id, handle=handle)
                if not invalid:
                    self._invalidate_locked(
                        run_id,
                        latest,
                        "resume_disabled",
                        unknown_usage=False,
                    )
                attempt = latest + 1
            else:
                attempt = 1

            attempt_path = attempts_root / f"{attempt:06d}"
            attempt_path.mkdir(parents=False, exist_ok=False)
            started = {
                "schema_version": 1,
                "run_id": run_id,
                "attempt": attempt,
                "claim_id": uuid.uuid4().hex,
                "claimed_by": f"{socket.gethostname()}:{os.getpid()}",
                "started_at": utc_now(),
            }
            atomic_write_json(attempt_path / "attempt.json", started)
            self._write_status(
                run_id,
                {"status": "active", "attempt": attempt, "state": "new", "updated_at": utc_now()},
            )
            return RunClaim(
                ClaimAction.NEW,
                run_id,
                handle=AttemptHandle(self, run_id, attempt, lock_handle),
            )
        except BaseException:
            if not lock_handle.closed:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                lock_handle.close()
            raise

    def _invalidate_locked(
        self,
        run_id: str,
        attempt: int,
        reason: str,
        *,
        unknown_usage: bool | None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = self.run_dir(run_id) / "attempts" / f"{attempt:06d}"
        checkpoint_path = path / "checkpoint.json"
        checkpoint = read_json(checkpoint_path) if checkpoint_path.exists() else None
        if unknown_usage is None:
            unknown_usage = bool(_active_requests(checkpoint))
        active_requests = _active_requests(checkpoint)
        unknown_upper_bound = sum(
            request.get("max_tokens")
            if isinstance(request.get("max_tokens"), int)
            and not isinstance(request.get("max_tokens"), bool)
            else 0
            for request in active_requests.values()
        )
        record = {
            "schema_version": 1,
            "run_id": run_id,
            "attempt": attempt,
            "reason": reason,
            "unknown_usage": unknown_usage,
            "in_flight_requests": active_requests,
            "unknown_usage_upper_bound": unknown_upper_bound if unknown_usage else 0,
            "last_checkpoint_sequence": checkpoint.get("sequence") if checkpoint else None,
            "details": dict(details or {}),
            "invalidated_at": utc_now(),
        }
        atomic_write_json(path / "invalidation.json", record)
        self._write_status(
            run_id,
            {
                "status": "invalidated",
                "attempt": attempt,
                "unknown_usage": unknown_usage,
                "updated_at": record["invalidated_at"],
            },
        )
        append_jsonl(path / "events.jsonl", {"event": "invalidated", **record})
        return record

    def load_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        path = self.run_dir(run_id)
        status_path = path / "status.json"
        if not status_path.exists():
            return None
        status = read_json(status_path)
        attempt = status.get("attempt")
        if not isinstance(attempt, int):
            return None
        checkpoint_path = path / "attempts" / f"{attempt:06d}" / "checkpoint.json"
        return read_json(checkpoint_path) if checkpoint_path.exists() else None

    def load_result(self, run_id: str) -> dict[str, Any] | None:
        path = self.run_dir(run_id) / "result.json"
        return read_json(path) if path.exists() else None

    def status(self, run_id: str) -> dict[str, Any] | None:
        path = self.run_dir(run_id) / "status.json"
        return read_json(path) if path.exists() else None

    def is_complete(self, run_id: str) -> bool:
        return (self.run_dir(run_id) / "result.json").is_file()

    def list_run_ids(self) -> tuple[str, ...]:
        if not self.runs_root.exists():
            return ()
        return tuple(
            sorted(path.name for path in self.runs_root.iterdir() if path.is_dir())
        )

    def iter_results(self) -> Iterable[dict[str, Any]]:
        for run_id in self.list_run_ids():
            if result := self.load_result(run_id):
                yield result

    def invalidations(self, run_id: str) -> list[dict[str, Any]]:
        attempts_root = self.run_dir(run_id) / "attempts"
        if not attempts_root.exists():
            return []
        return [
            read_json(path)
            for path in sorted(attempts_root.glob("*/invalidation.json"))
        ]

    def write_artifact(
        self,
        relative_path: str | Path,
        value: Mapping[str, Any] | list[Any] | str | bytes,
    ) -> Path:
        """Write an atomic artifact while rejecting paths outside the root."""

        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact path must be relative and may not contain '..'")
        target = self.root / relative
        if isinstance(value, bytes):
            atomic_write_bytes(target, value)
        elif isinstance(value, str):
            atomic_write_text(target, value)
        else:
            atomic_write_json(target, value)
        return target
