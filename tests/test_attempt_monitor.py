from __future__ import annotations

import importlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

OBSERVED = datetime(2026, 9, 29, 1, 45, tzinfo=UTC)


@pytest.fixture
def monitor(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("monitor_attempt_conditioning")


def make_queue(tmp_path, statuses):
    path = tmp_path / "queue.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE tasks (task_id TEXT, status TEXT, payload TEXT, "
            "started_at TEXT, completed_at TEXT, result TEXT)"
        )
        for index, status in enumerate(statuses):
            db.execute(
                "INSERT INTO tasks VALUES (?, ?, '{}', ?, ?, '{}')",
                (
                    str(index),
                    status,
                    (OBSERVED - timedelta(minutes=2)).isoformat(),
                    (OBSERVED - timedelta(minutes=1)).isoformat() if status == "complete" else None,
                ),
            )
    return path


@pytest.mark.parametrize("statuses", [["complete", "pending", "running"], ["pending"]])
def test_incomplete_full_solve_keeps_diagnostics_without_eta(monitor, tmp_path, statuses):
    path = make_queue(tmp_path, statuses)
    progress = monitor.queue_progress(str(path), OBSERVED, stage="solve")
    assert progress["eta_minutes"] is None
    assert "Problem costs vary" in progress["eta_caveat"]
    assert "queue order" in progress["eta_caveat"]
    if "complete" in statuses:
        assert progress["completed_per_minute"] == {"5": 0.2, "15": 1 / 15}
        assert progress["bulk_work_projection_minutes"] == 10
    else:
        assert progress["bulk_work_projection_minutes"] is None


def test_full_solve_failure_caveat_takes_precedence(monitor, tmp_path):
    path = make_queue(tmp_path, ["complete", "failed", "pending"])
    progress = monitor.queue_progress(str(path), OBSERVED, stage="solve")
    assert progress["eta_minutes"] is None
    assert progress["eta_caveat"] == "Failed tasks require recovery."
    assert progress["bulk_work_projection_minutes"] == 5


def test_completed_full_solve_has_zero_eta(monitor, tmp_path):
    path = make_queue(tmp_path, ["complete", "complete"])
    progress = monitor.queue_progress(str(path), OBSERVED, stage="solve")
    assert progress["eta_minutes"] == 0
    assert progress["eta_caveat"] is None


def test_other_stages_keep_existing_completion_projection(monitor, tmp_path):
    path = make_queue(tmp_path, ["complete", "pending"])
    progress = monitor.queue_progress(str(path), OBSERVED, stage="preprocess")
    assert progress["eta_minutes"] == progress["bulk_work_projection_minutes"] == 5
    assert progress["eta_caveat"] is None


def test_observe_and_printed_summary_carry_full_solve_caveat(
    monitor, tmp_path, monkeypatch, capsys
):
    queue = make_queue(tmp_path, ["complete", "pending"])
    manifest = tmp_path / "submission.json"
    manifest.write_text(
        json.dumps(
            {
                "state": "running",
                "stage": "solve",
                "jobs": {},
                "queues": {"solve": {"path": str(queue)}},
                "expected_runs": 2,
            }
        )
    )
    monkeypatch.setattr(monitor, "query_user_jobs", lambda: [])
    snapshot = monitor.observe(manifest)
    monitor.persist(tmp_path / "monitor", snapshot)
    summary = json.loads(capsys.readouterr().out)
    assert summary["stage_eta_minutes"] is None
    assert "Problem costs vary" in summary["eta_caveat"]
    assert "bulk_work_projection_minutes" in snapshot["queues"]["solve"]
