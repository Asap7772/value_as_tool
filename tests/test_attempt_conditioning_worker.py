from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        "invalidated",
        "failed",
        "accepted",
        "completed",
        "cycle_limit",
        "protocol_error",
        "context_exhausted",
        "budget_exhausted",
        "already_complete",
    ],
)
async def test_worker_blocks_invalidated_attempts_but_preserves_terminal_model_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    worker = importlib.import_module("attempt_conditioning_worker")
    launcher = importlib.import_module("submit_attempt_conditioning")
    queue_module = importlib.import_module("attempt_conditioning_queue")
    from value_as_tool import benchmarks, client, pipeline, tokenization

    config_path = tmp_path / "experiment.yaml"
    queue = queue_module.TaskQueue(tmp_path / "solve.sqlite3")
    queue.initialize([{"task_id": "run-one", "run_id": "run-one", "dependencies": []}], {})
    manifest = {
        "config": str(config_path),
        "stage": "solve",
        "state": "running",
        "jobs": {},
        "queues": {"solve": {"path": str(queue.path)}},
    }
    manifest_path = tmp_path / "submission.json"
    manifest_path.write_text(json.dumps(manifest))
    model = SimpleNamespace(api_key_env="MODEL_API_KEY", name="test-model")
    config = SimpleNamespace(
        runtime=SimpleNamespace(
            max_concurrency=2,
            solver_tensor_parallel_size=1,
            request_timeout_seconds=10,
        ),
        models=SimpleNamespace(
            solver=model,
            operational_base_url=lambda role: "http://localhost:1/v1",
        ),
        sampling=SimpleNamespace(enable_thinking=True),
    )
    context = SimpleNamespace(config=config)
    closed = AsyncMock()
    monkeypatch.setattr(launcher, "verify_source", lambda _: None)
    monkeypatch.setattr(pipeline, "load_context", lambda _: context)
    monkeypatch.setattr(pipeline, "_model_entry", lambda *args: {"path": "local-tokenizer"})
    monkeypatch.setattr(pipeline, "_load_schedule", lambda _: [SimpleNamespace(run_id="run-one")])
    monkeypatch.setattr(pipeline, "_store", lambda *args, **kwargs: object())
    solve = AsyncMock(return_value=outcome)
    monkeypatch.setattr(pipeline, "_solve_one", solve)
    monkeypatch.setattr(
        client, "OpenAIChatClient", lambda *args, **kwargs: SimpleNamespace(aclose=closed)
    )
    monkeypatch.setattr(tokenization, "HuggingFaceTokenCounter", lambda *args, **kwargs: object())
    monkeypatch.setattr(benchmarks, "QEDPromptSet", lambda: object())
    monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", lambda *args: None)

    result = await worker.run_worker(config_path, manifest_path, "solve", queue.path)

    blocked = outcome in {"invalidated", "failed"}
    assert result["queue"]["failed"] == int(blocked)
    assert result["queue"]["complete"] == int(not blocked)
    assert result["queue"]["pending"] == result["queue"]["running"] == 0
    assert result["outcomes"] == {outcome: 1}
    assert queue.results()[0]["result"]["outcome"] == outcome
    solve.assert_awaited_once()
    closed.assert_awaited_once()

    if blocked:
        monkeypatch.setattr(launcher, "observe_owned_jobs", lambda _: {})
        assert launcher.tick(manifest_path, manifest, allow_submit=False) is False
        assert manifest["stage"] == "solve" and manifest["state"] == "blocked"
        assert "failed durable tasks" in manifest["blocked_reason"]
        assert set(manifest["queues"]) == {"solve"}
