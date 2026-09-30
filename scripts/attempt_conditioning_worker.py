"""Persistent two-request worker for one immutable conditioning queue stage."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import uuid
from pathlib import Path

from attempt_conditioning_queue import TaskQueue


async def run_worker(config: Path, manifest_path: Path, stage: str, queue_path: Path) -> dict:
    from submit_attempt_conditioning import verify_source

    from value_as_tool import pipeline
    from value_as_tool.benchmarks import QEDPromptSet
    from value_as_tool.client import OpenAIChatClient
    from value_as_tool.conditioning import summarize_task
    from value_as_tool.judging import JudgeRunner
    from value_as_tool.tokenization import HuggingFaceTokenCounter

    manifest = json.loads(manifest_path.read_text())
    verify_source(manifest)
    expected_config = manifest["preprocess_config"] if stage == "preprocess" else manifest["config"]
    if (
        config.resolve() != Path(expected_config).resolve()
        or queue_path.resolve() != Path(manifest["queues"][stage]["path"]).resolve()
    ):
        raise ValueError("worker config/queue does not match the authoritative launch manifest")
    context = pipeline.load_context(config)
    if context.config.runtime.max_concurrency != 2:
        raise ValueError("attempt-conditioning workers require request concurrency two")
    if context.config.runtime.solver_tensor_parallel_size != 1:
        raise ValueError("attempt-conditioning workers require tensor parallel size one")
    role = "judge" if stage.endswith("judge") else "solver"
    model = getattr(context.config.models, role)
    client = OpenAIChatClient(
        context.config.models.operational_base_url(role),
        api_key=os.environ.get(model.api_key_env),
        timeout=context.config.runtime.request_timeout_seconds,
    )
    counter = HuggingFaceTokenCounter(
        pipeline._model_entry(context, role)["path"],
        enable_thinking=context.config.sampling.enable_thinking if role == "solver" else False,
    )
    prompts = QEDPromptSet()
    queue = TaskQueue(queue_path)
    session = f"{os.environ.get('SLURM_JOB_ID', 'local')}:{uuid.uuid4().hex}"
    schedule = None
    items = {}
    if stage != "preprocess":
        schedule = pipeline._load_schedule(context)
        items = {item.run_id: item for item in schedule}
        solve_store = pipeline._store(context, schedule, "solve", initialize=role == "solver")
        if role == "judge":
            judge_store = pipeline._store(context, schedule, "judge")
            runner = JudgeRunner(
                prompts,
                max_tokens=context.config.evaluation.judge_output_tokens,
                token_counter=counter,
                reasoning_effort=context.config.evaluation.judge_reasoning_effort,
                model=model.name,
            )
    outcomes: dict[str, int] = {}

    async def lane(index: int) -> None:
        while True:
            claim = queue.claim(f"{session}:{index}")
            if claim is None:
                counts = queue.counts()
                if counts["failed"] or not counts["pending"] + counts["running"]:
                    return
                await asyncio.sleep(3)
                continue
            try:
                if stage == "preprocess":
                    result = await summarize_task(
                        context,
                        claim.task_id,
                        client=client,
                        token_counter=counter,
                    )
                    outcome = result["status"]
                elif role == "solver":
                    outcome = await pipeline._solve_one(
                        items[claim.payload["run_id"]],
                        context=context,
                        schedule=schedule,
                        store=solve_store,
                        client=client,
                        token_counter=counter,
                        prompts=prompts,
                    )
                else:
                    outcome = await pipeline._judge_one(
                        items[claim.payload["run_id"]],
                        context=context,
                        solve_store=solve_store,
                        judge_store=judge_store,
                        client=client,
                        runner=runner,
                    )
                if outcome in {"busy", "claimed_elsewhere"}:
                    queue.release(claim)
                    await asyncio.sleep(5)
                    continue
                # Exhausted unknown-usage retries do not leave a finalized
                # artifact. They must block the stage, just like a missing solve.
                failed = outcome in {"failed", "invalidated", "blocked", "solve_missing"}
                queue.finish(claim, {"outcome": outcome, "worker": session}, failed=failed)
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
                print(json.dumps({"task_id": claim.task_id, "outcome": outcome}), flush=True)
            except asyncio.CancelledError:
                queue.release(claim)
                raise
            except Exception as exc:
                queue.finish(claim, {"error": f"{type(exc).__name__}: {exc}"}, failed=True)
                raise

    workers = [asyncio.create_task(lane(i)) for i in range(2)]
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: [task.cancel() for task in workers])
    try:
        await asyncio.gather(*workers)
    finally:
        for task in workers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        await client.aclose()
    return {"stage": stage, "outcomes": outcomes, "queue": queue.counts()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("preprocess", "pilot_solve", "pilot_judge", "solve", "judge"),
        required=True,
    )
    parser.add_argument("--queue", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run_worker(args.config, args.manifest, args.stage, args.queue))))


if __name__ == "__main__":
    main()
