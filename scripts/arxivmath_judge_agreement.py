"""Measure the answer judge against MathArena's labels on Qwen3.6-35B attempts.

Runs where a judge server is up (run_gpu.sbatch with stage judge-nodes and
VALUE_AS_TOOL_GPU_COMMAND). Samples attempts balanced across MathArena's
``correct`` label, judges each full response with the experiment's JudgeRunner,
and reports agreement, Cohen's kappa, the confusion matrix and disagreements.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from value_as_tool.benchmarks import BenchmarkItem, QEDPromptSet
from value_as_tool.client import OpenAIChatClient
from value_as_tool.judging import JudgeRunner, extract_candidate_answer
from value_as_tool.pipeline import load_context
from value_as_tool.tokenization import HuggingFaceTokenCounter

COLUMNS = ["problem_idx", "idx_answer", "problem", "gold_answer", "answer", "correct"]


def sample(source: Path, size: int) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(source.glob("data/*.parquet")):
        rows.extend(pq.read_table(path, columns=COLUMNS).to_pylist())
    key = lambda row: hashlib.sha256(  # noqa: E731
        f"{row['problem_idx']}:{row['idx_answer']}".encode()
    ).hexdigest()
    chosen = []
    for label in (True, False):
        group = sorted((row for row in rows if bool(row["correct"]) is label), key=key)
        chosen.extend(group[: size // 2])
    return chosen


def kappa(pairs: list[tuple[bool, bool]]) -> float:
    total = len(pairs)
    observed = sum(left == right for left, right in pairs) / total
    left_true = sum(left for left, _ in pairs) / total
    right_true = sum(right for _, right in pairs) / total
    expected = left_true * right_true + (1 - left_true) * (1 - right_true)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


async def judge_all(rows: list[dict[str, Any]], runner: JudgeRunner, client: Any, limit: int):
    semaphore = asyncio.Semaphore(limit)

    async def one(row: dict[str, Any]) -> dict[str, Any]:
        item = BenchmarkItem(
            benchmark="arxivmath_train",
            item_id=f"arxivmath-{row['problem_idx']}",
            problem=row["problem"],
            raw={},
            answer=row["gold_answer"],
        )
        async with semaphore:
            result = await runner.judge(item, row["answer"], client)
        return {**{key: row[key] for key in ("problem_idx", "idx_answer", "gold_answer")},
                "matharena_correct": bool(row["correct"]), "judge_correct": result.correct,
                "judge_status": result.status,
                "extracted_answer": extract_candidate_answer(row["answer"])[:300]}

    return await asyncio.gather(*(one(row) for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", type=Path, required=True, help="downloaded dataset directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=1_000)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    context = load_context(args.config)
    model = context.config.models.judge
    manifest = json.loads(
        (context.artifact_root / "prepared" / "models" / "manifest.json").read_text()
    )
    runner = JudgeRunner(
        QEDPromptSet(),
        max_tokens=context.config.evaluation.judge_output_tokens,
        token_counter=HuggingFaceTokenCounter(
            manifest["models"]["judge"]["path"], enable_thinking=False
        ),
        reasoning_effort=context.config.evaluation.judge_reasoning_effort,
        model=model.name,
    )
    client = OpenAIChatClient(
        context.config.models.operational_base_url("judge", os.environ),
        api_key=os.environ.get(model.api_key_env),
        timeout=context.config.runtime.request_timeout_seconds,
    )
    rows = sample(args.source, args.sample)

    async def run() -> list[dict[str, Any]]:
        try:
            return await judge_all(rows, runner, client, args.concurrency)
        finally:
            await client.aclose()

    results = asyncio.run(run())
    completed = [row for row in results if row["judge_status"] == "completed"]
    pairs = [(row["matharena_correct"], row["judge_correct"]) for row in completed]
    confusion = Counter(f"matharena={left} judge={right}" for left, right in pairs)
    report = {
        "sampled": len(rows),
        "completed": len(completed),
        "statuses": dict(Counter(row["judge_status"] for row in results)),
        "agreement": round(sum(left == right for left, right in pairs) / max(1, len(pairs)), 4),
        "cohen_kappa": round(kappa(pairs), 4) if pairs else None,
        "confusion": dict(confusion),
        "disagreements": [
            row for row in completed if row["matharena_correct"] != row["judge_correct"]
        ][:40],
    }
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "disagreements"}))


if __name__ == "__main__":
    sys.exit(main())
