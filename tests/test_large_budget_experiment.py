from __future__ import annotations

import json
from collections import Counter
from math import comb
from pathlib import Path

import pytest

from value_as_tool import pipeline
from value_as_tool.config import load_config
from value_as_tool.schedule import build_schedule, items_for_shard, write_schedule

CONFIG_PATH = Path(__file__).resolve().parents[1] / "experiment_qwen35_9b_harness_large_budget.yaml"


def _problems(imo_count: int, proofbench_count: int) -> dict[str, list[dict[str, str]]]:
    return {
        benchmark: [
            {
                "item_id": f"problem-{index}",
                "problem": f"Prove identity {index}.",
                "solution": "A reference proof.",
            }
            for index in range(count)
        ]
        for benchmark, count in (("imo_proof", imo_count), ("proofbench", proofbench_count))
    }


def test_eight_sample_schedule_covers_every_method_and_shards_without_overlap() -> None:
    config = load_config(CONFIG_PATH)
    schedule = build_schedule(_problems(60, 145), config)

    assert len(schedule) == 14_760
    assert config.evaluation.seeds == tuple(range(8))
    assert config.models.solver.name == "Qwen/Qwen3.5-9B"
    assert config.sampling.enable_thinking
    assert config.sampling.thinking_content_reserve_tokens == 0
    assert config.budget.generated_tokens > config.budget.context_tokens
    assert len({item.run_id for item in schedule}) == len(schedule)
    assert set(Counter(item.harness_id for item in schedule).values()) == {1_640}
    assert set(
        Counter((item.benchmark, item.problem_id, item.harness_id) for item in schedule).values()
    ) == {8}

    shards = [items_for_shard(schedule, index, 384) for index in range(384)]
    assert sorted(item.ordinal for shard in shards for item in shard) == list(range(14_760))
    assert {len(shard) for shard in shards} == {38, 39}


def test_pipeline_report_emits_pass_and_cost_through_eight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = pipeline.load_context(
        CONFIG_PATH,
        overrides={"evaluation.bootstrap_samples": 20},
        environment={
            "VALUE_AS_TOOL_ARTIFACT_ROOT": str(tmp_path / "artifacts"),
            "VALUE_AS_TOOL_ASSET_ROOT": str(tmp_path / "assets"),
        },
    )
    schedule = build_schedule(
        _problems(1, 1), context.config, config_fingerprint=context.config_fingerprint
    )
    write_schedule(pipeline.schedule_path(context), schedule)
    rows = [
        {
            **item.to_dict(),
            "solve_status": "completed",
            "judge_status": "completed",
            "score": 7 if item.seed in {0, 4} else 0,
            "generated_tokens": 10 * (item.seed + 1),
            "total_tokens": 20 * (item.seed + 1),
            "usage_exact": True,
        }
        for item in schedule
    ]
    monkeypatch.setattr(pipeline, "joined_rows", lambda _: rows)

    result = pipeline.report(context)

    assert result["complete"] is True
    assert result["scheduled_cells"] == 144
    assert len(result["curves"]) == 2 * 9 * 8
    assert {row["evaluation_runs"] for row in result["summary"]} == {8}
    curves = {
        row["k"]: row
        for row in result["curves"]
        if row["benchmark"] == "imo_proof" and row["method_id"] == "direct"
    }
    assert set(curves) == set(range(1, 9))
    assert curves[1]["pass_at_k"] == pytest.approx(2 / 8)
    assert curves[3]["pass_at_k"] == pytest.approx(1 - comb(6, 3) / comb(8, 3))
    assert curves[8]["pass_at_k"] == 1
    assert curves[1]["generated_cost_at_k"] == 45
    assert curves[3]["generated_cost_at_k"] == 135
    assert curves[8]["generated_cost_at_k"] == 360
    assert "pass@8" in {row["metric"] for row in result["paired_bootstrap"]}
    persisted = json.loads((context.artifact_root / "report" / "report.json").read_text())
    assert persisted == result
