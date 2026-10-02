"""Offline prepare → solve → judge-nodes → export for branched GVR collection."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_pipeline import CharacterCounter

from value_as_tool.benchmarks import BENCHMARKS
from value_as_tool.harnesses.gvr_branched import BRANCHES, ROUNDS
from value_as_tool.pipeline import judge_nodes, load_context, prepare, solve, status
from value_as_tool.schemas import AssistantMessage, ChatCompletion, TokenUsage, ToolCall

GOLD = "42"
OUTCOMES = ("correct", "minor_fix", "critical_flaw")
BRANCHED = "value_as_tool.harnesses.gvr_branched:AgentHarness"
JOINT = "value_as_tool.harnesses.gvr_replan:JointPlanHarness"
INDEPENDENT = "value_as_tool.harnesses.gvr_replan:IndependentPlanHarness"


class SolverClient:
    """Verdicts for verifier calls, right or wrong answers otherwise.

    Choices follow the label-derived seed, so they are deterministic however the
    concurrent siblings interleave.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages: Any, **kwargs: Any) -> ChatCompletion:
        self.calls += 1
        step = int(hashlib.sha256(str(kwargs["seed"]).encode()).hexdigest(), 16)
        tools = {tool["function"]["name"]: tool["function"] for tool in kwargs.get("tools") or []}
        if "submit_verdict" in tools:
            arguments: dict[str, Any] = {
                "outcome": OUTCOMES[step % 3],
                "fault_category": "logic",
                "candidate_excerpt": "",
            }
            if "success_probability" in tools["submit_verdict"]["parameters"]["properties"]:
                arguments.update(success_probability=(step % 10) / 10, rationale="Check it.")
            else:
                arguments["critique"] = "Check the last step."
            message = AssistantMessage(
                content=None,
                tool_calls=(ToolCall("verdict", "submit_verdict", json.dumps(arguments)),),
            )
        elif "submit_plans" in tools:
            slots = len(tools["submit_plans"]["parameters"]["properties"]) // 4
            arguments = {}
            for slot in range(1, slots + 1):
                arguments |= {
                    f"plan_{slot}_title": f"plan {slot}",
                    f"plan_{slot}_brief": f"Try idea {(step + slot) % 5}.",
                    f"plan_{slot}_show_current_solution": (step + slot) % 2 == 0,
                    f"plan_{slot}_success_probability": ((step + slot) % 10) / 10,
                }
            message = AssistantMessage(
                content=None,
                tool_calls=(ToolCall("plans", "submit_plans", json.dumps(arguments)),),
            )
        else:
            answer = GOLD if step % 2 else "41"
            message = AssistantMessage(content=f"Work. \\boxed{{{answer}}}", reasoning="Private.")
        return ChatCompletion(
            id=f"solve-{step}",
            model=str(kwargs.get("model")),
            message=message,
            finish_reason="tool_calls" if message.tool_calls else "stop",
            usage=TokenUsage(prompt_tokens=20, completion_tokens=9, total_tokens=29),
        )


class JudgeClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, messages: Any, **kwargs: Any) -> ChatCompletion:
        prompt = messages[0]["content"]
        self.prompts.append(copy.deepcopy(prompt))
        student = prompt.rsplit("Model Solution:", 1)[-1].split("Golden Answer:", 1)[0]
        grade = "Correct" if student.strip() == GOLD else "Incorrect"
        return ChatCompletion(
            id="judge",
            model=str(kwargs.get("model")),
            message=AssistantMessage(content=f"<thinking>ok</thinking>\\boxed{{{grade}}}"),
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=30, completion_tokens=4, total_tokens=34),
        )


def _write_experiment(tmp_path: Path, harness: str = BRANCHED) -> Path:
    rows = [
        {"item_id": f"arxivmath-{index}", "problem": f"Compute {index} + 41.", "gold_answer": GOLD}
        for index in (1, 2)
    ]
    dataset = tmp_path / "assets" / "benchmarks" / "arxivmath" / "train.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = {
        "paths": {"artifact_root": "artifacts", "asset_root": "assets"},
        "datasets": {
            "arxivmath_train": {
                "name": "MathArena/arxivmath-training_outputs",
                "revision": "e" * 40,
                "split": "train",
                "path": "benchmarks/arxivmath/train.jsonl",
                "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            }
        },
        "sampling": {"thinking_content_reserve_tokens": 0},
        "budget": {
            "generated_tokens": 8388608,
            "context_tokens": 262144,
            "context_headroom_tokens": 1024,
            "initial_generator_tokens": 2621440,
            "verifier_tokens": 524288,
            "correction_pool_tokens": 5242880,
            "max_candidate_versions": 3,
            "minimum_call_tokens": 1024,
        },
        "subagents": {"child_tokens": 262144, "final_candidate_reserve_tokens": 262144},
        "value_tool": {"verifier_tokens": 262144, "final_response_reserve_tokens": 262144},
        "evaluation": {
            "conditions": [],
            "harnesses": [harness],
            "benchmarks": ["arxivmath_train"],
            "seeds": [0],
            "direct_answer_compatibility_seed": 0,
            "bootstrap_samples": 10,
        },
        "runtime": {"max_retries": 1, "max_concurrency": 2},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "harness", [BRANCHED, JOINT, INDEPENDENT], ids=["branched", "joint", "independent"]
)
@pytest.mark.asyncio
async def test_branched_trees_are_solved_node_judged_and_exported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, harness: str
) -> None:
    monkeypatch.setitem(
        BENCHMARKS, "arxivmath_train", replace(BENCHMARKS["arxivmath_train"], expected_rows=2)
    )
    context = load_context(_write_experiment(tmp_path, harness))
    prepared = prepare(context, model_preparer=lambda config: {"models": {}})
    assert prepared["scheduled"] == 2

    solved = await solve(context, client=SolverClient(), token_counter=CharacterCounter())
    assert solved["outcomes"] == {"cycle_limit": 2}

    judge_client = JudgeClient()
    judged = await judge_nodes(context, client=judge_client, token_counter=CharacterCounter())
    assert judged["outcomes"] == {"completed": 2}
    solved_trees = [
        json.loads(path.read_text())
        for path in (context.artifact_root / "solve" / "runs").glob("*/result.json")
    ]
    distinct = {
        (tree["request"]["problem_id"], candidate["content"].rsplit("boxed", 1)[-1])
        for tree in solved_trees
        for candidate in tree["candidates"]
    }
    # One judgment per distinct (problem, final answer): answer judging dedups.
    assert len(judge_client.prompts) == len(distinct) < 2 * (1 + ROUNDS * BRANCHES)
    rerun = await judge_nodes(context, client=JudgeClient(), token_counter=CharacterCounter())
    assert rerun["outcomes"] == {"already_complete": 2}
    assert status(context)["node_judge"]["counts"] == {"completed": 2}

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    exporter = importlib.import_module("export_gvr_tree_dataset")
    output = tmp_path / "export"
    monkeypatch.setattr(
        sys,
        "argv",
        ["export", "--artifact-root", str(context.artifact_root), "--output", str(output)],
    )
    exporter.main()

    def rows(name: str) -> list[dict[str, Any]]:
        return [json.loads(line) for line in (output / name).read_text().splitlines()]

    trees, nodes, verifications, plans = (
        rows("trees.jsonl"),
        rows("nodes.jsonl"),
        rows("verifications.jsonl"),
        rows("plans.jsonl"),
    )
    assert len(trees) == 2
    assert len(nodes) == 2 * (1 + ROUNDS * BRANCHES)
    assert len(verifications) == 2 * ROUNDS * BRANCHES
    replan = harness != BRANCHED
    for tree in trees:
        tree_nodes = [node for node in nodes if node["run_id"] == tree["run_id"]]
        spine = [node for node in tree_nodes if node["on_spine"]]
        assert len(spine) == ROUNDS
        assert all(node["children_correct_share"] is not None for node in spine)
        assert all(node["verdicts_correct_share"] is not None for node in spine)
        assert {node["judge_correct"] for node in tree_nodes} == {True, False}
        assert "reasoning" not in tree_nodes[0]
        assert tree["distinct_answers"] == len({node["extracted_answer"] for node in tree_nodes})
        if not replan:
            assert {node["mode"] for node in tree_nodes} >= {"generate", "recheck", "revise"}
            continue
        assert {node["mode"] for node in tree_nodes} == {"generate", "exec"}
        executed = [node for node in tree_nodes if node["mode"] == "exec"]
        assert all(node["routing_verdict"] is None for node in executed)
        assert all(
            node["plan_brief"] and node["plan_success_probability"] is not None for node in executed
        )
        assert all(
            node["executor_saw_solution"] is node["plan_show_current_solution"] for node in executed
        )
        assert all(node["verdicts_mean_probability"] is not None for node in spine)
        promoted = [node for node in spine if node["cycle"] > 1]
        assert all(node["promotion_rank"] is not None for node in promoted)
    if replan:
        assert len(plans) == 2 * ROUNDS * BRANCHES
        assert all(row["valid"] and row["executed"] for row in plans)
        assert sum(row["promoted"] for row in plans) == 2 * (ROUNDS - 1)
        assert all(
            row["rationale"] and row["success_probability"] is not None for row in verifications
        )
    else:
        assert plans == []
    assert {row["candidate_judge_correct"] for row in verifications} <= {True, False}
