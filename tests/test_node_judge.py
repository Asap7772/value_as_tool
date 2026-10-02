"""Per-candidate judging: every node judged, once per distinct judge prompt."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from test_pipeline import CharacterCounter

from value_as_tool.benchmarks import BenchmarkItem, QEDPromptSet
from value_as_tool.client import ChatClientError
from value_as_tool.judging import JudgeRunner
from value_as_tool.pipeline import judge_tree_nodes, tree_nodes
from value_as_tool.schemas import AssistantMessage, ChatCompletion, TokenUsage

GOLD = "\\frac{3}{2}"


def _item() -> BenchmarkItem:
    return BenchmarkItem(
        benchmark="arxivmath_train",
        item_id="arxivmath-1",
        problem="Compute the offline sentinel ratio.",
        raw={},
        answer=GOLD,
    )


def _solve_result(contents: list[str]) -> dict[str, Any]:
    candidates = [
        {
            "cycle": 1,
            "role": "generator",
            "content": contents[0],
            "reasoning": None,
            "call_index": 0,
        }
    ]
    for branch, content in enumerate(contents[1:]):
        candidates.append(
            {
                "cycle": 2,
                "branch": branch,
                "parent_call_index": 0,
                "role": "reviser",
                "content": content,
                "reasoning": None,
                "call_index": 2 * branch + 2,
            }
        )
    return {"status": "cycle_limit", "candidates": list(reversed(candidates))}


class JudgeClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.prompts: list[str] = []

    async def complete(self, messages: Any, **kwargs: Any) -> ChatCompletion:
        prompt = messages[0]["content"]
        self.prompts.append(copy.deepcopy(prompt))
        if self.fail:
            raise ChatClientError("judge server unavailable")
        grade = "Incorrect" if "WRONG" in prompt else "Correct"
        return ChatCompletion(
            id="judge",
            model=kwargs.get("model"),
            message=AssistantMessage(content=f"<thinking>checked</thinking>\\boxed{{{grade}}}"),
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=40, completion_tokens=6, total_tokens=46),
        )


class RecordingHandle:
    def __init__(self) -> None:
        self.begun: list[str] = []
        self.completed: list[str] = []

    def begin_request(self, request_id: str, **kwargs: Any) -> None:
        assert request_id not in self.begun
        self.begun.append(request_id)

    def complete_request(self, request_id: str, **kwargs: Any) -> None:
        assert request_id in self.begun and request_id not in self.completed
        self.completed.append(request_id)


def _runner() -> JudgeRunner:
    return JudgeRunner(
        QEDPromptSet(),
        max_tokens=1_000,
        token_counter=CharacterCounter(),
        reasoning_effort="medium",
        model="openai/gpt-oss-20b",
    )


def test_tree_nodes_are_ordered_by_call_and_keep_their_tree_position() -> None:
    nodes = tree_nodes(_solve_result([f"a \\boxed{{{GOLD}}}", "b \\boxed{1}", "c \\boxed{2}"]))

    assert [node["call_index"] for node in nodes] == [0, 2, 4]
    assert [(node["cycle"], node["branch"]) for node in nodes] == [(1, None), (2, 0), (2, 1)]
    assert {node["parent_call_index"] for node in nodes[1:]} == {0}
    assert all(len(node["content_sha256"]) == 64 for node in nodes)


@pytest.mark.asyncio
async def test_every_candidate_is_judged_once_per_distinct_answer() -> None:
    contents = [
        f"First attempt, so \\boxed{{{GOLD}}}.",
        f"A different derivation, still \\boxed{{{GOLD}}}.",
        "A slip gives \\boxed{WRONG-1}.",
        f"Rechecked: \\boxed{{{GOLD}}}",
        "Another slip, \\boxed{WRONG-1}.",
        "",
    ]
    client, handle = JudgeClient(), RecordingHandle()
    judged = await judge_tree_nodes(
        _solve_result(contents), _item(), _runner(), client, concurrency=3, handle=handle
    )

    assert len(client.prompts) == 2
    assert handle.begun == handle.completed and len(handle.completed) == 2
    by_call = {node["call_index"]: node for node in judged["nodes"]}
    assert [by_call[index]["correct"] for index in (0, 2, 4, 6, 8)] == [
        True,
        True,
        False,
        True,
        False,
    ]
    assert by_call[10]["judge_status"] == "empty_answer"
    assert by_call[10]["judge_prompt_sha256"] is None
    assert judged["node_status_counts"] == {"completed": 5, "empty_answer": 1}
    assert judged["judge_status"] == "partial"
    assert set(judged["judgments"]) == {
        node["judge_prompt_sha256"] for node in judged["nodes"] if node["judge_prompt_sha256"]
    }
    assert judged["judge_usage"] == {
        "prompt_tokens": 80,
        "completion_tokens": 12,
        "total_tokens": 92,
        "usage_exact": True,
    }


@pytest.mark.asyncio
async def test_a_judgment_without_exact_usage_fails_the_whole_tree() -> None:
    handle = RecordingHandle()
    with pytest.raises(ExceptionGroup):
        await judge_tree_nodes(
            _solve_result([f"\\boxed{{{GOLD}}}", "\\boxed{WRONG-2}"]),
            _item(),
            _runner(),
            JudgeClient(fail=True),
            concurrency=2,
            handle=handle,
        )
    assert handle.completed == []
