from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from value_as_tool import judging
from value_as_tool.benchmarks import (
    QED_NANO_COMMIT,
    QED_PROMPTS,
    BenchmarkItem,
    QEDPromptSet,
    judge_uses_reference,
    row_to_item,
    validate_benchmark,
)
from value_as_tool.client import ChatClientError
from value_as_tool.judging import (
    JUDGE_CONTEXT_RECOVERY_OUTPUT_TOKENS,
    JUDGE_CONTEXT_TRUNCATION_MARKER,
    extract_candidate_answer,
    find_last_boxed_content,
    parse_answer_correct,
    parse_proof_score,
    prepare_judge_prompt,
    remove_self_evaluation,
    validate_judge_context_recovery,
)
from value_as_tool.metrics import (
    compatibility_metrics,
    compute_curves,
    compute_metrics,
    expected_best_at_k,
    expected_cost_at_k,
    paired_bootstrap,
    pass_at_k,
    summarize_metrics,
)
from value_as_tool.reporting import load_jsonl, render_markdown, write_report
from value_as_tool.schemas import TokenUsage


class CharacterTokenCounter:
    """A deterministic tokenizer for context-policy tests."""

    def count_messages(
        self,
        messages: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        tools: Any = None,
    ) -> int:
        del tools
        return sum(len(str(message.get("content", ""))) for message in messages)

    def count_text(self, text: str) -> int:
        return len(text)

    def encode_text(self, text: str) -> tuple[int, ...]:
        return tuple(ord(character) for character in text)

    def decode_tokens(self, token_ids: tuple[int, ...]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def _result_row(
    benchmark: str,
    item_id: str,
    condition: str,
    seed: int,
    reward: int | bool,
    *,
    generated: int = 10,
    total: int = 15,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "benchmark": benchmark,
        "item_id": item_id,
        "condition": condition,
        "seed": seed,
        "judge_status": "completed",
        "generated_tokens": generated,
        "total_tokens": total,
        "usage_exact": True,
    }
    if benchmark == "imo_answer":
        row["correct"] = bool(reward)
    else:
        row["score"] = int(reward)
    return row


@pytest.mark.asyncio
async def test_judge_preserves_exact_usage_from_response_shape_error() -> None:
    class ExactErrorClient:
        async def complete(self, messages: Any, **kwargs: Any) -> Any:
            del messages, kwargs
            raise ChatClientError(
                "malformed response after accounting",
                usage=TokenUsage(
                    prompt_tokens=101,
                    completion_tokens=11,
                    total_tokens=112,
                ),
            )

    item = BenchmarkItem(
        benchmark="imo_proof",
        item_id="p",
        problem="Prove P.",
        solution="Reference.",
        rubric={"7": "complete"},
        raw={},
    )
    runner = judging.JudgeRunner(QEDPromptSet(), model="judge")
    result = await runner.judge(item, "Candidate.", ExactErrorClient())

    assert result.status == "api_error"
    assert result.usage["usage_exact"] is True
    assert result.usage["completion_tokens"] == 11


def test_failed_solve_cannot_be_promoted_by_a_completed_judge_row() -> None:
    row = _result_row("imo_proof", "p", "direct", 0, 7)
    row["solve_status"] = "budget_exhausted"

    summary = summarize_metrics([row])

    assert summary[0]["raw_success_rate"] == 0
    assert summary[0]["completed"] == 0
    assert summary[0]["complete"] is False


def test_vendored_prompt_hashes_and_tamper_detection(tmp_path: Path) -> None:
    prompts = QEDPromptSet()
    manifest = json.loads((prompts.root / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["commit"] == QED_NANO_COMMIT
    for name, (filename, expected_digest, has_final_newline) in QED_PROMPTS.items():
        raw = (prompts.root / filename).read_bytes()
        upstream_bytes = raw if has_final_newline or not raw.endswith(b"\n") else raw[:-1]
        assert hashlib.sha256(upstream_bytes).hexdigest() == expected_digest
        assert manifest["files"][filename] == expected_digest
        assert prompts.read(name).encode("utf-8") == upstream_bytes

    copied = tmp_path / "qed_nano"
    shutil.copytree(prompts.root, copied)
    target = copied / QED_PROMPTS["proof_run"][0]
    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="prompt checksum mismatch for proof_run"):
        QEDPromptSet(copied)


def test_qed_prompt_rendering_is_benchmark_specific_and_condition_blind() -> None:
    prompts = QEDPromptSet()
    imo_proof = BenchmarkItem(
        benchmark="imo_proof",
        item_id="proof-1",
        problem="PROBLEM_SENTINEL",
        solution="REFERENCE_SENTINEL",
        rubric="RUBRIC_SENTINEL",
        raw={},
    )
    proofbench = BenchmarkItem(
        benchmark="proofbench",
        item_id="proof-2",
        problem="PB_PROBLEM_SENTINEL",
        solution="PB_REFERENCE_MUST_STAY_BLIND",
        rubric="PB_RUBRIC_SENTINEL",
        raw={},
    )
    answer = BenchmarkItem(
        benchmark="imo_answer",
        item_id="answer-1",
        problem="ANSWER_PROBLEM_SENTINEL",
        answer="GOLD_SENTINEL",
        raw={},
    )

    assert prompts.solve_prompt(imo_proof) == prompts.read("proof_run").format(
        problem_statement=imo_proof.problem
    )
    proof_judge = prompts.judge_prompt(
        imo_proof,
        "CANDIDATE_SENTINEL\n# Self Evaluation\nSELF_EVALUATION_MUST_BE_REMOVED",
    )
    assert all(
        sentinel in proof_judge
        for sentinel in (
            "PROBLEM_SENTINEL",
            "REFERENCE_SENTINEL",
            "RUBRIC_SENTINEL",
            "CANDIDATE_SENTINEL",
        )
    )
    assert "SELF_EVALUATION_MUST_BE_REMOVED" not in proof_judge
    assert judge_uses_reference(imo_proof)

    proofbench_judge = prompts.judge_prompt(proofbench, "PB_CANDIDATE_SENTINEL")
    assert "PB_PROBLEM_SENTINEL" in proofbench_judge
    assert "PB_RUBRIC_SENTINEL" in proofbench_judge
    assert "PB_CANDIDATE_SENTINEL" in proofbench_judge
    assert "PB_REFERENCE_MUST_STAY_BLIND" not in proofbench_judge
    assert not judge_uses_reference(proofbench)

    answer_judge = prompts.judge_prompt(
        answer,
        "reasoning that is not submitted\n\\boxed{CANDIDATE_ANSWER_SENTINEL}",
    )
    assert "ANSWER_PROBLEM_SENTINEL" in answer_judge
    assert "CANDIDATE_ANSWER_SENTINEL" in answer_judge
    assert "GOLD_SENTINEL" in answer_judge
    assert "reasoning that is not submitted" not in answer_judge
    assert prompts.solve_prompt(answer) == prompts.read("answer_run").format(
        problem_statement=answer.problem
    )


def test_dataset_rows_normalize_to_stable_items_and_validate() -> None:
    proof_row = {
        "problem": "Show that 1 = 1.",
        "solution": "Reflexivity.",
        "grading_guidelines": {"7": "complete"},
        "extra": "preserved",
    }
    first = row_to_item("imo-proofbench", proof_row, 4)
    second = row_to_item("imo_proof", proof_row, 4)
    digest = hashlib.sha256(proof_row["problem"].encode()).hexdigest()[:12]
    assert first == second
    assert first.item_id == f"row-00004-{digest}"
    assert first.raw == proof_row
    assert first.is_proof
    assert first.supports_reference_verification

    answer = row_to_item(
        "imo-answerbench",
        {
            "Problem ID": 17,
            "Problem": "Find x.",
            "Short Answer": 42,
        },
        0,
    )
    assert answer.item_id == "17"
    assert answer.problem == "Find x."
    assert answer.answer == "42"
    assert not answer.is_proof
    assert not answer.supports_reference_verification

    assert validate_benchmark("imo_proof", [first], check_size=False) == [first]
    with pytest.raises(ValueError, match="duplicate item IDs"):
        validate_benchmark("imo_proof", [first, first], check_size=False)
    with pytest.raises(ValueError, match="no reference solution"):
        row_to_item(
            "imo_proof",
            {"problem": "P", "grading_guidelines": "rubric"},
            0,
        )
    with pytest.raises(ValueError, match="no golden answer"):
        row_to_item("imo_answer", {"Problem": "P"}, 0)


def test_qed_response_parsers_match_released_conventions() -> None:
    assert find_last_boxed_content("first \\boxed{1}\nlast \\boxed{x} and \\fbox{y}") == "x,y"
    assert find_last_boxed_content(r"\boxed{\frac{1}{\boxed{2}}}") == r"\frac{1}{2}"
    assert find_last_boxed_content("unboxed") is None

    assert extract_candidate_answer("work\n\\boxed{17}") == "17"
    assert extract_candidate_answer("1\n2\n3\n4\n5\n6") == "...2\n3\n4\n5\n6"
    assert remove_self_evaluation("proof\n## Final Evaluation\nunsupported praise") == "proof"

    assert parse_proof_score("<points>2</points> then <points> 7 / 7 </points>") == 7
    assert parse_proof_score("<points>8</points>") is None
    assert parse_proof_score("no score") is None
    assert parse_answer_correct(r"grade: \boxed{Correct}")
    assert not parse_answer_correct(r"grade: \boxed{INCORRECT}")
    assert not parse_answer_correct("malformed grade")


def test_oversized_proof_is_balanced_and_reproducibly_truncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "proofbench.txt").write_text(
        "{problem_statement}|{solution}|{guidelines}|{student_answer}",
        encoding="utf-8",
    )
    prompts = QEDPromptSet(tmp_path, verify=False)
    item = BenchmarkItem(
        benchmark="imo_proof",
        item_id="p",
        problem="P",
        solution="R",
        rubric="G",
        raw={},
    )
    counter = CharacterTokenCounter()
    monkeypatch.setattr(judging, "JUDGE_CONTEXT_INPUT_LIMIT", 160)
    model_solution = "A" * 200 + "B" * 200 + "\n# Self Evaluation\nSECRET"

    plan = prepare_judge_prompt(prompts, item, model_solution, token_counter=counter)
    recovery = plan.context_recovery
    assert recovery is not None
    assert plan.input_tokens == len(plan.prompt)
    assert plan.input_tokens <= 160
    assert plan.max_tokens == JUDGE_CONTEXT_RECOVERY_OUTPUT_TOKENS
    assert JUDGE_CONTEXT_TRUNCATION_MARKER in plan.prompt
    assert "A" * 10 in plan.prompt and "B" * 10 in plan.prompt
    assert "SECRET" not in plan.prompt
    assert recovery["original_solution_tokens"] == 400
    assert recovery["omitted_solution_tokens"] > 0
    assert recovery["kept_head_solution_tokens"] == (
        recovery["kept_solution_tokens"] + 1
    ) // 2
    assert recovery["kept_tail_solution_tokens"] == recovery["kept_solution_tokens"] // 2
    assert recovery["effective_prompt_sha256"] == hashlib.sha256(
        plan.prompt.encode()
    ).hexdigest()

    persisted = {
        "status": "completed",
        "context_recovery": recovery,
        "messages": [{"role": "user", "content": plan.prompt}],
    }
    validate_judge_context_recovery(
        persisted,
        prompts,
        item,
        model_solution,
        token_counter=counter,
    )
    corrupt = {**persisted, "context_recovery": {**recovery, "kept_solution_tokens": -1}}
    with pytest.raises(ValueError, match="metadata does not match"):
        validate_judge_context_recovery(
            corrupt,
            prompts,
            item,
            model_solution,
            token_counter=counter,
        )


def test_pass_best_and_cost_at_k_have_expected_toy_values() -> None:
    assert pass_at_k(3, 1, 1) == pytest.approx(1 / 3)
    assert pass_at_k(3, 1, 2) == pytest.approx(2 / 3)
    assert pass_at_k(3, 1, 3) == 1
    assert expected_best_at_k([0, 3, 7], 1) == pytest.approx(10 / 3)
    assert expected_best_at_k([0, 3, 7], 2) == pytest.approx(17 / 3)
    assert expected_best_at_k([0, 3, 7], 3) == 7
    assert expected_cost_at_k([10, 20, 30], 1) == 20
    assert expected_cost_at_k([10, 20, 30], 2) == 40
    assert expected_cost_at_k([10, 20, 30], 3) == 60

    with pytest.raises(ValueError, match="1 <= k <= n"):
        pass_at_k(3, 1, 4)
    with pytest.raises(ValueError, match="number of samples"):
        expected_best_at_k([1], 0)


def test_curves_apply_pass_best_and_token_cost_formulas() -> None:
    rows = [
        _result_row("imo_proof", "p", "direct", 0, 0, generated=10, total=11),
        _result_row("imo_proof", "p", "direct", 1, 3, generated=20, total=22),
        _result_row("imo_proof", "p", "direct", 2, 7, generated=30, total=33),
    ]
    for row in rows:
        row["invalidated_solve_usage"] = {
            "completion_tokens": 5,
            "total_tokens": 7,
            "usage_exact": True,
        }
        row["invalidated_judge_usage"] = {
            "completion_tokens": 50,
            "total_tokens": 70,
            "usage_exact": True,
        }
        row["invalidated_usage"] = {
            "completion_tokens": 55,
            "total_tokens": 77,
            "usage_exact": True,
        }
    curves = {
        row["k"]: row for row in compute_curves(rows, ks=(1, 2, 3), expected_seeds=(0, 1, 2))
    }
    assert curves[1]["pass_at_k"] == pytest.approx(1 / 3)
    assert curves[2]["pass_at_k"] == pytest.approx(2 / 3)
    assert curves[3]["pass_at_k"] == 1
    assert curves[2]["best_grade_at_k"] == pytest.approx(17 / 3)
    assert curves[2]["generated_cost_at_k"] == 40
    assert curves[2]["total_cost_at_k"] == 44
    assert curves[2]["final_attempt_generated_cost_at_k"] == 40
    assert curves[2]["incurred_generated_cost_at_k"] == 50
    assert curves[2]["incurred_total_cost_at_k"] == 58
    assert curves[2]["all_incurred_token_counts_exact"] is True


def test_missing_scheduled_cells_are_materialized_and_score_zero() -> None:
    rows = [_result_row("imo_proof", "p", "direct", 0, 7, generated=12, total=20)]
    summary = summarize_metrics(rows, expected_seeds=(0, 1, 2))[0]
    assert summary["scheduled"] == 3
    assert summary["completed"] == 1
    assert summary["missing_or_failed"] == 2
    assert summary["status_counts"] == {"completed": 1, "missing": 2}
    assert summary["raw_success_rate"] == pytest.approx(1 / 3)
    assert summary["mean_grade"] == pytest.approx(7 / 3)
    assert summary["normalized_average_grade"] == pytest.approx(1 / 3)
    assert not summary["complete"]

    curves = {
        row["k"]: row for row in compute_curves(rows, expected_seeds=(0, 1, 2))
    }
    assert curves[1]["pass_at_k"] == pytest.approx(1 / 3)
    assert curves[3]["pass_at_k"] == 1
    assert curves[1]["generated_cost_at_k"] == pytest.approx(4)


def test_reference_condition_is_absent_and_rendered_na_for_answer_benchmark() -> None:
    rows = [
        _result_row("imo_proof", "p", "direct", 0, 7),
        _result_row("imo_proof", "p", "gvr_reference", 0, 7),
        _result_row("imo_answer", "a", "direct", 0, True),
        {
            "benchmark": "imo_answer",
            "item_id": "a",
            "condition": "gvr_reference",
            "seed": 0,
            "judge_status": "n/a",
            "applicable": False,
        },
    ]
    report = compute_metrics(rows, bootstrap_samples=20, ks=(1,), expected_seeds=(0,))
    compatibility_keys = {
        (row["benchmark"], row["condition"]) for row in report["compatibility"]
    }
    assert ("imo_answer", "gvr_reference") not in compatibility_keys
    assert ("imo_proof", "gvr_reference") in compatibility_keys

    markdown = render_markdown(report)
    oracle_line = next(
        line for line in markdown.splitlines() if line.startswith("| gvr_reference |")
    )
    assert oracle_line == "| gvr_reference | 100.00% | N/A |"


def test_answer_compatibility_uses_only_seed_zero() -> None:
    rows = [
        _result_row("imo_answer", "a", "direct", 0, False),
        _result_row("imo_answer", "a", "direct", 1, True),
    ]
    compatibility = compatibility_metrics(
        rows,
        answer_seed=0,
        expected_seeds=(0, 1),
    )[0]
    summary = summarize_metrics(rows, expected_seeds=(0, 1))[0]

    assert compatibility["metric"] == "accuracy_seed_0"
    assert compatibility["value"] == 0
    assert compatibility["percent"] == 0
    assert summary["answer_accuracy"] == pytest.approx(0.5)


def test_paired_bootstrap_is_deterministic_independent_of_input_order() -> None:
    rows = [
        _result_row("imo_answer", "a", "direct", 0, False),
        _result_row("imo_answer", "a", "gvr", 0, True),
        _result_row("imo_answer", "b", "direct", 0, True),
        _result_row("imo_answer", "b", "gvr", 0, False),
        _result_row("imo_answer", "c", "direct", 0, False),
        _result_row("imo_answer", "c", "gvr", 0, False),
    ]
    kwargs = {
        "comparisons": (("gvr", "direct"),),
        "samples": 500,
        "seed": 928,
        "ks": (1,),
        "expected_seeds": (0,),
    }
    first = paired_bootstrap(rows, **kwargs)
    second = paired_bootstrap(reversed(rows), **kwargs)

    assert first == second
    assert first
    assert all(row["paired_n"] == 3 for row in first)
    raw_success = next(row for row in first if row["metric"] == "raw_success_rate")
    assert raw_success["mean_delta"] == pytest.approx(0)
    assert raw_success["ci95_low"] <= 0 <= raw_success["ci95_high"]


def test_write_report_emits_reloadable_json_csv_and_markdown(tmp_path: Path) -> None:
    rows = [
        _result_row("imo_proof", "p", "direct", 0, 7),
        _result_row("imo_answer", "a", "direct", 0, True),
    ]
    output = tmp_path / "report"
    report = write_report(
        output,
        rows,
        bootstrap_samples=20,
        comparisons=(),
        ks=(1,),
        expected_seeds=(0,),
    )

    expected_artifacts = {
        "report.json",
        "report.md",
        "summary.csv",
        "curves.csv",
        "compatibility.csv",
        "bootstrap.csv",
    }
    assert {path.name for path in output.iterdir()} == expected_artifacts
    assert json.loads((output / "report.json").read_text(encoding="utf-8")) == report
    assert (output / "report.md").read_text(encoding="utf-8").startswith(
        "# IMO evaluation report\n\nStatus: **complete**."
    )
    with (output / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert {(row["benchmark"], row["condition"]) for row in summary_rows} == {
        ("imo_proof", "direct"),
        ("imo_answer", "direct"),
    }
    assert (output / "bootstrap.csv").read_text(encoding="utf-8") == ""

    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    assert load_jsonl(jsonl) == rows
