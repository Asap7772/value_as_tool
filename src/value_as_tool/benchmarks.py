"""Pinned benchmark loaders and byte-compatible QED-Nano prompt rendering.

The module deliberately imports :mod:`datasets` only inside ``load_benchmark``.
Importing the harness, inspecting a schedule, or generating a report therefore
never contacts Hugging Face.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

QED_NANO_COMMIT = "02a4699ed380f8e980a8d44c755b8bfd4e310718"
QED_COMMIT = QED_NANO_COMMIT

# Digests are for the upstream bytes at QED_NANO_COMMIT.  Four upstream files
# omit their final newline; PromptSet.read normalizes our packaged text to that
# exact representation before checking or rendering it.
QED_PROMPTS: dict[str, tuple[str, str, bool]] = {
    "proof_run": (
        "proofbench_run.txt",
        "01f6319829c11a11a554fbc5a7a2d81e5d79b17711c3eb21e698a7e28ca94355",
        False,
    ),
    "answer_run": (
        "answerbench_run.txt",
        "9be7ab551a3a9659b35896beba3d44e5d3ddbfdee2450634a4bf2c1257f28d55",
        False,
    ),
    "answer_judge": (
        "answerbench.txt",
        "77852b856741cd08de5e0de49f97aacfacd3e2453a561e71d610bfae0c8703a2",
        False,
    ),
    "imo_proof_judge": (
        "proofbench.txt",
        "1f05fe22b17b20bf670daf6faad21e7904827e86002b54fe420513bf2b065cbd",
        False,
    ),
    "proofbench_judge": (
        "proofbench_proofbench.txt",
        "773236ab6cf1bfccfb821657b6fd75801816fc75e85cd6f9d764656328e958af",
        True,
    ),
}


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """Immutable source and schema contract for one benchmark."""

    name: str
    display_name: str
    dataset: str
    revision: str
    expected_rows: int
    split: str = "train"
    kind: str = "proof"
    problem_columns: tuple[str, ...] = ("problem",)
    solution_columns: tuple[str, ...] = ("solution",)
    answer_columns: tuple[str, ...] = ("answer",)
    rubric_columns: tuple[str, ...] = ()
    judge_prompt: str = "imo_proof_judge"
    reference_verifier: bool = True


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "imo_proof": BenchmarkSpec(
        name="imo_proof",
        display_name="IMO-ProofBench",
        dataset="lm-provers/IMOProofBench",
        revision="4b02dfebc3a956682398ce5847d56d0e820a758d",
        expected_rows=60,
        rubric_columns=("grading_guidelines",),
        judge_prompt="imo_proof_judge",
    ),
    "proofbench": BenchmarkSpec(
        name="proofbench",
        display_name="ProofBench",
        dataset="lm-provers/ProofBench",
        revision="c9ef8ca58c1711c4e3f1415755fc1a5286610c61",
        expected_rows=145,
        rubric_columns=("grading_scheme",),
        judge_prompt="proofbench_judge",
    ),
    "imo_answer": BenchmarkSpec(
        name="imo_answer",
        display_name="IMO-AnswerBench",
        dataset="Hwilner/imo-answerbench",
        revision="0258becbd00fc07d34862bc8539e61c8742f0d14",
        expected_rows=400,
        kind="answer",
        problem_columns=("Problem", "problem"),
        answer_columns=("Short Answer", "answer"),
        solution_columns=(),
        judge_prompt="answer_judge",
        reference_verifier=False,
    ),
}
DEFAULT_BENCHMARKS = BENCHMARKS

BENCHMARK_ALIASES = {
    "imo_proof": "imo_proof",
    "imo-proofbench": "imo_proof",
    "imoproofbench": "imo_proof",
    "lm-provers/imoproofbench": "imo_proof",
    "proofbench": "proofbench",
    "lm-provers/proofbench": "proofbench",
    "imo_answer": "imo_answer",
    "imo-answerbench": "imo_answer",
    "imoanswerbench": "imo_answer",
    "imobench-finalanswer": "imo_answer",
    "hwilner/imo-answerbench": "imo_answer",
}


@dataclass(frozen=True, slots=True)
class BenchmarkItem:
    """Canonical item passed to solvers and external judges."""

    benchmark: str
    item_id: str
    problem: str
    raw: dict[str, Any]
    solution: str | None = None
    answer: str | None = None
    rubric: Any = None

    @property
    def is_proof(self) -> bool:
        return benchmark_spec(self.benchmark).kind == "proof"

    @property
    def supports_reference_verification(self) -> bool:
        return self.is_proof and bool(self.solution and self.solution.strip())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark_spec(name: str | BenchmarkSpec) -> BenchmarkSpec:
    if isinstance(name, BenchmarkSpec):
        return name
    key = BENCHMARK_ALIASES.get(str(name).casefold())
    if key is None:
        choices = ", ".join(spec.display_name for spec in BENCHMARKS.values())
        raise ValueError(f"unknown benchmark {name!r}; expected one of: {choices}")
    return BENCHMARKS[key]


def _first_present(row: Mapping[str, Any], columns: Iterable[str]) -> Any:
    for column in columns:
        value = row.get(column)
        if value is not None and value != "":
            return value
    return None


def _stable_item_id(row: Mapping[str, Any], index: int, problem: str) -> str:
    for key in (
        "item_id",
        "question_id",
        "problem_id",
        "Problem ID",
        "id",
        "name",
        "uuid",
        "source_id",
    ):
        value = row.get(key)
        if value is not None and value != "":
            return str(value)
    digest = hashlib.sha256(problem.encode("utf-8")).hexdigest()[:12]
    return f"row-{index:05d}-{digest}"


def row_to_item(
    spec: str | BenchmarkSpec,
    row: Mapping[str, Any],
    index: int,
) -> BenchmarkItem:
    """Normalize a dataset row without discarding its source schema."""

    resolved = benchmark_spec(spec)
    problem_value = _first_present(row, resolved.problem_columns)
    if problem_value is None:
        raise ValueError(
            f"{resolved.dataset} row {index} has no non-empty problem column; "
            f"available={sorted(row)}"
        )
    problem = str(problem_value)
    solution_value = _first_present(row, resolved.solution_columns)
    answer_value = _first_present(row, resolved.answer_columns)
    rubric = _first_present(row, resolved.rubric_columns)
    if resolved.kind == "proof" and solution_value is None:
        raise ValueError(f"{resolved.dataset} row {index} has no reference solution")
    if resolved.kind == "proof" and rubric is None:
        raise ValueError(f"{resolved.dataset} row {index} has no grading rubric")
    if resolved.kind == "answer" and answer_value is None:
        raise ValueError(f"{resolved.dataset} row {index} has no golden answer")
    raw = dict(row)
    return BenchmarkItem(
        benchmark=resolved.name,
        item_id=_stable_item_id(raw, index, problem),
        problem=problem,
        raw=raw,
        solution=None if solution_value is None else str(solution_value),
        answer=None if answer_value is None else str(answer_value),
        rubric=rubric,
    )


def validate_benchmark(
    spec: str | BenchmarkSpec,
    items: Iterable[BenchmarkItem],
    *,
    check_size: bool = True,
) -> list[BenchmarkItem]:
    resolved = benchmark_spec(spec)
    materialized = list(items)
    if check_size and len(materialized) != resolved.expected_rows:
        raise ValueError(
            f"{resolved.display_name} expected {resolved.expected_rows} rows at "
            f"revision {resolved.revision}, found {len(materialized)}"
        )
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in materialized:
        if item.item_id in seen:
            duplicates.add(item.item_id)
        seen.add(item.item_id)
    if duplicates:
        raise ValueError(
            f"{resolved.display_name} contains duplicate item IDs: {sorted(duplicates)[:5]}"
        )
    return materialized


def load_benchmark(
    spec: str | BenchmarkSpec,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    check_size: bool = True,
) -> list[BenchmarkItem]:
    """Load a pinned Hugging Face split and validate its public contract."""

    from datasets import load_dataset

    resolved = benchmark_spec(spec)
    selected_revision = revision or resolved.revision
    kwargs: dict[str, Any] = {
        "split": resolved.split,
        "revision": selected_revision,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    dataset = load_dataset(resolved.dataset, **kwargs)
    items = [row_to_item(resolved, dict(row), index) for index, row in enumerate(dataset)]
    return validate_benchmark(resolved, items, check_size=check_size)


def load_all_benchmarks(
    *,
    cache_dir: str | Path | None = None,
    check_size: bool = True,
) -> dict[str, list[BenchmarkItem]]:
    """Load all three pinned datasets. This is the only network-capable helper."""

    return {
        name: load_benchmark(spec, cache_dir=cache_dir, check_size=check_size)
        for name, spec in BENCHMARKS.items()
    }


def read_benchmark_jsonl(
    path: str | Path,
    spec: str | BenchmarkSpec,
    *,
    check_size: bool = True,
) -> list[BenchmarkItem]:
    """Read a prepared standalone snapshot without Hugging Face access."""

    rows: list[BenchmarkItem] = []
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                rows.append(row_to_item(spec, json.loads(line), index))
    return validate_benchmark(spec, rows, check_size=check_size)


def default_prompt_root() -> Path:
    package_root = Path(__file__).resolve().parent / "prompts" / "qed_nano"
    if package_root.is_dir():
        return package_root
    return Path(__file__).resolve().parents[2] / "prompts" / "qed_nano"


class QEDPromptSet:
    """Vendored QED-Nano solve and external-judge templates."""

    def __init__(self, root: str | Path | None = None, *, verify: bool = True):
        self.root = Path(root) if root is not None else default_prompt_root()
        self.root = self.root.resolve()
        if verify:
            self.verify()

    @staticmethod
    def _source_bytes(path: Path, has_final_newline: bool) -> bytes:
        value = path.read_bytes()
        if not has_final_newline and value.endswith(b"\n"):
            value = value[:-1]
        return value

    def verify(self) -> None:
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("commit") != QED_NANO_COMMIT:
            raise RuntimeError("vendored QED-Nano manifest has an unexpected commit")
        for logical_name, (filename, digest, has_final_newline) in QED_PROMPTS.items():
            path = self.root / filename
            actual = hashlib.sha256(
                self._source_bytes(path, has_final_newline)
            ).hexdigest()
            if actual != digest:
                raise RuntimeError(
                    f"QED-Nano prompt checksum mismatch for {logical_name}: {path}"
                )
            if manifest.get("files", {}).get(filename) != digest:
                raise RuntimeError(f"QED-Nano manifest mismatch for {filename}")

    def read(self, name: str) -> str:
        try:
            filename, _, has_final_newline = QED_PROMPTS[name]
        except KeyError as exc:
            raise ValueError(f"unknown QED-Nano prompt: {name}") from exc
        return self._source_bytes(self.root / filename, has_final_newline).decode("utf-8")

    def solve_prompt(self, item: BenchmarkItem) -> str:
        prompt_name = "proof_run" if item.is_proof else "answer_run"
        return self.read(prompt_name).format(problem_statement=item.problem)

    def judge_prompt(self, item: BenchmarkItem, model_solution: str) -> str:
        # Local import avoids a benchmarks <-> judging import cycle.
        from value_as_tool.judging import extract_candidate_answer, remove_self_evaluation

        spec = benchmark_spec(item.benchmark)
        candidate = remove_self_evaluation(model_solution)
        template = self.read(spec.judge_prompt)
        if spec.kind == "answer":
            return template.format(
                problem_statement=item.problem,
                student_answer=extract_candidate_answer(candidate),
                gold_answer=item.answer,
            )
        if spec.name == "proofbench":
            return template.format(
                problem_statement=item.problem,
                guidelines=item.rubric,
                student_answer=candidate,
            )
        return template.format(
            problem_statement=item.problem,
            solution=item.solution,
            guidelines=item.rubric,
            student_answer=candidate,
        )


def judge_uses_reference(item: BenchmarkItem) -> bool:
    """Whether the external QED prompt contains a reference proof."""

    return benchmark_spec(item.benchmark).judge_prompt == "imo_proof_judge"


def judge_prompt_name(item: BenchmarkItem) -> str:
    return benchmark_spec(item.benchmark).judge_prompt


PROMPTS = {
    name: (filename, digest)
    for name, (filename, digest, _has_final_newline) in QED_PROMPTS.items()
}


__all__ = [
    "BENCHMARKS",
    "DEFAULT_BENCHMARKS",
    "BenchmarkItem",
    "BenchmarkSpec",
    "QED_NANO_COMMIT",
    "QED_COMMIT",
    "QED_PROMPTS",
    "PROMPTS",
    "QEDPromptSet",
    "benchmark_spec",
    "default_prompt_root",
    "judge_uses_reference",
    "judge_prompt_name",
    "load_all_benchmarks",
    "load_benchmark",
    "read_benchmark_jsonl",
    "row_to_item",
    "validate_benchmark",
]
