"""Deterministic trajectory schedules and stable Slurm sharding."""

from __future__ import annotations

import fcntl
import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from value_as_tool.benchmarks import BenchmarkItem, benchmark_spec
from value_as_tool.config import ExperimentConfig
from value_as_tool.harnesses import HarnessSpec, resolve_harness
from value_as_tool.schemas import Condition, TrajectoryRequest
from value_as_tool.storage import ArtifactMismatchError, atomic_write_json, canonical_json


def stable_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a deterministic non-negative 31-bit child/call seed."""

    material = canonical_json([base_seed, *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31)


@dataclass(frozen=True, slots=True)
class ScheduleItem:
    ordinal: int
    run_id: str
    benchmark: str
    problem_id: str
    condition: Condition | None
    seed: int
    problem: str
    problem_fingerprint: str
    harness_id: str | None = None
    harness_entrypoint: str | None = None
    harness_source_sha256: str | None = None
    harness_access: str | None = None
    conditioning_mode: str | None = None
    conditioning_pack_sha256: str | None = None
    reference_proof: str | None = None
    golden_answer: str | None = None
    rubric: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        harness_values = (
            self.harness_id,
            self.harness_entrypoint,
            self.harness_source_sha256,
            self.harness_access,
        )
        if any(value is not None for value in harness_values):
            if any(not isinstance(value, str) or not value for value in harness_values):
                raise ValueError("harness schedule identity must be complete")
            assert self.harness_source_sha256 is not None
            if len(self.harness_source_sha256) != 64 or any(
                character not in "0123456789abcdef" for character in self.harness_source_sha256
            ):
                raise ValueError("harness source hash must be a lowercase SHA-256 digest")
            if self.harness_access not in {
                "blind",
                "reference_assisted",
                "attempt_assisted",
                "attempt_and_reference_assisted",
            }:
                raise ValueError("unsupported harness access class")
        elif self.condition is None:
            raise ValueError("a schedule item requires a harness or legacy condition")
        if self.conditioning_mode is not None:
            if self.conditioning_mode not in {"solutions", "solution_summary", "thinking_summary"}:
                raise ValueError("unsupported conditioning mode")
            digest = self.conditioning_pack_sha256 or ""
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("conditioned schedules require a pack SHA-256")
        elif self.conditioning_pack_sha256 is not None:
            raise ValueError("conditioning pack requires a mode")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["condition"] = self.condition.value if self.condition is not None else None
        if self.conditioning_mode is None:
            value.pop("conditioning_mode")
            value.pop("conditioning_pack_sha256")
        # Keep legacy schema-v1 bytes and fingerprints stable.  Harness fields
        # are an additive extension emitted only for harness-native schedules.
        if self.harness_id is None:
            for key in (
                "harness_id",
                "harness_entrypoint",
                "harness_source_sha256",
                "harness_access",
            ):
                value.pop(key)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ScheduleItem:
        return cls(
            ordinal=int(value["ordinal"]),
            run_id=str(value["run_id"]),
            benchmark=str(value["benchmark"]),
            problem_id=str(value["problem_id"]),
            condition=(
                Condition(str(value["condition"])) if value.get("condition") is not None else None
            ),
            seed=int(value["seed"]),
            problem=str(value["problem"]),
            problem_fingerprint=str(value["problem_fingerprint"]),
            harness_id=_optional_string(value.get("harness_id")),
            harness_entrypoint=_optional_string(value.get("harness_entrypoint")),
            harness_source_sha256=_optional_string(value.get("harness_source_sha256")),
            harness_access=_optional_string(value.get("harness_access")),
            conditioning_mode=_optional_string(value.get("conditioning_mode")),
            conditioning_pack_sha256=_optional_string(value.get("conditioning_pack_sha256")),
            reference_proof=_optional_string(value.get("reference_proof")),
            golden_answer=_optional_string(value.get("golden_answer")),
            rubric=value.get("rubric"),
            metadata=dict(value.get("metadata") or {}),
        )

    def to_request(
        self, *, verifier_evidence: Mapping[str, Any] | None = None
    ) -> TrajectoryRequest:
        metadata = {
            **dict(self.metadata),
            "schedule_ordinal": self.ordinal,
            "run_id": self.run_id,
            "problem_fingerprint": self.problem_fingerprint,
            "golden_answer": self.golden_answer,
            "rubric": self.rubric,
        }
        if self.harness_id is not None:
            metadata.update(
                harness_id=self.harness_id,
                harness_entrypoint=self.harness_entrypoint,
                harness_source_sha256=self.harness_source_sha256,
                harness_access=self.harness_access,
            )
        if self.conditioning_mode is not None:
            metadata.update(
                conditioning_mode=self.conditioning_mode,
                conditioning_pack_sha256=self.conditioning_pack_sha256,
            )
        return TrajectoryRequest(
            benchmark=self.benchmark,
            problem_id=self.problem_id,
            problem=self.problem,
            condition=self.condition,
            seed=self.seed,
            harness_id=self.harness_id,
            reference_proof=self.reference_proof,
            metadata=metadata,
            verifier_evidence=verifier_evidence,
        )


@dataclass(frozen=True, slots=True)
class Schedule(Sequence[ScheduleItem]):
    config_fingerprint: str
    items: tuple[ScheduleItem, ...]
    schema_version: int = 1

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index):  # type: ignore[no-untyped-def]
        return self.items[index]

    def __iter__(self) -> Iterator[ScheduleItem]:
        return iter(self.items)

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "schema_version": self.schema_version,
                "config_fingerprint": self.config_fingerprint,
                "items": [item.to_dict() for item in self.items],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_fingerprint": self.config_fingerprint,
            "schedule_fingerprint": self.fingerprint,
            "item_count": len(self.items),
            "items": [item.to_dict() for item in self.items],
        }


@dataclass(frozen=True, slots=True)
class _Problem:
    benchmark: str
    problem_id: str
    problem: str
    reference_proof: str | None
    golden_answer: str | None
    rubric: Any
    metadata: Mapping[str, Any]

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "benchmark": self.benchmark,
                "problem_id": self.problem_id,
                "problem": self.problem,
                "reference_proof": self.reference_proof,
                "golden_answer": self.golden_answer,
                "rubric": self.rubric,
            }
        )

    @property
    def supports_reference_verification(self) -> bool:
        try:
            is_proof = benchmark_spec(self.benchmark).kind == "proof"
        except ValueError:
            is_proof = self.golden_answer is None
        return is_proof and bool(self.reference_proof and self.reference_proof.strip())


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _normalize_problem(outer_benchmark: str, item: Any, index: int) -> _Problem:
    if isinstance(item, BenchmarkItem):
        return _Problem(
            benchmark=item.benchmark,
            problem_id=item.item_id,
            problem=item.problem,
            reference_proof=item.solution,
            golden_answer=item.answer,
            rubric=item.rubric,
            metadata={},
        )
    if isinstance(item, TrajectoryRequest):
        return _Problem(
            benchmark=item.benchmark,
            problem_id=item.problem_id,
            problem=item.problem,
            reference_proof=item.reference_proof,
            golden_answer=_optional_string(item.metadata.get("golden_answer")),
            rubric=item.metadata.get("rubric"),
            metadata=dict(item.metadata),
        )
    if isinstance(item, Mapping):
        benchmark = str(item.get("benchmark", outer_benchmark))
        problem_value = item.get("problem", item.get("statement"))
        if not isinstance(problem_value, str) or not problem_value:
            raise ValueError(f"{benchmark} item {index} has no non-empty problem text")
        problem_id = item.get("item_id", item.get("problem_id", item.get("id")))
        if problem_id is None:
            problem_id = f"row-{index:05d}-{stable_fingerprint(problem_value)[:12]}"
        reference = item.get("solution", item.get("reference_proof", item.get("reference")))
        answer = item.get("answer", item.get("golden_answer"))
        return _Problem(
            benchmark=benchmark,
            problem_id=str(problem_id),
            problem=problem_value,
            reference_proof=_optional_string(reference),
            golden_answer=_optional_string(answer),
            rubric=item.get("rubric"),
            metadata=dict(item.get("metadata") or {}),
        )
    # Lightweight support for dataclass/ORM-style prepared records.
    problem_value = getattr(item, "problem", getattr(item, "statement", None))
    if not isinstance(problem_value, str) or not problem_value:
        raise TypeError(f"unsupported scheduled problem type: {type(item).__name__}")
    problem_id = getattr(item, "item_id", getattr(item, "problem_id", None))
    if problem_id is None:
        problem_id = f"row-{index:05d}-{stable_fingerprint(problem_value)[:12]}"
    return _Problem(
        benchmark=str(getattr(item, "benchmark", outer_benchmark)),
        problem_id=str(problem_id),
        problem=problem_value,
        reference_proof=_optional_string(
            getattr(item, "solution", getattr(item, "reference_proof", None))
        ),
        golden_answer=_optional_string(
            getattr(item, "answer", getattr(item, "golden_answer", None))
        ),
        rubric=getattr(item, "rubric", None),
        metadata=dict(getattr(item, "metadata", {}) or {}),
    )


def _condition(value: Condition | str) -> Condition:
    return value if isinstance(value, Condition) else Condition(value)


def _resolve_harnesses(entrypoints: Iterable[str]) -> tuple[HarnessSpec, ...]:
    specs = tuple(resolve_harness(str(entrypoint)) for entrypoint in entrypoints)
    if not specs:
        raise ValueError("harnesses must be non-empty")
    ids = [spec.harness_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("selected harnesses must have unique harness IDs")
    for spec in specs:
        if not spec.entrypoint or not spec.source_sha256:
            raise ValueError(f"harness {spec.harness_id!r} has incomplete identity")
    return specs


def _benchmark_order(keys: Iterable[str]) -> list[str]:
    preferred = ("imo_proof", "proofbench", "imo_answer")
    key_set = set(keys)
    return [key for key in preferred if key in key_set] + sorted(key_set - set(preferred))


def build_schedule(
    problems_by_benchmark: Mapping[str, Sequence[Any]],
    config: ExperimentConfig,
    *,
    conditions: Iterable[Condition | str] | None = None,
    harnesses: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    config_fingerprint: str | None = None,
    conditioning_manifest: Mapping[str, Any] | None = None,
) -> Schedule:
    """Build the immutable full evaluation matrix.

    Iteration order is benchmark, source-row order, condition, then seed. The
    reference-assisted cell is absent (rather than recorded as a failure) for
    any item without a reference proof.
    """

    selected_fingerprint = (
        config.fingerprint if config_fingerprint is None else str(config_fingerprint)
    )
    if not selected_fingerprint:
        raise ValueError("config_fingerprint must not be empty")
    if conditions is not None and harnesses is not None:
        raise ValueError("conditions and harnesses are mutually exclusive selectors")
    configured_harnesses = config.evaluation.harnesses
    use_harnesses = harnesses is not None or (
        conditions is None and configured_harnesses is not None
    )
    selected_harnesses = (
        _resolve_harnesses(harnesses if harnesses is not None else configured_harnesses or ())
        if use_harnesses
        else ()
    )
    selected_conditions = (
        ()
        if use_harnesses
        else tuple(
            _condition(value)
            for value in (config.evaluation.conditions if conditions is None else conditions)
        )
    )
    selected_seeds = tuple(config.evaluation.seeds if seeds is None else seeds)
    if any(spec.conditioning_mode is not None for spec in selected_harnesses):
        if (
            config.conditioning is None
            or not config.conditioning.manifest_sha256
            or conditioning_manifest is None
        ):
            raise ValueError("conditioned evaluation requires a frozen conditioning manifest")
    if not use_harnesses and (
        not selected_conditions or len(set(selected_conditions)) != len(selected_conditions)
    ):
        raise ValueError("conditions must be non-empty and unique")
    if (
        not selected_seeds
        or len(set(selected_seeds)) != len(selected_seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in selected_seeds)
    ):
        raise ValueError("seeds must be non-empty unique integers")
    if config.conditioning is not None and set(selected_seeds) & set(
        config.conditioning.source_seeds
    ):
        raise ValueError("evaluation seeds must be disjoint from conditioning seeds")

    records: list[ScheduleItem] = []
    seen_identities: set[tuple[str, str, str, str, int]] = set()
    selected_benchmarks = set(config.evaluation.benchmarks)
    available_benchmarks = selected_benchmarks & set(problems_by_benchmark)
    for outer_benchmark in _benchmark_order(available_benchmarks):
        source_items = problems_by_benchmark[outer_benchmark]
        for index, source_item in enumerate(source_items):
            problem = _normalize_problem(outer_benchmark, source_item, index)
            methods: tuple[HarnessSpec | Condition, ...] = (
                selected_harnesses if use_harnesses else selected_conditions
            )
            for method in methods:
                if isinstance(method, HarnessSpec):
                    condition = method.condition
                    method_id = method.harness_id
                    method_source = method.source_sha256
                    requires_reference = method.requires_reference
                else:
                    condition = method
                    method_id = method.value
                    method_source = ""
                    requires_reference = condition in {
                        Condition.GVR_REFERENCE,
                        Condition.GVR_REFERENCE_RATIONALE_SCORE,
                    }
                if requires_reference and not problem.supports_reference_verification:
                    if isinstance(method, HarnessSpec) and method.conditioning_mode:
                        raise ValueError(
                            f"conditioned gold arm requires reference: {problem.problem_id}"
                        )
                    continue
                conditioning_mode = (
                    method.conditioning_mode if isinstance(method, HarnessSpec) else None
                )
                pack_sha256 = None
                if conditioning_mode is not None:
                    assert conditioning_manifest is not None
                    evidence_problem = conditioning_manifest["problems"][problem.benchmark][
                        problem.problem_id
                    ]
                    if evidence_problem["problem_fingerprint"] != problem.fingerprint:
                        raise ValueError(
                            f"conditioning problem fingerprint mismatch: {problem.problem_id}"
                        )
                    pack_sha256 = evidence_problem["packs"][conditioning_mode]["sha256"]
                for seed in selected_seeds:
                    identity = (
                        problem.benchmark,
                        problem.problem_id,
                        method_id,
                        method_source,
                        seed,
                    )
                    if identity in seen_identities:
                        raise ValueError(f"duplicate schedule identity: {identity!r}")
                    seen_identities.add(identity)
                    run_material = {
                        "config_fingerprint": selected_fingerprint,
                        "benchmark": problem.benchmark,
                        "problem_id": problem.problem_id,
                        "problem_fingerprint": problem.fingerprint,
                        "seed": seed,
                    }
                    if isinstance(method, HarnessSpec):
                        run_material.update(
                            harness_id=method.harness_id,
                            harness_entrypoint=method.entrypoint,
                            harness_source_sha256=method.source_sha256,
                            harness_access=method.access,
                            condition=(
                                method.condition.value if method.condition is not None else None
                            ),
                        )
                        if conditioning_mode is not None:
                            run_material.update(
                                conditioning_mode=conditioning_mode,
                                conditioning_pack_sha256=pack_sha256,
                            )
                    else:
                        # Preserve the legacy run-ID material exactly.
                        run_material["condition"] = condition.value
                    run_id = f"run-{stable_fingerprint(run_material)[:32]}"
                    records.append(
                        ScheduleItem(
                            ordinal=len(records),
                            run_id=run_id,
                            benchmark=problem.benchmark,
                            problem_id=problem.problem_id,
                            condition=condition,
                            seed=seed,
                            problem=problem.problem,
                            problem_fingerprint=problem.fingerprint,
                            harness_id=(
                                method.harness_id if isinstance(method, HarnessSpec) else None
                            ),
                            harness_entrypoint=(
                                method.entrypoint if isinstance(method, HarnessSpec) else None
                            ),
                            harness_source_sha256=(
                                method.source_sha256 if isinstance(method, HarnessSpec) else None
                            ),
                            harness_access=(
                                method.access if isinstance(method, HarnessSpec) else None
                            ),
                            reference_proof=problem.reference_proof,
                            conditioning_mode=conditioning_mode,
                            conditioning_pack_sha256=pack_sha256,
                            golden_answer=problem.golden_answer,
                            rubric=problem.rubric,
                            metadata=problem.metadata,
                        )
                    )
    return Schedule(config_fingerprint=selected_fingerprint, items=tuple(records))


def items_for_shard(
    items: Sequence[ScheduleItem] | Schedule,
    shard_index: int,
    shard_count: int,
) -> tuple[ScheduleItem, ...]:
    """Return the stable strided slice assigned to one array worker."""

    if isinstance(shard_count, bool) or shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if isinstance(shard_index, bool) or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    return tuple(items[shard_index::shard_count])


def shard_items(
    items: Sequence[ScheduleItem] | Schedule,
    shard_count: int,
) -> tuple[tuple[ScheduleItem, ...], ...]:
    return tuple(items_for_shard(items, index, shard_count) for index in range(shard_count))


def schedule_fingerprint(schedule: Schedule | Sequence[ScheduleItem]) -> str:
    if isinstance(schedule, Schedule):
        return schedule.fingerprint
    return stable_fingerprint([item.to_dict() for item in schedule])


def write_schedule(path: str | Path, schedule: Schedule) -> Path:
    """Write once, accepting an existing byte-equivalent schedule."""

    target = Path(path)
    record = schedule.to_dict()
    target.parent.mkdir(parents=True, exist_ok=True)
    with (target.parent / f".{target.name}.lock").open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        if target.exists():
            existing = load_schedule(target)
            if existing.fingerprint != schedule.fingerprint:
                raise ArtifactMismatchError(f"refusing to replace a different schedule: {target}")
            return target
        atomic_write_json(target, record)
    return target


def load_schedule(path: str | Path) -> Schedule:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise ValueError(f"unsupported schedule schema in {path}")
    raw_items = value.get("items")
    if not isinstance(raw_items, list):
        raise TypeError("schedule.items must be a list")
    schedule = Schedule(
        config_fingerprint=str(value["config_fingerprint"]),
        items=tuple(ScheduleItem.from_dict(item) for item in raw_items),
    )
    if value.get("item_count") != len(schedule):
        raise ArtifactMismatchError("schedule item_count does not match its records")
    if value.get("schedule_fingerprint") != schedule.fingerprint:
        raise ArtifactMismatchError("schedule fingerprint does not match its contents")
    if tuple(item.ordinal for item in schedule) != tuple(range(len(schedule))):
        raise ArtifactMismatchError("schedule ordinals are not contiguous")
    if len({item.run_id for item in schedule}) != len(schedule):
        raise ArtifactMismatchError("schedule contains duplicate run IDs")
    return schedule


__all__ = [
    "Schedule",
    "ScheduleItem",
    "build_schedule",
    "derive_seed",
    "items_for_shard",
    "load_schedule",
    "schedule_fingerprint",
    "shard_items",
    "stable_fingerprint",
    "write_schedule",
]
