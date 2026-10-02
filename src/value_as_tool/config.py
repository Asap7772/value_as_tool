"""Strict, reproducible experiment configuration.

The checked-in YAML is deliberately the single source of experiment defaults.
This module adds type checking and canonical fingerprints without consulting
environment variables (API keys are looked up only by the HTTP client).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, StrictInt, field_validator, model_validator

Condition = Literal[
    "direct",
    "gvr",
    "gvr_subagents",
    "gvr_reference",
    "value_tool",
    "gvr_rationale_score",
    "value_tool_rationale_score",
    "gvr_reference_rationale_score",
]
Benchmark = Literal["imo_proof", "proofbench", "imo_answer", "arxivmath_train", "arxivmath_eval"]
# Benchmarks read from a locally prepared, hash-pinned JSONL instead of the Hub.
PREPARED_BENCHMARKS = ("arxivmath_train", "arxivmath_eval")
VerifierFeedbackMode = Literal["legacy", "rationale_score"]
SolverBackend = Literal["sglang", "vllm"]


def canonical_json(value: Any) -> str:
    """Return the stable JSON representation used by every fingerprint."""

    def encode(item: Any) -> str:
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"{type(item).__name__} is not JSON serializable")

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=encode,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class FrozenModel(BaseModel):
    """Base class that rejects misspelled configuration keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(FrozenModel):
    artifact_root: Path = Path("artifacts/default")
    asset_root: Path = Path("artifacts/assets")

    def resolve(self, relative_to: str | Path) -> PathsConfig:
        """Resolve relative paths against a caller-selected directory."""

        base = Path(relative_to).resolve()
        return PathsConfig(
            artifact_root=(base / self.artifact_root).resolve()
            if not self.artifact_root.is_absolute()
            else self.artifact_root.resolve(),
            asset_root=(base / self.asset_root).resolve()
            if not self.asset_root.is_absolute()
            else self.asset_root.resolve(),
        )


class ModelConfig(FrozenModel):
    name: str
    revision: str
    base_url: str
    api_key_env: str

    @field_validator("name", "revision", "base_url", "api_key_env")
    @classmethod
    def nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("revision")
    @classmethod
    def pinned_revision(cls, value: str) -> str:
        if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("revision must be a lowercase 40-character commit hash")
        return value

    @field_validator("base_url")
    @classmethod
    def valid_base_url(cls, value: str) -> str:
        normalized = value.rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP(S) URL")
        return normalized


class ModelsConfig(FrozenModel):
    solver: ModelConfig = ModelConfig(
        name="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        base_url="http://127.0.0.1:8000/v1",
        api_key_env="MODEL_API_KEY",
    )
    judge: ModelConfig = ModelConfig(
        name="openai/gpt-oss-20b",
        revision="6cee5e81ee83917806bbde320786a8fb61efebee",
        base_url="http://127.0.0.1:8001/v1",
        api_key_env="LLAMA_API_KEY",
    )

    def operational_base_url(
        self,
        role: Literal["solver", "judge"],
        environment: Mapping[str, str] | None = None,
    ) -> str:
        """Resolve a runtime-only endpoint without changing fingerprints."""

        environ = os.environ if environment is None else environment
        variable = f"VALUE_AS_TOOL_{role.upper()}_BASE_URL"
        configured = self.solver if role == "solver" else self.judge
        return environ.get(variable, configured.base_url).rstrip("/")


class DatasetConfig(FrozenModel):
    name: str
    revision: str
    split: str = "train"

    @field_validator("name", "split")
    @classmethod
    def nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("revision")
    @classmethod
    def pinned_revision(cls, value: str) -> str:
        if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("revision must be a lowercase 40-character commit hash")
        return value


class PreparedDatasetConfig(DatasetConfig):
    """A derived split built offline from a pinned Hub revision.

    ``path`` is relative to ``paths.asset_root`` and ``sha256`` pins the exact
    prepared file, so preparation never needs network access.
    """

    path: str
    sha256: str

    @field_validator("path")
    @classmethod
    def relative_path(cls, value: str) -> str:
        path = Path(value)
        if not value.strip() or path.is_absolute() or ".." in path.parts:
            raise ValueError("path must be a relative path inside the asset root")
        return value

    @field_validator("sha256")
    @classmethod
    def pinned_digest(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("sha256 must be a lowercase 64-character hex digest")
        return value


class DatasetsConfig(FrozenModel):
    imo_proof: DatasetConfig = DatasetConfig(
        name="lm-provers/IMOProofBench",
        revision="4b02dfebc3a956682398ce5847d56d0e820a758d",
    )
    proofbench: DatasetConfig = DatasetConfig(
        name="lm-provers/ProofBench",
        revision="c9ef8ca58c1711c4e3f1415755fc1a5286610c61",
    )
    imo_answer: DatasetConfig = DatasetConfig(
        name="Hwilner/imo-answerbench",
        revision="0258becbd00fc07d34862bc8539e61c8742f0d14",
    )
    # Optional so that experiments which do not select them keep their identity.
    arxivmath_train: PreparedDatasetConfig | None = None
    arxivmath_eval: PreparedDatasetConfig | None = None

    def items(self) -> tuple[tuple[str, DatasetConfig], ...]:
        return tuple((name, getattr(self, name)) for name in self.__class__.model_fields)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.items())


class SamplingConfig(FrozenModel):
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    enable_thinking: bool = True
    thinking_content_reserve_tokens: int = 8_192

    @model_validator(mode="after")
    def validate_sampling(self) -> SamplingConfig:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 <= self.top_p <= 1 or not 0 <= self.min_p <= 1:
            raise ValueError("top_p and min_p must be between zero and one")
        if self.top_k < -1:
            raise ValueError("top_k must be -1 or non-negative")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if self.thinking_content_reserve_tokens < 0:
            raise ValueError("thinking_content_reserve_tokens must be non-negative")
        if self.thinking_content_reserve_tokens and not self.enable_thinking:
            raise ValueError("thinking_content_reserve_tokens requires enable_thinking")
        return self

    def openai_kwargs(self) -> dict[str, Any]:
        """Arguments shared by Qwen's OpenAI-compatible backends."""

        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "extra_body": {
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            },
        }


class BudgetConfig(FrozenModel):
    generated_tokens: int = 229_376
    context_tokens: int = 262_144
    context_headroom_tokens: int = 1_024
    initial_generator_tokens: int = 98_304
    verifier_tokens: int = 32_768
    correction_pool_tokens: int = 98_304
    cch_stage_tokens: int | None = None
    max_candidate_versions: int = 3
    minimum_call_tokens: int = 1_024

    @field_validator("cch_stage_tokens", mode="before")
    @classmethod
    def validate_cch_stage_tokens(cls, value: Any) -> int | None:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError("cch_stage_tokens must be a positive integer or null")
        return value

    @model_validator(mode="after")
    def validate_budgets(self) -> BudgetConfig:
        values = self.model_dump()
        values.pop("cch_stage_tokens")
        if any(isinstance(value, bool) or value <= 0 for value in values.values()):
            raise ValueError("all token and cycle budgets must be positive integers")
        # The shared ledger spans multiple requests; only an individual request
        # must fit within the model's context window.
        if self.context_headroom_tokens >= self.context_tokens:
            raise ValueError("context_headroom_tokens must be smaller than context_tokens")
        if self.cch_stage_tokens is not None:
            if self.cch_stage_tokens < self.minimum_call_tokens:
                raise ValueError("cch_stage_tokens is smaller than minimum_call_tokens")
            if 7 * self.cch_stage_tokens > self.generated_tokens:
                raise ValueError(
                    "seven CCH stage allowances exceed the shared generated-token budget"
                )
        if self.max_candidate_versions != 3:
            raise ValueError("the fixed protocol requires exactly three candidate versions")
        reserved = (
            self.initial_generator_tokens + self.verifier_tokens + self.correction_pool_tokens
        )
        if reserved > self.generated_tokens:
            raise ValueError("role caps exceed the shared generated-token budget")
        required_correction_pool = (self.max_candidate_versions - 1) * (
            self.verifier_tokens + self.minimum_call_tokens
        )
        if self.correction_pool_tokens < required_correction_pool:
            raise ValueError(
                "correction pool cannot fit the minimum correction and verification "
                "calls for every configured candidate version"
            )
        if self.minimum_call_tokens > self.verifier_tokens:
            raise ValueError("minimum call size exceeds verifier cap")
        return self


class SubagentsConfig(FrozenModel):
    max_children: int = 3
    max_depth: int = 1
    child_tokens: int = 16_384
    final_candidate_reserve_tokens: int = 32_768
    max_context_chars: int = 4_000

    @model_validator(mode="after")
    def validate_subagents(self) -> SubagentsConfig:
        values = self.model_dump()
        if any(isinstance(value, bool) or value <= 0 for value in values.values()):
            raise ValueError("all subagent limits must be positive integers")
        if self.max_depth != 1:
            raise ValueError("the experiment permits exactly one level of subagents")
        if self.max_children != 3:
            raise ValueError("the fixed protocol permits exactly three subagents")
        return self


class ValueToolConfig(FrozenModel):
    max_queries: int = 3
    verifier_tokens: int = 32_768
    final_response_reserve_tokens: int = 32_768

    @field_validator(
        "max_queries",
        "verifier_tokens",
        "final_response_reserve_tokens",
        mode="before",
    )
    @classmethod
    def strict_integer(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("value-tool limits must be integers")
        return value

    @model_validator(mode="after")
    def validate_value_tool(self) -> ValueToolConfig:
        values = self.model_dump()
        if any(value <= 0 for value in values.values()):
            raise ValueError("all value-tool limits must be positive integers")
        return self


class VerifierFeedbackConfig(FrozenModel):
    """Select the information returned from a verifier to its solver."""

    mode: VerifierFeedbackMode = "legacy"


class EvaluationConfig(FrozenModel):
    conditions: tuple[Condition, ...] = (
        "direct",
        "gvr",
        "gvr_subagents",
        "gvr_reference",
    )
    # Harness entrypoints use the explicit ``module:attribute`` form.  This is
    # optional so existing condition-based experiment files retain byte-for-
    # byte equivalent semantics; when present it is the schedule's method
    # selector and ``conditions`` is only a legacy default.
    harnesses: tuple[str, ...] | None = None
    benchmarks: tuple[Benchmark, ...] = ("imo_proof", "proofbench", "imo_answer")
    seeds: tuple[int, ...] = (0, 1, 2)
    direct_answer_compatibility_seed: int = 0
    bootstrap_samples: int = 10_000
    judge_reasoning_effort: Literal["low", "medium", "high"] = "medium"
    judge_output_tokens: int = 95_000

    @field_validator("harnesses")
    @classmethod
    def validate_harness_entrypoints(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is None:
            return None
        if not value or len(value) != len(set(value)):
            raise ValueError("harnesses must be non-empty and unique when configured")
        for entrypoint in value:
            module, separator, attribute = entrypoint.partition(":")
            if (
                not separator
                or not module
                or not attribute
                or entrypoint != entrypoint.strip()
                or any(not part.isidentifier() for part in module.split("."))
                or not attribute.isidentifier()
            ):
                raise ValueError(
                    "harness entrypoints must use an explicit module:attribute import path"
                )
        return value

    @model_validator(mode="after")
    def validate_evaluation(self) -> EvaluationConfig:
        if len(self.conditions) != len(set(self.conditions)):
            raise ValueError("conditions must be unique")
        if self.harnesses is None and not self.conditions:
            raise ValueError("conditions must be non-empty without harnesses")
        if not self.benchmarks or len(self.benchmarks) != len(set(self.benchmarks)):
            raise ValueError("benchmarks must be non-empty and unique")
        if (
            not self.seeds
            or len(self.seeds) != len(set(self.seeds))
            or any(isinstance(seed, bool) for seed in self.seeds)
        ):
            raise ValueError("seeds must be non-empty unique integers")
        if self.direct_answer_compatibility_seed not in self.seeds:
            raise ValueError("direct answer compatibility seed must be one of evaluation.seeds")
        if self.bootstrap_samples <= 0 or self.judge_output_tokens <= 0:
            raise ValueError("evaluation sample/token counts must be positive")
        return self


class RuntimeConfig(FrozenModel):
    solver_backend: SolverBackend = "sglang"
    solver_tensor_parallel_size: int = 1
    request_timeout_seconds: float = 18_000
    max_retries: int = 3
    max_concurrency: int = 16
    solve_shards: int = 16
    judge_shards: int = 16
    resume: bool = True

    @model_validator(mode="after")
    def validate_runtime(self) -> RuntimeConfig:
        if self.request_timeout_seconds <= 0:
            raise ValueError("request timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if (
            min(
                self.solver_tensor_parallel_size,
                self.max_concurrency,
                self.solve_shards,
                self.judge_shards,
            )
            <= 0
        ):
            raise ValueError("concurrency and shard counts must be positive")
        return self


class ConditioningConfig(FrozenModel):
    """Immutable prior-attempt evidence and shared summary settings."""

    source_artifact_root: str
    bank_root: str
    bank_sha256: str | None = None
    manifest_sha256: str | None = None
    source_seeds: tuple[StrictInt, ...] = tuple(range(8))
    map_target_tokens: StrictInt = 2_048
    summary_target_tokens: StrictInt = 8_192
    max_summary_attempts: StrictInt = 3

    @field_validator("source_artifact_root", "bank_root")
    @classmethod
    def nonempty_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("conditioning paths must not be empty")
        return value

    @field_validator("bank_sha256", "manifest_sha256")
    @classmethod
    def digest(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("conditioning digest must be a lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_conditioning(self) -> ConditioningConfig:
        if not self.source_seeds or len(set(self.source_seeds)) != len(self.source_seeds):
            raise ValueError("conditioning source seeds must be nonempty and unique")
        if min(self.map_target_tokens, self.summary_target_tokens, self.max_summary_attempts) <= 0:
            raise ValueError("conditioning summary limits must be positive")
        if self.manifest_sha256 is not None and self.bank_sha256 is None:
            raise ValueError("frozen conditioning requires the source bank digest")
        return self


class ExperimentConfig(FrozenModel):
    schema_version: Literal[1] = 1
    paths: PathsConfig = PathsConfig()
    models: ModelsConfig = ModelsConfig()
    datasets: DatasetsConfig = DatasetsConfig()
    sampling: SamplingConfig = SamplingConfig()
    budget: BudgetConfig = BudgetConfig()
    subagents: SubagentsConfig = SubagentsConfig()
    value_tool: ValueToolConfig = ValueToolConfig()
    verifier_feedback: VerifierFeedbackConfig = VerifierFeedbackConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    conditioning: ConditioningConfig | None = None

    @model_validator(mode="after")
    def validate_cross_section_constraints(self) -> ExperimentConfig:
        if self.conditioning is not None and set(self.evaluation.seeds) & set(
            self.conditioning.source_seeds
        ):
            raise ValueError("evaluation seeds must be disjoint from conditioning seeds")
        unconfigured = [
            name for name in self.evaluation.benchmarks if getattr(self.datasets, name) is None
        ]
        if unconfigured:
            raise ValueError(f"selected benchmarks have no dataset entry: {unconfigured}")
        rationale_conditions = {
            "gvr_rationale_score",
            "value_tool_rationale_score",
            "gvr_reference_rationale_score",
        }
        legacy_verifier_conditions = {
            "gvr",
            "gvr_subagents",
            "gvr_reference",
            "value_tool",
        }
        # A harness owns its feedback protocol, so legacy and rationale-score
        # implementations can coexist in one harness-selected experiment.  The
        # global switch remains enforced for old condition-selected configs.
        if self.evaluation.harnesses is None:
            selected_conditions = set(self.evaluation.conditions)
            if self.verifier_feedback.mode == "rationale_score":
                if selected_conditions & legacy_verifier_conditions:
                    raise ValueError(
                        "rationale_score feedback requires explicitly versioned "
                        "rationale-score conditions"
                    )
            elif selected_conditions & rationale_conditions:
                raise ValueError(
                    "rationale-score conditions require verifier_feedback.mode=rationale_score"
                )
        subagent_need = (
            self.subagents.max_children * self.subagents.child_tokens
            + self.subagents.final_candidate_reserve_tokens
        )
        if subagent_need > self.budget.initial_generator_tokens:
            raise ValueError("subagent wave and synthesis reserves exceed initial Generator cap")
        if self.value_tool.verifier_tokens < self.budget.minimum_call_tokens:
            raise ValueError("value-tool verifier cap is smaller than the minimum call size")
        if self.value_tool.final_response_reserve_tokens < self.budget.minimum_call_tokens:
            raise ValueError(
                "value-tool final response reserve is smaller than the minimum call size"
            )
        value_tool_reserved = (
            self.value_tool.max_queries * self.value_tool.verifier_tokens
            + self.value_tool.final_response_reserve_tokens
            + self.budget.minimum_call_tokens
        )
        if value_tool_reserved > self.budget.generated_tokens:
            raise ValueError(
                "value-tool query caps, final response reserve, and minimum solver call "
                "exceed the shared generated-token budget"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        value = self.model_dump(mode="json")
        if value.get("conditioning") is None:
            value.pop("conditioning")
        if value["budget"].get("cch_stage_tokens") is None:
            # Preserve config identities for experiments using legacy CCH caps.
            value["budget"].pop("cch_stage_tokens")
        for name in PREPARED_BENCHMARKS:
            # Likewise for experiments created before prepared datasets existed.
            if value["datasets"].get(name) is None:
                value["datasets"].pop(name, None)
        evaluation = value.get("evaluation")
        if isinstance(evaluation, dict) and evaluation.get("harnesses") is None:
            # Preserve condition-only experiment identities created before the
            # additive harness selector existed.
            evaluation.pop("harnesses")
        return value

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.to_dict())


def _deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def _expand_dotted(values: Mapping[str, Any]) -> dict[str, Any]:
    expanded: dict[str, Any] = {}
    for key, value in values.items():
        cursor = expanded
        parts = key.split(".")
        if any(not part for part in parts):
            raise ValueError(f"invalid empty component in override {key!r}")
        for part in parts[:-1]:
            existing = cursor.setdefault(part, {})
            if not isinstance(existing, dict):
                raise ValueError(f"conflicting override path {key!r}")
            cursor = existing
        cursor[parts[-1]] = value
    return expanded


def load_config(
    path: str | Path | None = "experiment.yaml",
    overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    """Load strict YAML over stable defaults.

    Overrides may use nested mappings or dotted keys such as
    ``runtime.solve_shards``. Relative artifact paths remain relative so that
    fingerprints do not depend on the checkout location.
    """

    values: dict[str, Any] = ExperimentConfig().to_dict()
    if path is not None:
        source = Path(path)
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, Mapping):
            raise TypeError("configuration root must be a mapping")
        values = _deep_merge(values, loaded)
    if overrides:
        values = _deep_merge(values, _expand_dotted(overrides))
    return ExperimentConfig.model_validate(values)


def parse_overrides(entries: Sequence[str]) -> dict[str, Any]:
    """Parse CLI ``KEY=YAML_VALUE`` overrides without shell evaluation."""

    parsed: dict[str, Any] = {}
    for entry in entries:
        key, separator, raw = entry.partition("=")
        if not separator or not key:
            raise ValueError(f"override must have KEY=VALUE form: {entry!r}")
        parsed[key] = yaml.safe_load(raw)
    return parsed
