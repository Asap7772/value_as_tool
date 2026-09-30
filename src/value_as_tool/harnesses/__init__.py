"""Discovery and validation for proof harness modules."""

from __future__ import annotations

import hashlib
import importlib
import inspect
from dataclasses import replace
from typing import Any

from value_as_tool.harnesses.attempt_conditioned import ATTEMPT_CONDITIONED_HARNESSES
from value_as_tool.harnesses.base import (
    ATTEMPT_CONDITIONING_MODES,
    HarnessRuntime,
    HarnessSpec,
    ProofHarness,
    module_source_path,
)

BUILTIN_HARNESSES: tuple[str, ...] = (
    "value_as_tool.harnesses.direct:AgentHarness",
    "value_as_tool.harnesses.gvr:AgentHarness",
    "value_as_tool.harnesses.gvr_subagents:AgentHarness",
    "value_as_tool.harnesses.gvr_reference:AgentHarness",
    "value_as_tool.harnesses.value_tool:AgentHarness",
    "value_as_tool.harnesses.gvr_rationale_score:AgentHarness",
    "value_as_tool.harnesses.value_tool_rationale_score:AgentHarness",
    "value_as_tool.harnesses.gvr_reference_rationale_score:AgentHarness",
    "value_as_tool.harnesses.cch_plan_work_review:AgentHarness",
)


def _load_class(entrypoint: str) -> type[Any]:
    module_name, separator, attribute = entrypoint.partition(":")
    if not separator or not module_name or not attribute or ":" in attribute:
        raise ValueError(
            f"harness entrypoint must use 'module:Class' syntax: {entrypoint!r}"
        )
    module = importlib.import_module(module_name)
    value = getattr(module, attribute, None)
    if not inspect.isclass(value):
        raise ValueError(f"harness entrypoint is not a class: {entrypoint!r}")
    if not isinstance(getattr(value, "spec", None), HarnessSpec):
        raise ValueError(f"harness class has no valid HarnessSpec: {entrypoint!r}")
    if not inspect.iscoroutinefunction(getattr(value, "run", None)):
        raise ValueError(f"harness run method must be async: {entrypoint!r}")
    return value


def resolve_harness(entrypoint: str) -> HarnessSpec:
    """Resolve immutable metadata and hash the selected harness source."""

    value = _load_class(entrypoint)
    source = module_source_path(value)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return replace(value.spec, entrypoint=entrypoint, source_sha256=digest)


def load_harness(entrypoint: str, *, source_sha256: str | None = None) -> ProofHarness:
    """Instantiate a validated harness and optionally verify its scheduled hash."""

    spec = resolve_harness(entrypoint)
    if source_sha256 is not None and spec.source_sha256 != source_sha256:
        raise ValueError(
            f"harness source hash changed for {entrypoint!r}: "
            f"{spec.source_sha256} != {source_sha256}"
        )
    value = _load_class(entrypoint)
    instance = value()
    instance.spec = spec
    if not isinstance(instance, ProofHarness):
        raise ValueError(f"harness does not satisfy ProofHarness: {entrypoint!r}")
    return instance


def builtin_entrypoint_for_condition(condition: str) -> str:
    suffix = condition.replace("-", "_")
    candidate = f"value_as_tool.harnesses.{suffix}:AgentHarness"
    if candidate not in BUILTIN_HARNESSES:
        raise ValueError(f"no built-in harness for condition {condition!r}")
    return candidate


__all__ = [
    "ATTEMPT_CONDITIONED_HARNESSES",
    "ATTEMPT_CONDITIONING_MODES",
    "BUILTIN_HARNESSES",
    "HarnessRuntime",
    "HarnessSpec",
    "ProofHarness",
    "builtin_entrypoint_for_condition",
    "load_harness",
    "resolve_harness",
]
