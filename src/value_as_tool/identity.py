"""Path-independent identity for every result-affecting experiment input."""

from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
from typing import Any

from .benchmarks import QED_NANO_COMMIT, QED_PROMPTS, QEDPromptSet
from .config import ExperimentConfig, sha256_json
from .judging import JUDGE_CONTEXT_RECOVERY_POLICY_SHA256
from .orchestrator import (
    GENERATOR_SYSTEM_PROMPT,
    GVR_FORCED_CANDIDATE_REMINDER,
    GVR_VERDICT_RECOVERY_SUFFIX,
    QUERY_SUCCESS_PROBABILITY_TOOL,
    RATIONALE_SCORE_REFERENCE_VERIFIER_SYSTEM_PROMPT,
    RATIONALE_SCORE_VALUE_SOLVER_SYSTEM_PROMPT,
    RATIONALE_SCORE_VALUE_VERIFIER_SYSTEM_PROMPT,
    RATIONALE_SCORE_VERDICT_RECOVERY_SUFFIX,
    RATIONALE_SCORE_VERIFIER_SYSTEM_PROMPT,
    REFERENCE_VERIFIER_SYSTEM_PROMPT,
    REVISER_SYSTEM_PROMPT,
    SPAWN_SUBAGENTS_TOOL,
    SUBAGENT_SYSTEM_PROMPT,
    SUBMIT_PROBABILITY_TOOL,
    SUBMIT_RATIONALE_SCORE_TOOL,
    SUBMIT_RATIONALE_SCORE_VERDICT_TOOL,
    SUBMIT_VERDICT_TOOL,
    VALUE_FORCED_FINAL_REMINDER,
    VALUE_SOLVER_SYSTEM_PROMPT,
    VALUE_VERIFIER_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
)
from .server import SUPPORTED_SGLANG_VERSION, SUPPORTED_VLLM_VERSION

EXPERIMENT_IDENTITY_SCHEMA_VERSION = 1
SOLVER_PROTOCOL_VERSION = "aletheia-style-gvr-and-value-tool-v4-rationale-score"


def package_source_sha256(root: str | Path | None = None) -> str:
    """Hash all shipped Python sources without depending on checkout location."""

    package_root = Path(root) if root is not None else Path(__file__).resolve().parent
    files = sorted(path for path in package_root.rglob("*.py") if path.is_file())
    if not files:
        raise RuntimeError(f"no Python sources found below {package_root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _dependency_identity() -> dict[str, Any]:
    """Bind the checked-in lockfile, with wheel metadata as a fallback."""

    project_root = Path(__file__).resolve().parents[2]
    files: dict[str, str] = {}
    for name in ("pyproject.toml", "uv.lock"):
        path = project_root / name
        if path.is_file():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if files:
        return {"source_files": files}
    try:
        distribution = metadata.metadata("value-as-tool")
        return {
            "installed_distribution": {
                "version": distribution.get("Version"),
                "requires_dist": sorted(distribution.get_all("Requires-Dist") or ()),
            }
        }
    except metadata.PackageNotFoundError:
        return {"installed_distribution": None}


def build_experiment_identity(config: ExperimentConfig) -> dict[str, Any]:
    """Bind configuration, exact prompts, protocol, and implementation bytes."""

    # Construction verifies the vendored prompt bytes against the pinned
    # upstream digests. This makes a tampered prompt fail before any run ID is
    # created, rather than silently creating results under an expected hash.
    QEDPromptSet()
    qed_files = {
        logical_name: {"file": filename, "sha256": digest}
        for logical_name, (filename, digest, _final_newline) in sorted(
            QED_PROMPTS.items()
        )
    }
    return {
        "schema_version": EXPERIMENT_IDENTITY_SCHEMA_VERSION,
        "experiment_config": config.to_dict(),
        "implementation": {
            "package_source_sha256": package_source_sha256(),
            "dependencies": _dependency_identity(),
            "backend_profiles": {
                "sglang": SUPPORTED_SGLANG_VERSION,
                "vllm": SUPPORTED_VLLM_VERSION,
                "invocation_bound_by_package_source": True,
            },
        },
        "qed_nano": {
            "commit": QED_NANO_COMMIT,
            "prompts": qed_files,
            "judge_context_recovery_policy_sha256": (
                JUDGE_CONTEXT_RECOVERY_POLICY_SHA256
            ),
        },
        "solver_protocol": {
            "version": SOLVER_PROTOCOL_VERSION,
            "system_prompts": {
                "generator": GENERATOR_SYSTEM_PROMPT,
                "gvr_forced_candidate_reminder": GVR_FORCED_CANDIDATE_REMINDER,
                "gvr_verdict_recovery_suffix": GVR_VERDICT_RECOVERY_SUFFIX,
                "reviser": REVISER_SYSTEM_PROMPT,
                "verifier": VERIFIER_SYSTEM_PROMPT,
                "reference_verifier": REFERENCE_VERIFIER_SYSTEM_PROMPT,
                "rationale_score_verifier": RATIONALE_SCORE_VERIFIER_SYSTEM_PROMPT,
                "rationale_score_reference_verifier": (
                    RATIONALE_SCORE_REFERENCE_VERIFIER_SYSTEM_PROMPT
                ),
                "rationale_score_verdict_recovery": (
                    RATIONALE_SCORE_VERDICT_RECOVERY_SUFFIX
                ),
                "subagent": SUBAGENT_SYSTEM_PROMPT,
                "value_forced_final_reminder": VALUE_FORCED_FINAL_REMINDER,
                "value_solver": VALUE_SOLVER_SYSTEM_PROMPT,
                "value_verifier": VALUE_VERIFIER_SYSTEM_PROMPT,
                "rationale_score_value_solver": RATIONALE_SCORE_VALUE_SOLVER_SYSTEM_PROMPT,
                "rationale_score_value_verifier": (
                    RATIONALE_SCORE_VALUE_VERIFIER_SYSTEM_PROMPT
                ),
            },
            "tools": {
                "submit_verdict": SUBMIT_VERDICT_TOOL,
                "spawn_subagents": SPAWN_SUBAGENTS_TOOL,
                "query_success_probability": QUERY_SUCCESS_PROBABILITY_TOOL,
                "submit_probability": SUBMIT_PROBABILITY_TOOL,
                "submit_rationale_score": SUBMIT_RATIONALE_SCORE_TOOL,
                "submit_rationale_score_verdict": SUBMIT_RATIONALE_SCORE_VERDICT_TOOL,
            },
        },
    }


def experiment_fingerprint(identity: dict[str, Any]) -> str:
    return sha256_json(identity)


__all__ = [
    "EXPERIMENT_IDENTITY_SCHEMA_VERSION",
    "SOLVER_PROTOCOL_VERSION",
    "build_experiment_identity",
    "experiment_fingerprint",
    "package_source_sha256",
]
