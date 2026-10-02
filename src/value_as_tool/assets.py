"""Pinned model and benchmark materialization.

Nothing in this module downloads data at import time.  The network-capable
helpers run only when the user explicitly invokes ``prepare``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from .benchmarks import (
    BENCHMARKS,
    BenchmarkItem,
    load_benchmark,
    read_benchmark_jsonl,
    row_to_item,
)
from .config import ExperimentConfig
from .storage import atomic_write_json, atomic_write_text

TOKENIZER_ALLOW_PATTERNS = (
    "*.json",
    "*.model",
    "*.py",
    "*.tiktoken",
    "*.txt",
    "chat_template*",
    "merges*",
    "tokenizer*",
    "vocab*",
)


def _safe_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepared_benchmark_path(artifact_root: str | Path, benchmark: str) -> Path:
    return Path(artifact_root) / "prepared" / "benchmarks" / f"{benchmark}.jsonl"


def model_snapshot_path(asset_root: str | Path, model: str, revision: str) -> Path:
    return Path(asset_root) / "models" / _safe_repo_id(model) / revision


def model_manifest_path(config: ExperimentConfig) -> Path:
    """Return the experiment-scoped model-role manifest path."""

    return config.paths.artifact_root / "prepared" / "models" / "manifest.json"


def _load_prepared_source(config: ExperimentConfig, name: str) -> list[BenchmarkItem]:
    """Read a hash-pinned, locally built split without network access."""

    source = getattr(config.datasets, name)
    path = config.paths.asset_root / source.path
    if _sha256(path) != source.sha256:
        raise ValueError(f"prepared dataset hash mismatch for {name}: {path}")
    configured_spec = replace(
        BENCHMARKS[name],
        dataset=source.name,
        revision=source.revision,
        split=source.split,
    )
    return read_benchmark_jsonl(path, configured_spec, check_size=True)


def prepare_benchmark_assets(config: ExperimentConfig) -> dict[str, Any]:
    """Download and normalize the experiment's selected benchmark snapshots."""

    root = config.paths.artifact_root
    cache = config.paths.asset_root / "huggingface"
    manifest: dict[str, Any] = {"schema_version": 1, "benchmarks": {}}
    for name in config.evaluation.benchmarks:
        schema = BENCHMARKS[name]
        source = getattr(config.datasets, name)
        if schema.source == "prepared_jsonl":
            items = _load_prepared_source(config, name)
        else:
            configured_spec = replace(
                schema,
                dataset=source.name,
                revision=source.revision,
                split=source.split,
            )
            items = load_benchmark(configured_spec, cache_dir=cache, check_size=True)
        destination = prepared_benchmark_path(root, name)
        payload = "".join(
            json.dumps(item.raw, ensure_ascii=False, sort_keys=True, default=str) + "\n"
            for item in items
        )
        atomic_write_text(destination, payload)
        manifest["benchmarks"][name] = {
            "dataset": source.name,
            "revision": source.revision,
            "split": source.split,
            "rows": len(items),
            "sha256": _sha256(destination),
            "path": str(destination),
        }
    atomic_write_json(root / "prepared" / "benchmarks" / "manifest.json", manifest)
    return manifest


def load_prepared_benchmarks(config: ExperimentConfig) -> dict[str, list[BenchmarkItem]]:
    """Load normalized items while validating hashes and row contracts."""

    root = config.paths.artifact_root / "prepared" / "benchmarks"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("benchmarks"), dict):
        raise ValueError(f"invalid benchmark manifest: {manifest_path}")
    result: dict[str, list[BenchmarkItem]] = {}
    for name in config.evaluation.benchmarks:
        spec = BENCHMARKS[name]
        path = prepared_benchmark_path(config.paths.artifact_root, name)
        entry = manifest["benchmarks"].get(name)
        source = getattr(config.datasets, name)
        expected_source = {
            "dataset": source.name,
            "revision": source.revision,
            "split": source.split,
        }
        if not isinstance(entry, dict):
            raise ValueError(f"prepared benchmark manifest entry is missing: {name}")
        for field, expected in expected_source.items():
            if entry.get(field) != expected:
                raise ValueError(
                    f"prepared benchmark {name} {field} does not match config: "
                    f"{entry.get(field)!r} != {expected!r}"
                )
        if entry.get("sha256") != _sha256(path):
            raise ValueError(f"prepared benchmark hash mismatch: {name}")
        rows: list[BenchmarkItem] = []
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if line.strip():
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise TypeError(f"expected object at {path}:{index + 1}")
                    rows.append(row_to_item(spec, raw, index))
        if len(rows) != spec.expected_rows:
            raise ValueError(
                f"{name} expected {spec.expected_rows} prepared rows, found {len(rows)}"
            )
        result[name] = rows
    return result


def prepare_model_assets(
    config: ExperimentConfig,
    *,
    tokenizer_only: bool = False,
) -> dict[str, Any]:
    """Download pinned model snapshots, or only tokenizer/configuration files."""

    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or None
    manifest: dict[str, Any] = {"schema_version": 1, "models": {}}
    for role in ("solver", "judge"):
        model = getattr(config.models, role)
        destination = model_snapshot_path(config.paths.asset_root, model.name, model.revision)
        kwargs: dict[str, Any] = {
            "repo_id": model.name,
            "revision": model.revision,
            "local_dir": destination,
            "token": token,
        }
        if tokenizer_only:
            kwargs["allow_patterns"] = list(TOKENIZER_ALLOW_PATTERNS)
        resolved = snapshot_download(**kwargs)
        manifest["models"][role] = {
            "name": model.name,
            "revision": model.revision,
            "path": str(Path(resolved).resolve()),
            "tokenizer_only": tokenizer_only,
        }
    atomic_write_json(model_manifest_path(config), manifest)
    return manifest


def load_model_manifest(config: ExperimentConfig) -> dict[str, Any]:
    path = model_manifest_path(config)
    if not path.exists():
        # Read-only compatibility for artifacts prepared before manifests were
        # scoped per experiment. New prepares never overwrite this shared file.
        path = config.paths.asset_root / "models" / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"invalid model manifest: {path}")
    return value


__all__ = [
    "load_model_manifest",
    "load_prepared_benchmarks",
    "model_snapshot_path",
    "model_manifest_path",
    "prepare_benchmark_assets",
    "prepare_model_assets",
    "prepared_benchmark_path",
    "TOKENIZER_ALLOW_PATTERNS",
]
