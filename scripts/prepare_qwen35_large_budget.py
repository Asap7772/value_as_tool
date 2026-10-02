"""Prepare an isolated experiment from verified, already available local assets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from value_as_tool.assets import (
    load_prepared_benchmarks,
    model_manifest_path,
    model_snapshot_path,
    prepared_benchmark_path,
)
from value_as_tool.config import ExperimentConfig
from value_as_tool.pipeline import load_context, prepare
from value_as_tool.storage import atomic_write_json


def local_model_manifest(destination_config: ExperimentConfig) -> dict[str, Any]:
    """Verify the pinned local model snapshots and record them without downloading."""

    manifest: dict[str, Any] = {"schema_version": 1, "models": {}}
    for role in ("solver", "judge"):
        model = getattr(destination_config.models, role)
        path = model_snapshot_path(
            destination_config.paths.asset_root, model.name, model.revision
        ).resolve()
        for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            if not (path / filename).is_file() or (path / filename).stat().st_size == 0:
                raise FileNotFoundError(f"missing pinned {role} asset: {path / filename}")
        tokenizer_config = json.loads((path / "tokenizer_config.json").read_text())
        if not (path / "chat_template.jinja").is_file() and not tokenizer_config.get(
            "chat_template"
        ):
            raise FileNotFoundError(f"missing pinned {role} chat template: {path}")
        index_path = path / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            shards = set(index["weight_map"].values())
            if not shards or any(Path(name).name != name for name in shards):
                raise ValueError(f"invalid pinned {role} weight index: {index_path}")
            expected_bytes = int(index.get("metadata", {}).get("total_size", 0))
        else:
            shards = {"model.safetensors"}
            expected_bytes = 1
        for name in shards:
            if not (path / name).is_file() or (path / name).stat().st_size == 0:
                raise FileNotFoundError(f"missing pinned {role} weight shard: {path / name}")
        if sum((path / name).stat().st_size for name in shards) < expected_bytes:
            raise ValueError(f"pinned {role} weight shards are smaller than their index")
        manifest["models"][role] = {
            "name": model.name,
            "revision": model.revision,
            "path": str(path),
            "tokenizer_only": False,
        }
    atomic_write_json(model_manifest_path(destination_config), manifest)
    return manifest


def prepare_local(config: str, source_artifact_root: Path) -> dict[str, Any]:
    context = load_context(config)
    source_paths = context.config.paths.model_copy(
        update={"artifact_root": source_artifact_root.resolve()}
    )
    source_config = context.config.model_copy(update={"paths": source_paths})
    # This checks dataset identity, revisions, split, hashes, and row contracts.
    load_prepared_benchmarks(source_config)

    def benchmarks(destination_config: ExperimentConfig) -> dict[str, Any]:
        source_manifest = source_artifact_root / "prepared" / "benchmarks" / "manifest.json"
        manifest = json.loads(source_manifest.read_text())
        for name, entry in manifest["benchmarks"].items():
            source = prepared_benchmark_path(source_artifact_root, name)
            destination = prepared_benchmark_path(destination_config.paths.artifact_root, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            entry["path"] = str(destination)
        atomic_write_json(
            destination_config.paths.artifact_root / "prepared" / "benchmarks" / "manifest.json",
            manifest,
        )
        return manifest


    return prepare(context, benchmark_preparer=benchmarks, model_preparer=local_model_manifest)


def prepare_offline(config: str) -> dict[str, Any]:
    """Prepare hash-pinned local datasets with verified local model snapshots."""

    return prepare(load_context(config), model_preparer=local_model_manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--source-artifact-root",
        type=Path,
        help="copy prepared benchmarks from this root; omit for prepared-JSONL datasets",
    )
    args = parser.parse_args()
    result = (
        prepare_local(args.config, args.source_artifact_root)
        if args.source_artifact_root is not None
        else prepare_offline(args.config)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
