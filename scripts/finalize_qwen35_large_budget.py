"""Run the immutable finalizer with a separately recorded audit correction."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


def verified_audit_path(manifest: dict[str, Any]) -> Path:
    override = manifest["audit_override"]
    path = Path(override["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != override["sha256"]:
        raise RuntimeError("recorded audit correction has changed")
    return path


def load_launcher(manifest: dict[str, Any]):
    source = Path(manifest["source"])
    hashes = json.loads((Path(manifest["run_root"]) / "snapshot-files.json").read_text())
    for name, expected in hashes.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"immutable source changed: {name}")
    spec = importlib.util.spec_from_file_location(
        "pinned_qwen35_launcher", source / "scripts/submit_qwen35_large_budget.py",
    )
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    return launcher


def install_audit_override(launcher, manifest: dict[str, Any]) -> None:
    corrected_path = verified_audit_path(manifest)
    original = launcher.audit_command

    def audit_command(current_manifest: dict[str, Any], *, pilot: bool) -> list[str]:
        command = original(current_manifest, pilot=pilot)
        command[1] = str(corrected_path)
        return command

    launcher.audit_command = audit_command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    launcher = load_launcher(manifest)
    with launcher.manifest_lock(args.manifest):
        manifest = json.loads(args.manifest.read_text())
        install_audit_override(launcher, manifest)
        launcher.finalize(args.manifest, manifest)


if __name__ == "__main__":
    main()
