import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    path = Path(__file__).parents[1] / "scripts/finalize_qwen35_large_budget.py"
    spec = importlib.util.spec_from_file_location("corrected_finalizer", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_override_preserves_all_audit_requirements(tmp_path):
    script = tmp_path / "audit.py"
    script.write_text("# corrected audit\n")
    manifest = {"audit_override": {
        "path": str(script), "sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
    }}
    expected = [
        "python", "pinned/audit.py", "--config", "pinned/config.yaml",
        "--shard-count", "384", "--check-context", "--require-ready",
        "--output", "full-audit.json",
    ]
    launcher = SimpleNamespace(audit_command=lambda current, *, pilot: expected.copy())
    module().install_audit_override(launcher, manifest)
    assert launcher.audit_command(manifest, pilot=False) == [
        expected[0], str(script), *expected[2:],
    ]


def test_changed_audit_is_rejected(tmp_path):
    script = tmp_path / "audit.py"
    script.write_text("original")
    manifest = {"audit_override": {
        "path": str(script), "sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
    }}
    script.write_text("changed")
    with pytest.raises(RuntimeError, match="audit correction has changed"):
        module().verified_audit_path(manifest)


def test_pinned_source_is_verified_before_loading(tmp_path):
    source = tmp_path / "source"
    script = source / "scripts/submit_qwen35_large_budget.py"
    script.parent.mkdir(parents=True)
    script.write_text("marker = 42\n")
    hashes = {str(script.relative_to(source)): hashlib.sha256(script.read_bytes()).hexdigest()}
    (tmp_path / "snapshot-files.json").write_text(json.dumps(hashes))
    manifest = {"source": str(source), "run_root": str(tmp_path)}
    assert module().load_launcher(manifest).marker == 42
    script.write_text("marker = 43\n")
    with pytest.raises(RuntimeError, match="immutable source changed"):
        module().load_launcher(manifest)
