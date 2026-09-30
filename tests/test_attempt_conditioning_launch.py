from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return SimpleNamespace(
        resources=importlib.import_module("attempt_conditioning_resources"),
        queue=importlib.import_module("attempt_conditioning_queue"),
        launch=importlib.import_module("submit_attempt_conditioning"),
    )


def slurm_job(job_id, *, qos="g3_scientific-reasoning_high", gpus=1, array="", task=None):
    return {
        "job_id": job_id,
        "array_job_id": {"set": True, "number": job_id if array or task is not None else 0},
        "array_task_id": {"set": task is not None, "number": task or 0},
        "array_task_string": array,
        "job_state": ["PENDING"],
        "qos": qos,
        "tres_req_str": f"cpu=24,mem=96G,node=1,gres/gpu={gpus},gres/gpu:a100={gpus}",
    }


def test_pending_arrays_and_multigpu_are_reserved_without_parent_duplication(modules):
    r = modules.resources
    raw = [
        slurm_job(100, gpus=2, array="0-99%8"),
        slurm_job(100, gpus=2, task=5),
        slurm_job(200, qos=r.QOS["shared"], gpus=4, array="0-7"),
        slurm_job(300, qos="unrelated_qos", gpus=8),
    ]
    rows = r.expand_jobs(raw)
    assert len(rows) == 109
    plan = r.admission_plan(rows, {}, 276)
    assert plan["other_requested"] == {"high": 200, "shared": 32}
    assert plan["add_workers"] == {"high": 12, "shared": 32}


def test_own_unobserved_submissions_still_reserve_capacity_and_pilot_uses_both_pools(modules):
    r = modules.resources
    own = {"batch": {"job_id": "123", "pool": "high", "workers": 200}}
    plan = r.admission_plan([], own, 276)
    assert plan["add_workers"] == {"high": 12, "shared": 64}
    pilot = r.admission_plan([], {}, 24)["add_workers"]
    assert sum(pilot.values()) == 24 and all(pilot.values())
    own["batch"]["terminal_confirmed"] = True
    assert r.admission_plan([], own, 276)["add_workers"] == r.LIMITS


def test_resource_accounting_fails_closed_on_ambiguous_requests(modules):
    r = modules.resources
    job = slurm_job(1)
    job["tres_req_str"] = "cpu=24,gres/gpu:a100=2,gres/gpu=1"
    with pytest.raises(ValueError, match="exceed"):
        r.expand_jobs([job])
    job.pop("tres_req_str")
    with pytest.raises(ValueError, match="omitted"):
        r.expand_jobs([job])


def task(task_id, group="a", dependencies=()):
    return {
        "task_id": task_id,
        "benchmark": "proofbench",
        "method": group,
        "dependencies": list(dependencies),
    }


def test_balanced_claims_dependencies_and_crash_recovery(modules, tmp_path):
    q = modules.queue.TaskQueue(tmp_path / "work.sqlite3")
    tasks = [task("a1"), task("a2"), task("b1", "b"), task("reduce", "c", ["a1", "a2"])]
    q.initialize(tasks, {"stage": "test"})
    a = q.claim("worker1")
    b = q.claim("worker2")
    assert (a.task_id, b.task_id) == ("a1", "b1")
    # No timeout can reclaim a held model-task lock.
    assert q.recover_abandoned() == []
    a.close()  # Simulates process death, without completing its durable queue row.
    assert q.recover_abandoned() == ["a1"]
    resumed = q.claim("worker3")
    assert resumed.task_id == "a1"
    q.finish(resumed, {"outcome": "completed"})
    q.finish(b, {"outcome": "completed"})
    a2 = q.claim("worker3")
    assert a2.task_id == "a2"
    assert q.claim("worker4") is None  # Reducer cannot bypass its outstanding dependency.
    q.finish(a2, {"outcome": "completed"})
    reducer = q.claim("worker4")
    assert reducer.task_id == "reduce"
    q.finish(reducer, {"outcome": "completed"})
    assert q.counts() == {"pending": 0, "running": 0, "complete": 4, "failed": 0, "ready": 0}
    q.initialize(tasks, {"stage": "test"})
    with pytest.raises(ValueError, match="identity"):
        q.initialize(tasks, {"stage": "different"})


def test_failed_queue_does_not_dispatch_dependents_or_repeat_outcomes(modules, tmp_path):
    q = modules.queue.TaskQueue(tmp_path / "work.sqlite3")
    q.initialize([task("map"), task("reduce", dependencies=["map"])], {})
    q.finish(q.claim("worker"), {"error": "preserved failure"}, failed=True)
    assert q.claim("replacement") is None
    assert q.recover_abandoned() == []
    with pytest.raises(ValueError, match="cyclic"):
        modules.queue.TaskQueue(tmp_path / "cycle.sqlite3").initialize(
            [task("a", dependencies=["b"]), task("b", dependencies=["a"])], {}
        )


def test_worker_commands_only_submit_current_stage_without_gpu_dependencies(modules, tmp_path):
    launch = modules.launch
    manifest = {
        "source": str(tmp_path),
        "control_root": str(tmp_path),
        "exports": {},
        "stage": "preprocess",
        "preprocess_config": "pre.yaml",
        "manifest": "manifest.json",
        "queues": {"preprocess": {"path": "queue.sqlite3"}},
    }
    command = launch.worker_command(manifest, "preprocess", "high", 3, "test")
    assert "--array=0-2" in command and "--segment=1" in command
    assert not any("dependency" in arg for arg in command)
    with pytest.raises(ValueError, match="current stage"):
        launch.worker_command(manifest, "judge", "high", 3, "test")
    assert "--mem=32G" in launch.controller_command(manifest)


def test_pilot_rejects_protocol_native_limit_and_judge_parse_failures(modules):
    valid = {
        "status": "accepted",
        "usage_exact": True,
        "calls": [{"response": {"finish_reason": "stop"}}],
    }
    assert (
        modules.launch.pilot_result_issues(valid, {"judge_status": "completed", "score": 0}) == []
    )
    for status in ("protocol_error", "context_exhausted", "budget_exhausted", "failed"):
        assert modules.launch.pilot_result_issues({**valid, "status": status})
    assert modules.launch.pilot_result_issues(valid, {"judge_status": "parse_error"})
    assert modules.launch.pilot_result_issues({**valid, "usage_exact": False})


def test_controller_blocks_after_pilot_failure_without_submitting_full_run(
    modules, tmp_path, monkeypatch
):
    launch = modules.launch
    q = modules.queue.TaskQueue(tmp_path / "pilot.sqlite3")
    q.initialize([task("pilot")], {})
    q.finish(q.claim("worker"), {"outcome": "protocol_error"})
    manifest = {
        "stage": "pilot_solve",
        "state": "running",
        "queues": {"pilot_solve": {"path": str(q.path)}},
        "jobs": {"pilot": {"stage": "pilot_solve", "terminal_confirmed": True}},
        "control_root": str(tmp_path),
    }
    monkeypatch.setattr(launch, "verify_source", lambda _: None)
    monkeypatch.setattr(launch, "observe_owned_jobs", lambda _: {})
    monkeypatch.setattr(
        launch, "pilot_gate", lambda *a, **k: {"ready": False, "issues": ["protocol"]}
    )
    monkeypatch.setattr(launch, "submit_job", lambda *a, **k: pytest.fail("must not submit"))
    assert launch.tick(tmp_path / "manifest.json", manifest) is False
    assert manifest["state"] == "blocked" and manifest["stage"] == "pilot_solve"


def test_finalizer_refuses_unfinished_queue_or_live_workers(modules, tmp_path):
    manifest = {"stage": "judge", "queues": {}, "jobs": {}}
    for stage in modules.launch.STAGES:
        q = modules.queue.TaskQueue(tmp_path / f"{stage}.sqlite3")
        q.initialize([task("pending")] if stage == "judge" else [], {})
        manifest["queues"][stage] = {"path": str(q.path)}
    with pytest.raises(RuntimeError, match="unfinished judge"):
        modules.launch.finalize(manifest)
    q.finish(q.claim("worker"), {"outcome": "completed"})
    manifest["jobs"]["live"] = {"pool": "high", "terminal_confirmed": False}
    with pytest.raises(RuntimeError, match="GPU workers"):
        modules.launch.finalize(manifest)


def test_ambiguous_sbatch_intent_is_not_retried(modules, tmp_path, monkeypatch):
    launch = modules.launch
    manifest = {"jobs": {}}
    calls = []

    def bad_submit(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="unparseable", stderr="")

    monkeypatch.setattr(launch.subprocess, "run", bad_submit)
    path = tmp_path / "manifest.json"
    with pytest.raises(RuntimeError, match="did not confirm"):
        launch.submit_job(
            path, manifest, "batch", ["sbatch"], stage="solve", pool="high", workers=1
        )
    assert json.loads(path.read_text())["jobs"]["batch"]["intent_at"]
    with pytest.raises(RuntimeError, match="ambiguous"):
        launch.submit_job(
            path, manifest, "batch", ["sbatch"], stage="solve", pool="high", workers=1
        )
    assert len(calls) == 1


def test_freeze_precedes_schedule_and_no_next_stage_gpu_is_presubmitted(
    modules, tmp_path, monkeypatch
):
    launch = modules.launch
    q = modules.queue.TaskQueue(tmp_path / "preprocess.sqlite3")
    q.initialize([], {"stage": "preprocess"})
    manifest = {
        "stage": "preprocess",
        "state": "running",
        "jobs": {},
        "queues": {"preprocess": {"path": str(q.path)}},
    }
    events = []
    monkeypatch.setattr(launch, "verify_source", lambda _: None)
    monkeypatch.setattr(launch, "observe_owned_jobs", lambda _: {})
    monkeypatch.setattr(
        launch, "freeze_and_prepare", lambda _: events.append("freeze_then_prepare")
    )

    def initialize(current, stage):
        events.append(stage)
        current["stage"] = stage

    monkeypatch.setattr(launch, "initialize_stage", initialize)
    monkeypatch.setattr(launch, "submit_job", lambda *a, **k: pytest.fail("future GPU submission"))
    assert launch.tick(tmp_path / "manifest.json", manifest)
    assert events == ["freeze_then_prepare", "pilot_solve"]


def test_dry_run_emits_reviewable_commands_without_sbatch(modules, tmp_path, monkeypatch, capsys):
    launch = modules.launch
    q = modules.queue.TaskQueue(tmp_path / "preprocess.sqlite3")
    q.initialize([task(f"map-{i}") for i in range(48)], {})
    manifest = {
        "source": str(tmp_path),
        "control_root": str(tmp_path),
        "exports": {},
        "stage": "preprocess",
        "state": "prepared",
        "preprocess_config": "pre.yaml",
        "manifest": str(tmp_path / "manifest.json"),
        "expected_runs": 39360,
        "jobs": {},
        "queues": {"preprocess": {"path": str(q.path)}},
    }
    path = Path(manifest["manifest"])
    path.write_text(json.dumps(manifest))
    original = path.read_bytes()
    monkeypatch.setattr(launch, "verify_source", lambda _: None)
    monkeypatch.setattr(launch, "query_user_jobs", lambda: [])
    monkeypatch.setattr(launch, "submit_job", lambda *a, **k: pytest.fail("dry run submitted"))
    monkeypatch.setattr("sys.argv", ["launcher", "launch", "--manifest", str(path), "--dry-run"])
    launch.main()
    preview = json.loads(capsys.readouterr().out)
    assert preview["dry_run"] and preview["expected_runs"] == 39360
    assert len(preview["current_stage_worker_commands"]) == 2
    assert sum(preview["admission"]["add_workers"].values()) == 24
    assert path.read_bytes() == original


def test_finalizer_reports_preserved_outcomes_and_requires_exact_coverage(
    modules, tmp_path, monkeypatch
):
    launch = modules.launch
    manifest = {
        "stage": "judge",
        "queues": {},
        "jobs": {},
        "config": "frozen.yaml",
        "control_root": str(tmp_path),
        "expected_runs": 39360,
        "config_fingerprint": "config",
        "schedule_fingerprint": "schedule",
        "conditioning_manifest_sha256": "conditioning",
        "source_sha256": "source",
    }
    for stage in launch.STAGES:
        q = modules.queue.TaskQueue(tmp_path / f"{stage}.sqlite3")
        q.initialize([], {})
        manifest["queues"][stage] = {"path": str(q.path)}
    import audit_budget_run

    from value_as_tool import pipeline

    monkeypatch.setattr(pipeline, "load_context", lambda _: SimpleNamespace(artifact_root=tmp_path))
    monkeypatch.setattr(pipeline, "report", lambda _: {"scheduled_cells": 39360})
    audit = {
        "ready": True,
        "selected": 39360,
        "solve_statuses": {"protocol_error": 1},
        "judge_statuses": {"solve_failed": 1},
    }
    monkeypatch.setattr(audit_budget_run, "audit", lambda *a, **k: audit)
    completion = launch.finalize(manifest)
    assert completion["execution_complete"]
    assert completion["model_statuses"] == {"protocol_error": 1}
    audit["ready"] = False
    with pytest.raises(RuntimeError, match="coverage/context audit"):
        launch.finalize(manifest)


@pytest.fixture
def launch_inputs(modules, tmp_path, monkeypatch):
    """A movable config and tiny local assets exercise real launch preparation."""
    launch = modules.launch
    actual_repo = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load(
        (actual_repo / "experiment_qwen35_9b_attempt_conditioning.yaml").read_text()
    )
    repo = tmp_path / "repo"
    for name in launch.COPY_ITEMS:
        destination = repo / name
        if name in {"src", "scripts", "prompts"}:
            destination.mkdir(parents=True)
            (destination / "fixture.txt").write_text(name)
        else:
            destination.write_text(name)
    monkeypatch.setattr(launch, "__file__", str(repo / "scripts/submit_attempt_conditioning.py"))
    monkeypatch.setattr(launch, "query_user_jobs", lambda: [])
    monkeypatch.setattr(
        launch.subprocess, "run", lambda *a, **k: pytest.fail("scheduler or model invocation")
    )
    monkeypatch.setenv("VALUE_AS_TOOL_ARTIFACT_ROOT", str(tmp_path / "wrong-output"))
    monkeypatch.setenv("VALUE_AS_TOOL_ASSET_ROOT", str(tmp_path / "wrong-assets"))
    config_path = repo / "configs/experiment.yaml"
    config_path.parent.mkdir()
    payload["paths"] = {"artifact_root": "../output", "asset_root": "../assets"}
    payload["conditioning"].update(
        source_artifact_root="../prior", bank_root="../output/conditioning"
    )
    bank_path = repo / "output/conditioning/bank.json"
    bank_path.parent.mkdir(parents=True)
    bank_path.write_text(json.dumps({
        "source_seeds": list(range(8)),
        "problems": {
            benchmark: {f"{benchmark}-problem": {
                "labels": [{"attempt_id": f"{benchmark}-seed-{seed}"} for seed in range(8)]
            }}
            for benchmark in ("imo_proof", "proofbench")
        },
    }))
    payload["conditioning"]["bank_sha256"] = launch.digest(bank_path)
    models = {"schema_version": 1, "models": {}}
    for role, selected in payload["models"].items():
        model_root = repo / "assets" / role
        model_root.mkdir(parents=True)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (model_root / name).write_text("{}")
        (model_root / "model.safetensors").write_bytes(b"fake-model-weights")
        models["models"][role] = {
            "name": selected["name"],
            "revision": selected["revision"],
            "path": str(model_root),
            "tokenizer_only": False,
        }
    source_manifest = repo / "prior/prepared/models/manifest.json"
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_text(json.dumps(models))
    config_path.write_text(yaml.safe_dump(payload))
    return SimpleNamespace(
        repo=repo,
        config=config_path,
        payload=payload,
        models=models,
        source_manifest=source_manifest,
        args=SimpleNamespace(
            config=config_path, control_root=tmp_path / "control", source=None, pilot_problem=[]
        ),
    )


def test_prepare_launch_pins_relative_paths_and_bootstraps_models(
    modules, launch_inputs, monkeypatch, capsys
):
    from value_as_tool.pipeline import load_context

    inputs = launch_inputs
    monkeypatch.setattr("sys.argv", [
        "launcher", "launch", "--config", str(inputs.config),
        "--control-root", str(inputs.args.control_root), "--dry-run",
    ])
    modules.launch.main()
    preview = json.loads(capsys.readouterr().out)
    manifest = json.loads(Path(preview["manifest"]).read_text())
    original = load_context(inputs.config, environment={})
    copied = load_context(manifest["preprocess_config"], environment={})
    assert copied.config.to_dict() == original.config.to_dict()
    assert copied.artifact_root == inputs.repo / "output"
    assert copied.asset_root == inputs.repo / "assets"
    assert manifest["source_artifact_root"] == str(inputs.repo / "prior")
    assert manifest["bank_root"] == str(inputs.repo / "output/conditioning")
    assert manifest["expected_runs"] == 2 * 24 * 8
    assert manifest["jobs"] == {} and manifest["state"] == "prepared"
    assert manifest["queues"]["preprocess"]["tasks"] == 2 * 2 * (8 + 1)
    assert manifest["exports"]["VALUE_AS_TOOL_ARTIFACT_ROOT"] == str(copied.artifact_root)
    bootstrapped = Path(manifest["preprocess_models"]["path"])
    assert json.loads(bootstrapped.read_text()) == inputs.models
    assert manifest["preprocess_models"]["sha256"] == modules.launch.digest(bootstrapped)
    assert not (copied.artifact_root / "schedule.json").exists()
    assert preview["dry_run"] and len(preview["current_stage_worker_commands"]) == 2
    modules.launch.verify_source(manifest)
    Path(manifest["preprocess_config"]).chmod(0o644)
    with Path(manifest["preprocess_config"]).open("a") as stream:
        stream.write("\n# changed after launch preparation\n")
    with pytest.raises(RuntimeError, match="pinned preprocess_config changed"):
        modules.launch.verify_source(manifest)


def test_prepare_launch_rejects_an_unrelated_arm_before_writing_control(modules, launch_inputs):
    inputs = launch_inputs
    inputs.payload["evaluation"]["harnesses"][-1] = (
        "value_as_tool.harnesses.direct:AgentHarness"
    )
    inputs.config.write_text(yaml.safe_dump(inputs.payload))
    with pytest.raises(ValueError, match="24-arm"):
        modules.launch.prepare_launch(inputs.args)
    assert not inputs.args.control_root.exists()


@pytest.mark.parametrize("fault", ["revision", "tokenizer_only", "weights", "existing_manifest"])
def test_model_bootstrap_rejects_incompatible_or_incomplete_assets(
    modules, launch_inputs, fault
):
    from value_as_tool.pipeline import load_context

    inputs = launch_inputs
    solver = inputs.models["models"]["solver"]
    if fault == "revision":
        solver["revision"] = "different-revision"
    elif fault == "tokenizer_only":
        solver["tokenizer_only"] = True
    elif fault == "weights":
        (Path(solver["path"]) / "model.safetensors").unlink()
    else:
        existing = inputs.repo / "output/prepared/models/manifest.json"
        existing.parent.mkdir(parents=True)
        existing.write_text(json.dumps({"models": {}}))
    inputs.source_manifest.write_text(json.dumps(inputs.models))
    context = load_context(inputs.config, environment={})
    expected_error = FileNotFoundError if fault == "weights" else ValueError
    with pytest.raises(expected_error):
        modules.launch.bootstrap_preprocess_models(context)


def test_frozen_config_preserves_copied_runtime_paths_and_schedule_identity(
    modules, launch_inputs, monkeypatch
):
    from value_as_tool import conditioning
    from value_as_tool.pipeline import load_context

    launch = modules.launch
    manifest_path = launch.prepare_launch(launch_inputs.args)
    manifest = json.loads(manifest_path.read_text())
    frozen_digest = "f" * 64
    monkeypatch.setattr(
        conditioning, "freeze_bank", lambda *a, **k: {"manifest_sha256": frozen_digest}
    )
    preparer = importlib.import_module("prepare_qwen35_large_budget")
    bank = json.loads((Path(manifest["bank_root"]) / "bank.json").read_text())
    schedule_items = [
        {
            "run_id": f"{benchmark}-{index}-{seed}",
            "benchmark": benchmark,
            "problem_id": next(iter(problems)),
            "seed": seed,
            "harness_id": arm.rsplit(":", 1)[-1],
        }
        for benchmark, problems in bank["problems"].items()
        for index, arm in enumerate(launch_inputs.payload["evaluation"]["harnesses"])
        for seed in range(8, 16)
    ]

    def prepare_local(config, source_root):
        context = load_context(config, environment={})
        assert context.artifact_root == launch_inputs.repo / "output"
        assert Path(context.config.conditioning.bank_root) == Path(manifest["bank_root"])
        assert source_root == launch_inputs.repo / "prior"
        assert context.config.conditioning.manifest_sha256 == frozen_digest
        schedule_path = context.artifact_root / "schedule.json"
        schedule_path.write_text(json.dumps({"items": schedule_items}))
        return {
            "scheduled": len(schedule_items),
            "schedule_path": str(schedule_path),
            "config_fingerprint": context.config_fingerprint,
            "schedule_fingerprint": "schedule-fingerprint",
        }

    monkeypatch.setattr(preparer, "prepare_local", prepare_local)
    launch.freeze_and_prepare(manifest)
    frozen_context = load_context(manifest["config"], environment={})
    assert manifest["config_fingerprint"] == frozen_context.config_fingerprint
    assert manifest["conditioning_manifest_sha256"] == frozen_digest
    assert len(manifest["pilot_run_ids"]) == 48
    assert manifest["config_sha256"] == launch.digest(Path(manifest["config"]))
    launch.verify_source(manifest)
