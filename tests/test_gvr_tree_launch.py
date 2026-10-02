"""The branched-GVR launcher: pilot selection, submission graph and approval gate."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def _launcher(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("submit_gvr_tree_collection")


def _artifact_root(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    (root / "prepared" / "benchmarks").mkdir(parents=True)
    items = []
    rates = (0.0, 0.25, 0.75, 1.0)
    for benchmark, count in (("arxivmath_train", 40), ("arxivmath_eval", 20)):
        rows = []
        for index in range(count):
            problem_id = f"arxivmath-{benchmark}-{index}"
            rows.append({"item_id": problem_id, "qwen36_pass_rate": rates[index % 4]})
            items.append(
                {"benchmark": benchmark, "problem_id": problem_id, "run_id": f"run-{problem_id}"}
            )
        (root / "prepared" / "benchmarks" / f"{benchmark}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    (root / "schedule.json").write_text(json.dumps({"items": items}))
    return root


def _manifest(tmp_path: Path, **values: Any) -> dict[str, Any]:
    run_root = tmp_path / "run"
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    return {
        "run_root": str(run_root),
        "source": str(tmp_path / "source"),
        "config": str(tmp_path / "source" / "experiment.source.yaml"),
        "artifact_root": str(tmp_path / "artifacts"),
        "exports": {"VALUE_AS_TOOL_MAX_CONCURRENCY": "2"},
        "jobs": {},
        **values,
    }


def test_pilot_selection_is_deterministic_and_balanced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher(monkeypatch)
    root = _artifact_root(tmp_path)

    pilot = launcher.select_pilot(root)

    assert pilot == launcher.select_pilot(root)
    assert len(pilot) == len(set(pilot)) == 30
    train = [run_id for run_id in pilot if "train" in run_id]
    assert len(train) == 20
    buckets = [int(run_id.rsplit("-", 1)[-1]) % 4 for run_id in train]
    assert sorted(buckets.count(bucket) for bucket in range(4)) == [5, 5, 5, 5]


def test_pilot_submission_covers_every_tree_once_and_judges_after_all_solves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher(monkeypatch)
    run_ids = [f"run-{index}" for index in range(30)]
    manifest = _manifest(tmp_path, pilot_run_ids=run_ids, pilot_gpus=8)
    submitted: dict[str, list[str]] = {}

    def fake_submit(path: Path, manifest: dict[str, Any], key: str, command: list[str], **_: Any):
        submitted[key] = command
        return str(1000 + len(submitted))

    monkeypatch.setattr(launcher, "submit_job", fake_submit)
    monkeypatch.setattr(launcher, "write_json", lambda path, value: None)
    launcher.submit_pilot(tmp_path / "run" / "submission.json", manifest)

    solves = {key: command for key, command in submitted.items() if key.startswith("pilot-solve")}
    assert len(solves) == 8
    covered = [
        command[index + 1]
        for command in solves.values()
        for index, value in enumerate(command)
        if value == "--run-id"
    ]
    assert sorted(covered) == sorted(run_ids)
    judge = submitted["pilot-judge-nodes"]
    assert "judge-nodes" in judge and judge.count("--run-id") == 30
    dependency = next(value for value in judge if value.startswith("--dependency="))
    assert dependency == "--dependency=afterany:" + ":".join(
        str(1000 + index) for index in range(1, 9)
    )
    for command in submitted.values():
        assert "--partition=g3" in command and "--account=scientific-reasoning" in command
        assert "--qos=g3_scientific-reasoning_high" in command
    assert manifest["state"] == "pilot_submitted"


def test_full_run_requires_a_human_approval_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher(monkeypatch)
    manifest = _manifest(tmp_path, state="pilot_submitted")
    path = tmp_path / "run" / "submission.json"
    path.write_text(json.dumps(manifest))

    def no_submission(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("nothing may be submitted without approval")

    monkeypatch.setattr(launcher, "submit_job", no_submission)
    with pytest.raises(SystemExit, match="approve-full.json"):
        launcher.full(SimpleNamespace(manifest=path, lanes=None))


def test_harness_profiles_cover_every_data_collection_harness_within_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from value_as_tool.harnesses import DATA_COLLECTION_HARNESSES

    launcher = _launcher(monkeypatch)
    assert set(launcher.HARNESS_PROFILES) == set(DATA_COLLECTION_HARNESSES)
    prefixes = [prefix for _, prefix in launcher.HARNESS_PROFILES.values()]
    assert len(set(prefixes)) == len(prefixes)
    for entrypoint, (attribute, _) in launcher.HARNESS_PROFILES.items():
        module = importlib.import_module(entrypoint.partition(":")[0])
        assert 0 < getattr(module, attribute) <= 8_388_608


def test_job_names_carry_the_launch_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _launcher(monkeypatch)
    submitted: dict[str, list[str]] = {}

    def fake_submit(path: Path, manifest: dict[str, Any], key: str, command: list[str], **_: Any):
        submitted[key] = command
        return str(2000 + len(submitted))

    monkeypatch.setattr(launcher, "submit_job", fake_submit)
    monkeypatch.setattr(launcher, "write_json", lambda path, value: None)
    manifest = _manifest(
        tmp_path, pilot_run_ids=["run-a", "run-b"], pilot_gpus=2, job_prefix="gvrj"
    )
    launcher.submit_pilot(tmp_path / "run" / "submission.json", manifest)
    names = {
        next(v for v in command if v.startswith("--job-name=")) for command in submitted.values()
    }
    assert names == {
        "--job-name=gvrj-pilot-solve-0",
        "--job-name=gvrj-pilot-solve-1",
        "--job-name=gvrj-pilot-judge-nodes",
    }


def test_pilots_of_two_launches_are_matched_by_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher(monkeypatch)
    first = _artifact_root(tmp_path / "first")
    second = _artifact_root(tmp_path / "second")
    schedule = json.loads((second / "schedule.json").read_text())
    for item in schedule["items"]:
        item["run_id"] = item["run_id"].replace("run-", "other-")
    (second / "schedule.json").write_text(json.dumps(schedule))

    problems = launcher._pilot_problems(first, launcher.select_pilot(first))
    assert problems == launcher._pilot_problems(second, launcher.select_pilot(second))
    assert len(problems) == 30 and problems == sorted(problems)
