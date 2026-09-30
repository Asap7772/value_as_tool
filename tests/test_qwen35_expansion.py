from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def expansion_module() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "expand_qwen35_large_budget.py"
    spec = importlib.util.spec_from_file_location("expand_qwen35_large_budget", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def manifest() -> dict[str, Any]:
    jobs = {
        f"pilot-{stage}-{shard}": {"job_id": str(1_614_949 + offset), "array": str(shard)}
        for offset, (shard, stage) in enumerate(
            (shard, stage) for shard in (0, 272, 280) for stage in ("solve", "judge")
        )
    }
    jobs["advance"] = {"job_id": "1614955"}
    return {"state": "pilot_submitted", "jobs": jobs}


def _array_indices_and_cap(array: str) -> tuple[set[int], int]:
    interval, cap = array.split("%")
    start, end = (int(value) for value in interval.split("-"))
    return set(range(start, end + 1)), int(cap)


@pytest.mark.parametrize("high_cap, shared_cap", [(212, 64), (2, 3), (256, 128)])
def test_plan_covers_every_shard_once_and_reserves_pilot_capacity(
    expansion_module: ModuleType,
    manifest: dict[str, Any],
    high_cap: int,
    shared_cap: int,
) -> None:
    original = copy.deepcopy(manifest)
    plan = expansion_module.plan_expansion(manifest, high_cap, shared_cap)

    assert manifest == original
    for stage, reservations in (
        ("solve", {"high": 1, "shared": 2}),
        ("judge", {"high": 0, "shared": 0}),
    ):
        high, high_throttle = _array_indices_and_cap(plan[f"{stage}_arrays"]["high"])
        shared, shared_throttle = _array_indices_and_cap(plan[f"{stage}_arrays"]["shared"])
        assert high.isdisjoint(shared)
        assert high | shared == set(range(384))
        assert high == set(range(256))
        assert high_throttle + reservations["high"] == high_cap
        assert shared_throttle + reservations["shared"] == shared_cap
        assert high_throttle > 0 and shared_throttle > 0
    assert plan["max_concurrent_gpus"] == {
        "high": high_cap,
        "shared": shared_cap,
        "combined": high_cap + shared_cap,
    }


def test_plan_preserves_pilot_shards_with_dependencies_on_their_exact_jobs(
    expansion_module: ModuleType,
    manifest: dict[str, Any],
) -> None:
    plan = expansion_module.plan_expansion(manifest, 212, 64)

    assert plan["solve_arrays"] == {"high": "0-255%211", "shared": "256-383%62"}
    assert plan["judge_arrays"] == {"high": "0-255%212", "shared": "256-383%64"}
    assert plan["pilot_dependencies"] == {
        "high": {"0": "1614949"},
        "shared": {"272": "1614951", "280": "1614953"},
    }
    assert plan["judge_pilot_dependencies"] == ["1614950", "1614952", "1614954"]


@pytest.mark.parametrize(
    "high_cap, shared_cap", [(0, 64), (1, 64), (212, 0), (212, 1), (212, 2), (257, 64), (212, 129)]
)
def test_plan_rejects_caps_that_cannot_reserve_pilots_or_exceed_pool_size(
    expansion_module: ModuleType,
    manifest: dict[str, Any],
    high_cap: int,
    shared_cap: int,
) -> None:
    with pytest.raises(ValueError):
        expansion_module.plan_expansion(manifest, high_cap, shared_cap)


@pytest.mark.parametrize("stage", ["solve", "judge"])
@pytest.mark.parametrize("shard", [0, 272, 280])
@pytest.mark.parametrize("invalid", [{"job_id": ""}, {"array": "1"}])
def test_plan_requires_confirmed_pilots_for_each_expected_shard(
    expansion_module: ModuleType,
    manifest: dict[str, Any],
    stage: str,
    shard: int,
    invalid: dict[str, str],
) -> None:
    manifest["jobs"][f"pilot-{stage}-{shard}"].update(invalid)

    with pytest.raises(ValueError, match="confirmed IDs and shards"):
        expansion_module.plan_expansion(manifest, 212, 64)


class RecordingLauncher:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []
        self.dependencies: dict[str, tuple[str, ...]] = {}
        self.job_ids = {
            "solve-high": "2000",
            "solve-shared": "2001",
            "judge-high": "2002",
            "judge-shared": "2003",
            "finalize": "2004",
        }

    def gpu_command(
        self,
        manifest: dict[str, Any],
        key: str,
        stage: str,
        pool: str,
        array: str,
        dependencies: tuple[str, ...] = (),
    ) -> list[str]:
        self.dependencies[key] = dependencies
        return ["sbatch", f"--job-name={key}", f"--array={array}"]

    def controller_command(
        self,
        manifest: dict[str, Any],
        stage: str,
        dependencies: tuple[str, ...],
    ) -> list[str]:
        self.dependencies[stage] = dependencies
        return ["sbatch", f"--job-name={stage}"]

    def submit_job(
        self,
        path: Path,
        manifest: dict[str, Any],
        key: str,
        command: list[str],
        **metadata: Any,
    ) -> str:
        if key not in manifest["jobs"]:
            self.events.append(("submit", (key, command)))
            manifest["jobs"][key] = {"job_id": self.job_ids[key], **metadata}
        return manifest["jobs"][key]["job_id"]

    def record_command(
        self,
        path: Path,
        manifest: dict[str, Any],
        label: str,
        command: list[str],
    ) -> SimpleNamespace:
        self.events.append(("command", command))
        return SimpleNamespace(returncode=0, stderr="")

    def write_json(self, path: Path, manifest: dict[str, Any]) -> None:
        self.events.append(("persist", copy.deepcopy(manifest)))

    def now(self) -> str:
        return "2026-09-28T12:00:00+00:00"


@pytest.fixture
def launcher(
    expansion_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> RecordingLauncher:
    result = RecordingLauncher()
    expected_pilots = {"2000_0": "1614949", "2001_272": "1614951", "2001_280": "1614953"}

    def show_held_job(command: list[str], **kwargs: Any) -> SimpleNamespace:
        assert command[:3] == ["scontrol", "show", "job"]
        job_id = command[3]
        result.events.append(("verify", job_id))
        return SimpleNamespace(
            stdout=(
                f"JobId={job_id} JobState=PENDING Priority=0 "
                f"Dependency=afterany:{expected_pilots[job_id]} "
            )
        )

    monkeypatch.setattr(expansion_module.subprocess, "run", show_held_job)
    return result


def test_expansion_builds_complete_barriers_before_releasing_solve_arrays(
    expansion_module: ModuleType,
    manifest: dict[str, Any],
    launcher: RecordingLauncher,
    tmp_path: Path,
) -> None:
    plan = expansion_module.plan_expansion(manifest, 212, 64)
    expansion_module.expand(tmp_path / "manifest.json", manifest, launcher, plan)

    events = [event for event in launcher.events if event[0] != "persist"]
    assert events[0] == ("command", ["scontrol", "hold", "1614955"])
    submissions = dict(payload for kind, payload in events if kind == "submit")
    assert set(submissions) == set(launcher.job_ids)
    assert "--hold" in submissions["solve-high"]
    assert "--hold" in submissions["solve-shared"]
    for element, pilot in (("2000_0", "1614949"), ("2001_272", "1614951"), ("2001_280", "1614953")):
        dependency = (
            "command",
            ["scontrol", "update", f"JobId={element}", f"Dependency=afterany:{pilot}"],
        )
        assert events.index(dependency) < events.index(("verify", element))
    assert launcher.dependencies["judge-high"] == (
        "2000",
        "2001",
        "1614950",
        "1614952",
        "1614954",
    )
    assert launcher.dependencies["judge-shared"] == launcher.dependencies["judge-high"]
    assert launcher.dependencies["finalize"] == ("2002", "2003")
    assert events[-3:] == [
        ("command", ["scontrol", "release", "2000"]),
        ("command", ["scontrol", "release", "2001"]),
        ("command", ["scontrol", "release", "1614955"]),
    ]
    release_index = next(
        index for index, event in enumerate(launcher.events) if event == events[-3]
    )
    assert launcher.events[release_index - 1][0] == "persist"
    assert launcher.events[release_index - 1][1]["state"] == "full_submitted"
    assert manifest["immediate_expansion"]["complete"] is True

    previous_events = copy.deepcopy(launcher.events)
    expansion_module.expand(tmp_path / "manifest.json", manifest, launcher, plan)
    assert launcher.events == previous_events


@pytest.mark.parametrize(
    "observed",
    [
        "JobState=PENDING Priority=0 Dependency=(null) ",
        "JobState=RUNNING Priority=0 Dependency=afterany:1614949 ",
        "JobState=PENDING Priority=100 Dependency=afterany:1614949 ",
    ],
)
def test_expansion_keeps_arrays_held_when_pilot_dependency_is_not_verified(
    expansion_module: ModuleType,
    manifest: dict[str, Any],
    launcher: RecordingLauncher,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observed: str,
) -> None:
    monkeypatch.setattr(
        expansion_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=observed),
    )
    plan = expansion_module.plan_expansion(manifest, 212, 64)

    with pytest.raises(RuntimeError, match="held dependency was not verified"):
        expansion_module.expand(tmp_path / "manifest.json", manifest, launcher, plan)

    commands = [payload for kind, payload in launcher.events if kind == "command"]
    assert not any(command[:2] == ["scontrol", "release"] for command in commands)
    assert manifest["state"] == "expanding"
    assert not manifest["immediate_expansion"].get("complete")
