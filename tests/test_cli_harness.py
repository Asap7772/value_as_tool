from __future__ import annotations

import json
from pathlib import Path

import pytest

from value_as_tool import cli
from value_as_tool.harnesses import BUILTIN_HARNESSES


def test_harness_list_is_config_independent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_config = tmp_path / "does-not-exist.yaml"

    assert cli.main(["--config", str(missing_config), "harness", "list"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["ok"] is True
    assert output["count"] == 9
    assert [item["entrypoint"] for item in output["harnesses"]] == list(
        BUILTIN_HARNESSES
    )
    assert {item["harness_id"] for item in output["harnesses"]} >= {
        "direct",
        "cch_plan_work_review",
    }
    assert all(len(item["source_sha256"]) == 64 for item in output["harnesses"])


def test_harness_validate_accepts_explicit_entrypoints_without_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    entrypoint = "value_as_tool.harnesses.cch_plan_work_review:AgentHarness"

    assert (
        cli.main(
            [
                "--config",
                str(tmp_path / "does-not-exist.yaml"),
                "harness",
                "validate",
                entrypoint,
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)

    assert output["ok"] is True
    assert output["count"] == 1
    assert output["harnesses"][0]["harness_id"] == "cch_plan_work_review"


def test_harness_validate_reports_import_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        cli.main(
            [
                "--config",
                str(tmp_path / "does-not-exist.yaml"),
                "harness",
                "validate",
                "value_as_tool.harnesses.direct:MissingHarness",
            ]
        )
        == 1
    )
    output = json.loads(capsys.readouterr().out)

    assert output["ok"] is False
    assert output["count"] == 0
    assert output["errors"][0]["entrypoint"].endswith(":MissingHarness")
