"""The ArXivMath split builder: dedup, duplicate texts and the shared hold-out rule."""

from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import unicodedata
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("prepare_arxivmath_dataset")


def _meta_rubrics_split(problem: str) -> str:
    """Copied from meta-rubrics-midtraining/scripts/build_hard_screen_index.py."""

    text = " ".join(unicodedata.normalize("NFKC", problem).casefold().split())
    group = "group:" + hashlib.sha256(f"arxivmath\n{text}".encode()).hexdigest()
    return "validation" if int(group[-8:], 16) % 10 == 0 else "train"


def _problem_in(split: str, start: int = 0) -> str:
    return next(
        text
        for text in (f"Compute sentinel quantity {index}." for index in itertools.count(start))
        if _meta_rubrics_split(text) == split
    )


def _attempt(idx: str, problem: str, gold: str, correct: bool, tokens: int = 10) -> dict[str, Any]:
    return {
        "problem_idx": idx,
        "problem": problem,
        "gold_answer": gold,
        "correct": correct,
        "output_tokens": tokens,
        "source": f"2501.{idx.zfill(5)}",
        "answer": f"MODEL OUTPUT for {idx}",
    }


def _write_source(prep: ModuleType, source: Path, rows: list[dict[str, Any]]) -> None:
    (source / "data").mkdir(parents=True)
    halves = (rows[: len(rows) // 2], rows[len(rows) // 2 :])
    files = {}
    for index, half in enumerate(halves):
        path = source / "data" / f"train-0000{index}-of-00002.parquet"
        pq.write_table(pa.Table.from_pylist(half), path)
        files[str(path.relative_to(source))] = {"sha256": prep.sha256_file(path)}
    manifest = {"repo_id": prep.REPO_ID, "revision": prep.REVISION, "files": files}
    (source / "download-manifest.json").write_text(json.dumps(manifest))


def test_build_dedups_merges_and_uses_the_shared_hold_out_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prep = _module(monkeypatch)
    held_out = _problem_in("validation")
    kept = _problem_in("train")
    duplicate = _problem_in("train", start=1_000)
    conflict = _problem_in("train", start=2_000)
    rows = [
        _attempt("1", held_out, "3/2", True, 30),
        _attempt("1", held_out, "3/2", False, 10),
        _attempt("2", kept, "7", False),
        # The same problem under two indices, differing only in case and spacing.
        _attempt("3", duplicate, "x", True),
        _attempt("40", "  " + duplicate.upper().replace(" ", "   "), "x", False),
        # Identical text with different gold answers cannot be kept.
        _attempt("5", conflict, "1", True),
        _attempt("6", conflict, "2", True),
    ]
    source, assets = tmp_path / "source", tmp_path / "assets"
    _write_source(prep, source, rows)
    monkeypatch.setattr(prep, "EXPECTED_ROWS", len(rows))
    output = assets / "benchmarks" / "arxivmath" / prep.REVISION
    prep.build(SimpleNamespace(source=source, output=output, asset_root=assets, upstream=None))

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["problems"] == 6
    assert manifest["merged_duplicate_texts"] == [["3", "40"]]
    assert manifest["dropped_conflicting_answers"] == [["5", "6"]]
    prepared = {
        split: [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
        for split in ("train", "eval")
    }
    assert [row["item_id"] for row in prepared["eval"]] == ["arxivmath-1"]
    assert [row["item_id"] for row in prepared["train"]] == ["arxivmath-2", "arxivmath-3"]
    for split, items in prepared.items():
        expected = "validation" if split == "eval" else "train"
        assert all(_meta_rubrics_split(row["problem"]) == expected for row in items)
        entry = manifest["splits"][split]
        assert entry["sha256"] == prep.sha256_file(output / f"{split}.jsonl")
        assert entry["path"] == f"benchmarks/arxivmath/{prep.REVISION}/{split}.jsonl"
    first = prepared["eval"][0]
    assert (first["qwen36_attempts"], first["qwen36_correct"]) == (2, 1)
    assert first["qwen36_mean_output_tokens"] == 20
    assert prepared["train"][1]["qwen36_attempts"] == 2
    assert "MODEL OUTPUT" not in json.dumps(prepared)


def test_build_rejects_tampered_shards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prep = _module(monkeypatch)
    source = tmp_path / "source"
    _write_source(prep, source, [_attempt("1", "Compute 1.", "1", True)] * 2)
    next((source / "data").glob("*.parquet")).write_bytes(b"not parquet")
    with pytest.raises(SystemExit, match="hash mismatch"):
        prep.build(
            SimpleNamespace(
                source=source, output=tmp_path / "out", asset_root=tmp_path, upstream=None
            )
        )
