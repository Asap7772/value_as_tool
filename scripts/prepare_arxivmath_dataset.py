"""Build the ArXivMath train/eval splits for branched GVR collection.

``download`` runs where huggingface.co is reachable (a cpu_x86 node: the login
sandbox blocks it) and needs only huggingface_hub. ``build`` runs anywhere with
pyarrow and writes the prepared JSONL files an experiment pins by sha256.

Source rows are Qwen3.6-35B attempts. Problems are deduplicated by
``problem_idx``, problems whose normalized text is identical are merged so they
cannot straddle the split, and the eval split reuses the hold-out rule of
meta-rubrics-midtraining (scripts/build_hard_screen_index.py) so both projects
hold out the same problems.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPO_ID = "MathArena/arxivmath-training_outputs"
REVISION = "02002a6d4e39033de27adeb4e4683deeb6f22850"
EXPECTED_ROWS = 10_420
DEFAULT_SOURCE = REPO / "artifacts/assets/datasets" / REPO_ID.replace("/", "--") / REVISION
DEFAULT_OUTPUT = REPO / "artifacts/assets/benchmarks/arxivmath" / REVISION
COLUMNS = ["problem_idx", "problem", "gold_answer", "correct", "output_tokens", "source"]
SPLIT_RULE = (
    'sha256("arxivmath\\n" + normalized(problem)); eval when int(hex[-8:], 16) % 10 == 0, '
    "normalized = NFKC, casefold, collapse whitespace "
    "(meta-rubrics-midtraining/scripts/build_hard_screen_index.py)"
)


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def split_group(problem: str) -> str:
    return hashlib.sha256(f"arxivmath\n{normalized(problem)}".encode()).hexdigest()


def split_of(problem: str) -> str:
    return "eval" if int(split_group(problem)[-8:], 16) % 10 == 0 else "train"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def env_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    env_file = REPO / ".env"
    if token or not env_file.exists():
        return token
    for line in env_file.read_text().splitlines():
        match = re.match(r"^\s*(?:export\s+)?HF_TOKEN\s*=\s*(.*)$", line)
        if match:
            return match[1].strip().strip("'\"")
    return None


def download(args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download

    destination = args.source.resolve()
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=destination,
        allow_patterns=["README.md", "data/*.parquet"],
        token=env_token(),
    )
    files = sorted(destination.glob("data/*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {destination}")
    manifest = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "files": {
            str(path.relative_to(destination)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        },
    }
    (destination / "download-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


def build(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    source = args.source.resolve()
    manifest = json.loads((source / "download-manifest.json").read_text())
    if manifest["repo_id"] != REPO_ID or manifest["revision"] != REVISION:
        raise SystemExit("download manifest does not match the pinned dataset revision")
    rows: list[dict] = []
    for name, entry in sorted(manifest["files"].items()):
        path = source / name
        if sha256_file(path) != entry["sha256"]:
            raise SystemExit(f"hash mismatch: {path}")
        rows.extend(pq.read_table(path, columns=COLUMNS).to_pylist())
    if len(rows) != EXPECTED_ROWS:
        raise SystemExit(f"expected {EXPECTED_ROWS} attempt rows, found {len(rows)}")

    problems: dict[str, dict] = {}
    for row in rows:
        entry = problems.setdefault(
            row["problem_idx"],
            {
                "problem_idx": row["problem_idx"],
                "problem": row["problem"],
                "gold_answer": row["gold_answer"],
                "arxiv_id": row["source"],
                "attempts": 0,
                "correct": 0,
                "output_tokens": 0,
            },
        )
        for field in ("problem", "gold_answer"):
            if row[field] != entry[field]:
                raise SystemExit(f"problem {row['problem_idx']} has inconsistent {field}")
        entry["attempts"] += 1
        entry["correct"] += bool(row["correct"])
        entry["output_tokens"] += int(row["output_tokens"] or 0)

    by_text: dict[str, list[dict]] = defaultdict(list)
    for entry in problems.values():
        by_text[normalized(entry["problem"])].append(entry)
    kept, merged, dropped = [], [], []
    for group in by_text.values():
        group.sort(key=lambda entry: (len(entry["problem_idx"]), entry["problem_idx"]))
        if len({normalized(entry["gold_answer"]) for entry in group}) > 1:
            dropped.append([entry["problem_idx"] for entry in group])
            continue
        head = dict(group[0])
        for other in group[1:]:
            for field in ("attempts", "correct", "output_tokens"):
                head[field] += other[field]
        if len(group) > 1:
            merged.append([entry["problem_idx"] for entry in group])
        kept.append(head)

    splits: dict[str, list[dict]] = {"train": [], "eval": []}
    for entry in kept:
        split = split_of(entry["problem"])
        splits[split].append(
            {
                "item_id": f"arxivmath-{entry['problem_idx']}",
                "problem_idx": entry["problem_idx"],
                "problem": entry["problem"],
                "gold_answer": entry["gold_answer"],
                "arxiv_id": entry["arxiv_id"],
                "split": split,
                "split_group": split_group(entry["problem"]),
                "qwen36_attempts": entry["attempts"],
                "qwen36_correct": entry["correct"],
                "qwen36_pass_rate": round(entry["correct"] / entry["attempts"], 6),
                "qwen36_mean_output_tokens": round(entry["output_tokens"] / entry["attempts"]),
            }
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "license": "cc-by-4.0",
        "attempt_rows": len(rows),
        "problems": len(problems),
        "merged_duplicate_texts": merged,
        "dropped_conflicting_answers": dropped,
        "split_rule": SPLIT_RULE,
        "splits": {},
    }
    for split, items in splits.items():
        items.sort(key=lambda item: (len(item["problem_idx"]), item["problem_idx"]))
        path = output / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in items),
            encoding="utf-8",
        )
        summary["splits"][split] = {
            "rows": len(items),
            "path": str(path.relative_to(args.asset_root.resolve())),
            "sha256": sha256_file(path),
            "qwen36_pass_rate_mean": round(
                sum(item["qwen36_pass_rate"] for item in items) / max(1, len(items)), 4
            ),
        }
    if args.upstream is not None:
        upstream = {
            normalized(question)
            for question in pq.read_table(args.upstream, columns=["question"])[
                "question"
            ].to_pylist()
        }
        summary["upstream_question_matches"] = sum(
            normalized(entry["problem"]) in upstream for entry in kept
        )
    (output / "manifest.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="fetch the pinned parquet shards")
    fetch.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    make = commands.add_parser("build", help="write train.jsonl, eval.jsonl and manifest.json")
    make.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    make.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    make.add_argument("--asset-root", type=Path, default=REPO / "artifacts/assets")
    make.add_argument(
        "--upstream",
        type=Path,
        default=None,
        help="MathArena/arxivmath-training parquet; counts questions shared with the upstream set",
    )
    args = parser.parse_args()
    {"download": download, "build": build}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
