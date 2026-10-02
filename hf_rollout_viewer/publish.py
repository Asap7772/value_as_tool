"""Publish the rollout viewer: a public dataset repo with the data and a static Space that reads it.

The Space pins the dataset commit it was built against, so later data uploads cannot silently change
what an existing Space link shows. Before any upload, every file is scanned for credentials. The
scan covers token-shaped strings and the literal secret values of the local .env files; any hit
aborts the publish.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ENV_FILES = [REPO / ".env"] + [
    REPO.parent / name / ".env"
    for name in ("test_time_rewards_minimal", "meta-rubrics-midtraining", "test-time-rewards")
]
TOKEN_PATTERNS = re.compile(
    r"(?<![A-Za-z0-9_-])(?:hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{24,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"
    r"|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35})"
)
SECRET_NAME = re.compile(r"(TOKEN|KEY|SECRET|PASSWORD)", re.I)


def env_values(path):
    values = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if match:
            values[match[1]] = match[2].strip().strip("'\"")
    return values


def secret_values():
    secrets = set()
    for path in ENV_FILES:
        for name, value in env_values(path).items():
            if SECRET_NAME.search(name) and len(value) >= 12:
                secrets.add(value)
    return secrets


def scan_file(task):
    path, secrets = task
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    text = data.decode("utf-8", errors="ignore")
    found = [match.group(0)[:6] + "…" for match in TOKEN_PATTERNS.finditer(text)]
    found += ["<.env secret value>" for secret in secrets if secret in text]
    return path, found[:3]


def scan(root, secrets):
    paths = sorted(
        p
        for p in Path(root).rglob("*")
        if p.is_file() and ".cache" not in p.parts and p.suffix not in {".woff", ".woff2", ".ttf"}
    )
    with ProcessPoolExecutor(min(32, os.cpu_count() or 1)) as pool:
        results = pool.map(scan_file, [(path, secrets) for path in paths], chunksize=32)
        return [(path, found) for path, found in results if found]


def dataset_card(data, dataset, space):
    meta = json.loads((data / "index" / "experiments.json").read_text())
    rows = []
    for exp in meta["experiments"]:
        rows.append(
            f"| `{exp['id']}` | {exp['label']} | `{exp['model']}` | {len(exp['arms'])} | "
            f"{exp['seeds'][0]}–{exp['seeds'][-1]} |"
        )
    table = "\n".join(rows)
    return f"""---
license: other
viewer: false
pretty_name: Value-as-Tool proof rollouts
tags:
- mathematics
- theorem-proving
- verifiers
- agent-trajectories
---

# Value-as-Tool proof rollouts

Complete trajectories behind the [rollout explorer Space](https://huggingface.co/spaces/{space}):
every model call (prompt, private reasoning, response and tool calls), every verifier verdict and
value estimate, and the external judge's score. The trajectories cover olympiad proof problems from
IMO-ProofBench (60) and ProofBench (145).

| id | experiment | model | methods | seeds |
|---|---|---|---:|---|
{table}

Methods include Direct generation; Generator/Verifier/Reviser loops (GVR), GVR with subagents,
reference-assisted GVR and rationale+score feedback; solver-invoked value tools; a plan-work-review
harness; and 24 attempt-conditioned verifier arms. In those arms the verifier sees 8 labeled prior
Direct attempts (seeds 0–7 of the 9B harness run) as full solutions, solution summaries or
solution+thinking summaries, with or without the reference proof. The external judge is
`openai/gpt-oss-20b`, scoring 0–7, and strict success is 7/7.

## Layout

- `index/experiments.json`: experiments, methods (with descriptions) and code legends.
- `index/summary.json`: per experiment, benchmark scope and method: 7/7 rate with 95% bootstrap
  intervals, tokens, how runs end, GVR verifier calls, value-tool queries and the calibration of its
  probabilities, and fix and break rates against Direct. Built from each artifact's official
  `report/rows.jsonl`.
- `index/<benchmark>.json.gz`: per problem, one cell per rollout:
  `[seed, judge score, status code, verdict codes, generated tokens, calls]`.
- `problems/<benchmark>/<problem>.json.gz`: statement, solver prompt, reference proof, judge grading
  prompts (the candidate's text replaced by `⟦final⟧`), and the three prior-attempt evidence packs.
- `rollouts/<experiment>/<benchmark>/<problem>/<method>.json.gz`: all seeds of one method on one
  problem.

Trajectories are de-duplicated. Continuation calls list only the messages added after the call they
extend (`cont`). Assistant turns that repeat an earlier response are `{{"ref": call}}`. Text shown
elsewhere is replaced by markers:
- `⟦prompt⟧`, `⟦problem⟧`, `⟦reference⟧` and `⟦evidence⟧`;
- `⟦c<i>⟧` and `⟦r<i>⟧`: output and reasoning of call `i`;
- `⟦a<i>.<field>⟧`: a tool-call argument of call `i`.

Files are gzipped JSON.

## Provenance and terms

Generated with the `value_as_tool` harness from the artifacts listed in `index/experiments.json`.
Problem statements, reference proofs and grading guidelines come from `lm-provers/IMOProofBench` and
`lm-provers/ProofBench` and keep those datasets' terms. Model outputs are from Qwen models and are
subject to their licenses. Scores come from an automatic judge and are noisy, so treat individual
7/7 or 0/7 grades as evidence, not ground truth.
"""


def space_readme(dataset):
    return f"""---
title: Value-as-Tool Rollout Explorer
emoji: 🧮
colorFrom: indigo
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: mit
datasets:
- {dataset}
short_description: Problem-by-problem viewer of proof-harness rollouts
---

# Value-as-Tool Rollout Explorer

Browse olympiad proof problems one at a time, per benchmark. For each problem, compare every rollout
of every method:
- Direct generation;
- GVR, GVR with subagents, reference-assisted GVR and rationale+score feedback;
- value tools;
- plan-work-review;
- 24 attempt-conditioned verifier arms, on Qwen3.5-9B and Qwen3.8-27B.

Open any rollout to read it turn by turn: each generator, verifier and reviser call with its prompt,
private reasoning, response, tool call and routed verdict, then the external judge's score and
grading rubric.

The **Summary** tab compares the methods over all rollouts: 7/7 rate with 95% intervals, tokens,
how runs end, GVR verifier calls, value-tool queries and whether the value tool's probabilities are
calibrated. The **Analysis** tab asks what each method does on problems Direct gets wrong: fix and
break rates against Direct, and what GVR's verifier does with its first candidate.

This is a static app with no backend. It loads gzipped JSON on demand from
[{dataset}](https://huggingface.co/datasets/{dataset}) at a pinned revision. Model text is rendered
as plain DOM text with a limited Markdown subset, and math is rendered locally with KaTeX (license
in `vendor/katex/LICENSE`). App code is MIT licensed; data terms are in the dataset card.
"""


def config_js(dataset, revision):
    base = f"https://huggingface.co/datasets/{dataset}/resolve/{revision}"
    return f"window.ROLLOUT_CONFIG = {{dataBase: '{base}', datasetRepo: '{dataset}'}};\n"


def preflight(data):
    meta = json.loads((data / "index" / "experiments.json").read_text())
    for bench in meta["benchmarks"]:
        index = data / "index" / f"{bench['id']}.json.gz"
        assert index.exists(), f"missing {index}"
    rollouts = list((data / "rollouts").glob("*/*/*/*.json.gz"))
    problems = list((data / "problems").glob("*/*.json.gz"))
    assert rollouts and problems, f"no data under {data}"
    assert (data / "index" / "summary.json").exists(), (
        "missing index/summary.json; run build_summary.py"
    )
    size = sum(p.stat().st_size for p in data.rglob("*") if p.is_file() and ".cache" not in p.parts)
    print(f"data: {len(rollouts)} rollout files, {len(problems)} problems, {size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/checkpoint/fort/anikaitsingh/value_as_tool_rollout_viewer/data"),
    )
    parser.add_argument(
        "--namespace", default=None, help="HF user or org; defaults to the token's user"
    )
    parser.add_argument("--dataset-name", default="value-as-tool-rollouts")
    parser.add_argument("--space-name", default="value-as-tool-rollout-explorer")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run every local check and build the Space, upload nothing",
    )
    args = parser.parse_args()
    data = args.data.resolve()

    preflight(data)
    secrets = secret_values()
    print(
        f"scanning for credentials ({len(secrets)} local secret values + token patterns)…",
        flush=True,
    )
    hits = scan(data, secrets) + scan(HERE / "space", secrets)
    assert not hits, "credential-like content found; refusing to publish:\n" + "\n".join(
        f"  {p}: {f}" for p, f in hits
    )
    print("credential scan: clean")

    token = os.environ.get("HF_TOKEN") or env_values(REPO / ".env").get("HF_TOKEN")
    assert token, f"HF_TOKEN not set and not found in {REPO / '.env'}"
    namespace = args.namespace
    if args.dry_run and namespace is None:
        namespace = "<token-user>"
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    if namespace is None:
        namespace = api.whoami()["name"]
    dataset, space = f"{namespace}/{args.dataset_name}", f"{namespace}/{args.space_name}"
    (data / "README.md").write_text(dataset_card(data, dataset, space))

    staging = Path(tempfile.mkdtemp(prefix="rollout-space-"))
    shutil.copytree(HERE / "space", staging, dirs_exist_ok=True)
    (staging / "README.md").write_text(space_readme(dataset))
    if args.dry_run:
        (staging / "config.js").write_text(config_js(dataset, "<revision>"))
        print(f"dry run: would create public dataset {dataset} and public static Space {space}")
        print(f"staged Space at {staging}")
        return

    api.create_repo(dataset, repo_type="dataset", private=False, exist_ok=True)
    print(
        f"uploading {data} → datasets/{dataset} (resumable; rerun this command if interrupted)…",
        flush=True,
    )
    api.upload_large_folder(
        repo_id=dataset,
        repo_type="dataset",
        folder_path=data,
        num_workers=args.workers,
        ignore_patterns=[".cache/**"],
    )
    revision = api.dataset_info(dataset).sha
    print(f"dataset revision {revision}")
    (staging / "config.js").write_text(config_js(dataset, revision))
    api.create_repo(space, repo_type="space", space_sdk="static", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=space,
        repo_type="space",
        folder_path=staging,
        commit_message=f"Rollout explorer pinned to {dataset}@{revision[:12]}",
    )
    print(f"\nSpace:   https://huggingface.co/spaces/{space}")
    print(f"Dataset: https://huggingface.co/datasets/{dataset}")


if __name__ == "__main__":
    sys.exit(main())
