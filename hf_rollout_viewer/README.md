# Rollout explorer

A static Hugging Face Space for reading the proof-harness rollouts in `artifacts/` problem by problem.
Pick a benchmark and a problem to see its statement, reference proof, judge rubric and prior-attempt
evidence packs, plus a methods × seeds grid of judge scores for every experiment. Click a cell to read
that rollout turn by turn: each call's prompt, private reasoning, response, tool call and routed
verdict, then the judge's score. The **Summary** tab compares methods over all rollouts (7/7 rate,
tokens, how runs end, verifier calls, value-tool queries and calibration); the **Analysis** tab asks
what each method does on the problems Direct gets wrong.

| Experiment | Artifact | Methods | Seeds |
|---|---|---:|---|
| `q9_base` | `qwen35_9b_harness_large_budget_v1` | 9 | 0–7 |
| `q9_attempt` | `qwen35_9b_attempt_conditioning_v1` | 24 | 8–15 |
| `q27_base` | `qwen38_27b_harness_v4` | 9 | 0–2 |

`build_data.py` turns the 23 GB of `result.json` trajectories into 2.8 GB of gzipped JSON, one file per
(experiment, benchmark, problem, method). Repeated text (the problem, candidates, evidence packs,
earlier turns) is stored once and referenced by `⟦key⟧` markers, which the viewer expands on click.

```bash
python hf_rollout_viewer/build_data.py --out /checkpoint/fort/$USER/value_as_tool_rollout_viewer/data
```

`build_summary.py` then writes `index/summary.json` for the two statistics tabs. It reads each artifact's
official `report/rows.jsonl`, so its 7/7 rates, mean grades and token means match `summary.csv`, and
takes calls per run from the rollout index. It needs numpy and runs in seconds.

```bash
python hf_rollout_viewer/build_summary.py --data /checkpoint/fort/$USER/value_as_tool_rollout_viewer/data
```

Preview locally by serving a directory that contains both `space/` and `data/`:

```bash
mkdir -p /tmp/rv_serve
ln -sfn "$PWD/hf_rollout_viewer/space" /tmp/rv_serve/space
ln -sfn /checkpoint/fort/$USER/value_as_tool_rollout_viewer/data /tmp/rv_serve/data
python3 -m http.server 8765 --directory /tmp/rv_serve   # open http://localhost:8765/space/
# statistics tabs: http://localhost:8765/space/#v=summary and #v=analysis
```

`publish.py` scans every file for credentials and aborts on any hit. It then uploads the data to a
public dataset repo (resumable) and publishes the public static Space, pinned to that dataset commit.
It reads `HF_TOKEN` from `.env` and must run where huggingface.co is reachable.

```bash
python hf_rollout_viewer/publish.py --dry-run
python hf_rollout_viewer/publish.py
```

## GVR tree explorer

A second static Space for the branched GVR collection on ArXivMath (Qwen3.5-9B; harness
`gvr_branched`): one tree per problem with 10 verification points × 4 branches, every candidate
coloured by the answer judge and every edge by the verdict that routed it. It has a **Trees** tab
(diagram, table view, candidate text, critiques, reasoning and prompt templates) and an **Overview**
tab of dataset-level tables.

`build_trees.py` reads the artifact root's `export/` (from `scripts/export_gvr_tree_dataset.py`) plus
each tree's `result.json` for reasoning and prompts. It writes `index/`, one small
`trees/<problem>.json.gz` per tree (~40 KB) and one `reasoning/<problem>.json.gz` per tree (~1 MB),
which the Space loads only when a reasoning panel is opened. It uses the standard library only.

```bash
python hf_rollout_viewer/build_trees.py --out /checkpoint/fort/$USER/value_as_tool_tree_viewer/data
```

Preview by serving `tree_space/` next to the data, then open http://localhost:8765/tree_space/:

```bash
mkdir -p /tmp/tree_serve
ln -sfn "$PWD/hf_rollout_viewer/tree_space" /tmp/tree_serve/tree_space
ln -sfn /checkpoint/fort/$USER/value_as_tool_tree_viewer/data /tmp/tree_serve/data
python3 -m http.server 8765 --directory /tmp/tree_serve
```

`publish_trees.py` reuses `publish.py`'s credential scan, uploads the data to the public dataset
`value-as-tool-gvr-trees` and publishes the static Space `value-as-tool-gvr-tree-explorer`, pinned to
that dataset commit. Like `publish.py`, it reads `HF_TOKEN` from `.env` and must run where
huggingface.co is reachable.
