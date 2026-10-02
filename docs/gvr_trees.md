# Branched GVR trees on ArXivMath: status and what's left

Last updated 2026-10-02 (harness v2 re-pilot running).

## Where things stand

**Goal.** Collect training data for value and verifier learning from Generator → Verifier → Reviser (GVR) rollouts of Qwen3.5-9B on ArXivMath. The data has the shape of a tree: at every verification point the rollout branches into several continuations, and an external judge grades every candidate.

**Status of each piece:**
1. **Baseline collection (`gvr_branched`): complete.** 1,649 trees, every candidate judged and exported, browsable in a public Space. The trees turned out *flat*, meaning answers rarely change, so a second harness was built.
2. **Replanning harness (`gvr_replan`): piloted.** At every branching point the model plans its own next steps. Two variants were piloted on 30 problems:
   - joint: one planner call proposes four plans;
   - independent: four planner calls propose one plan each.
3. **Next collection: a decision is pending** (see [What's left](#whats-left)).

## What was built

| Piece | Where |
|---|---|
| ArXivMath dataset (download on a `cpu_x86` node, train/eval split matching meta-rubrics) | `scripts/prepare_arxivmath_dataset.py`, `scripts/slurm/run_arxivmath_download.sh` |
| Prepared-JSONL benchmarks `arxivmath_train` / `arxivmath_eval` | `config.py` (`PreparedDatasetConfig`), `benchmarks.py`, `assets.py` |
| Judge agreement study (judge vs MathArena labels) | `scripts/arxivmath_judge_agreement.py` |
| Branched GVR harness: 10 points × 4 branches, verdict-routed re-check / revise / regenerate | `src/value_as_tool/harnesses/gvr_branched.py`. Runtime support: `HarnessRuntime.verify`, branch keys on candidates and verdicts, `orchestrator._verifier_request` |
| Replanning harness, joint and independent planners | `src/value_as_tool/harnesses/gvr_replan.py` |
| Per-candidate ("node") judging | `pipeline.judge_nodes`, CLI `judge-nodes` |
| Launcher: pilot, gate, approval-gated full run, finalize | `scripts/submit_gvr_tree_collection.py`, `scripts/slurm/run_tree_controller.sbatch` |
| Flat dataset export (trees, nodes, verifications, plans) | `scripts/export_gvr_tree_dataset.py` |
| Paired comparison of collections | `scripts/compare_gvr_tree_variants.py` |
| Hugging Face viewers: rollout explorer and GVR tree explorer | `hf_rollout_viewer/` (`build_*.py`, `publish*.py`, `space/`, `tree_space/`) |
| Experiment configs | `experiment_qwen35_9b_gvr_tree_arxivmath.yaml`, `experiment_qwen35_9b_gvr_replan_{joint,independent}_arxivmath.yaml` |

`uv run pytest` covers all of it. Relevant files:
- `tests/test_gvr_branched_harness.py`
- `tests/test_gvr_replan_harness.py`
- `tests/test_gvr_tree_end_to_end.py` (prepare → solve → judge-nodes → export for all three harnesses)
- `tests/test_gvr_tree_launch.py`
- `tests/test_gvr_tree_comparison.py`
- `tests/test_node_judge.py`
- `tests/test_prepare_arxivmath.py`

## Where the data lives (not in git)

| What | Control root (`artifacts/`) | Artifact root (`/checkpoint/fort/anikaitsingh/value_as_tool/`) |
|---|---|---|
| Baseline collection | `qwen35_9b_gvr_tree_launch_20260930T191640Z`: `submission.json`, `pilot-report.json`, `completion.json` | `qwen35_9b_gvr_tree_arxivmath_v1`, with the export in `export/` |
| Joint pilot | `qwen35_9b_gvr_replan_joint_launch_20261001T214715Z`: `pilot-report.json`, `pilot-comparison.json` | `qwen35_9b_gvr_replan_joint_arxivmath_v1` |
| Independent pilot | `qwen35_9b_gvr_replan_independent_launch_20261001T214715Z`: `pilot-report.json` | `qwen35_9b_gvr_replan_independent_arxivmath_v1` |
| Joint re-pilot (harness v2, running) | `qwen35_9b_gvr_replan_joint_v2_launch_20261002T124031Z` | `qwen35_9b_gvr_replan_joint_arxivmath_v2` |

**Public Hugging Face repos:**
- **Tree explorer.** Space: https://huggingface.co/spaces/asingh15/value-as-tool-gvr-tree-explorer. Data: https://huggingface.co/datasets/asingh15/value-as-tool-gvr-trees, pinned to revision `c1c7b17f`. It shows the baseline trees.
- **Rollout explorer.** Space: https://huggingface.co/spaces/asingh15/value-as-tool-rollout-explorer.

## Results so far

### Baseline collection (`gvr_branched`, 1,649 trees)

**Size and cost:**
- 1,502 train and 147 eval trees.
- 67,609 candidates, all judged, and 65,960 verdicts.
- 2.01B generated tokens.
- 376 GPU-hours to solve, plus 1.5 to judge; about 4 hours of wall time at 8 trees per GPU.

**Label quality.** The answer judge (gpt-oss-20b, extracted `\boxed{}` vs gold) agrees with MathArena's labels on 98.7% of 999 Qwen3.6 attempts (κ 0.974). All disagreements are lenient; most are equivalent forms.

**Flatness:**
- 38.5% of trees have a single final answer.
- 60% of spines never change their answer.
- 21% of trees contain both correct and incorrect candidates.

**Why the trees are flat:**
- The verifier says "correct" for 82% of wrong candidates.
- So 85% of revisions are re-checks, and re-checks keep the parent's answer 98% of the time. Revise keeps it 86% of the time and regenerate 36%.
- Regenerate fixes 8.8% of wrong parents but breaks 39% of right ones.

**Accuracy and the verifier:**
- Spine accuracy rises from 28.8% (c1) to 32.5% (c10); 42.3% of trees have at least one correct candidate.
- The verifier is weakly informative: it says "correct" for 89.6% of right candidates vs 82.4% of wrong ones.

### Replanning pilots (30 paired problems, same as the baseline pilot)

| Metric | Baseline | Joint planner | Independent planners |
|---|---|---|---|
| Child changes the parent's answer | 6.9% | **19.7%** | 10.0% |
| Verification points whose children include both ✓ and ✗ | 5.3% | 7.0% | 2.3% |
| Distinct answers per tree | 2.67 | **4.90** | 2.70 |
| Spines that never change answer | 63% | **30%** | 53% |
| Fix rate (✗ parent → ✓ child) / break rate (✓ → ✗) | 1.0% / 2.8% | 1.8% / 7.3% | 1.1% / 2.2% |
| Trees with any ✓ / c10 accuracy | 47% / 33% | 43% / 33% | 40% / 30% |
| Plans that show the current solution | – | 56% | 82% |
| Tokens per tree / wall time per tree | 1.11M / 38 min | 1.20M / 61 min | 1.14M / 41 min |

**What the pilots show:**
- **Joint is the only real change.** Its answer-change rate exceeds both the baseline (+12.8 points, Holm-corrected p = 0.001) and independent (+9.7 points, p = 0.014). Independent is not distinguishable from the baseline.
- **Why independent stays flat.** Independent planners mostly write "confirm the solution" plans when all four verifiers agree.
- **The extra changes are mostly between wrong answers.** Points with mixed child labels did not increase significantly, breaks rose more than fixes, and accuracy is unchanged. Problems Qwen3.6 never solved stay all-wrong in every arm.
- **Hiding the current solution is where exploration happens.** Such plans fix 3–4% of wrong parents (under 1% when the solution is shown), and they also break more.
- **The model's own probabilities are uninformative.** Verifier and planner probabilities average 0.83–0.97 against about 31% accuracy, with AUROC 0.48–0.56.
- **Gate failures.** The independent pilot passed its health gate. The joint pilot failed one check: 46 of 1,200 branches (3.8%) got no plan, because the model sometimes leaves one plan's `success_probability` out of the 16-field tool call.
  - **Cause.** SGLang enforces a tool's parameter schema only for `strict` tools, so "required" fields are advisory. The model most often ended the call right after plan 4's show flag (`plan_4_success_probability` was missing in 48 of 61 bad calls), and once wrote the fields as prose inside a brief.
  - **Fix (harness v2), without constrained decoding:** the planner prompt names all fields in order, each plan's show flag and probability now come before the brief, and a plan without a valid probability still runs (probability recorded as `null`).

## What's left

### 1. Decide the next collection (blocked on you)

**Recommended:** the joint planner (harness v2) if the goal is diverse actions and states.

1. **Done: fix the joint failure.** Harness v2 (described under Gate failures above) has a more explicit planner prompt, short fields before the brief, and runs plans without a probability. Editing the harness changed its source hash, so v2 has new run IDs.
2. **In progress: re-pilot.** The joint v2 re-pilot runs on the same 30 problems (launch `qwen35_9b_gvr_replan_joint_v2_launch_20261002T124031Z`). When it finishes:
   - run `pilot-report` and `compare_gvr_tree_variants.py` with the baseline, joint v1 and joint v2 as arms;
   - check that field omissions and planner failures are near zero;
   - check that diversity matches v1.
3. **Full run.** `launch` → you write `approve-full.json` in the launch's control root → `full --lanes 8` → finalize/export. Estimated cost: about 2.0B tokens (about 400 GPU-hours). Trees take about 1 hour each, because the planner waits for all four verifiers.

**If the goal is instead contrast between ✓ and ✗ within states,** neither variant moves that much. Options:
- drop or down-weight problems Qwen3.6 never solved (44% of the set);
- deeper branching, where siblings continue for several steps;
- planner inputs without verifier probabilities, which are uninformative.

### 2. Viewer support for replanning trees

Before publishing replanning trees, `hf_rollout_viewer/build_trees.py` and `tree_space/app.js` need:
- the `exec` mode (edges currently fall back to the re-check style);
- plan display: title, brief, show flag and probability;
- verifier probabilities;
- a collection switcher (baseline / joint / independent).

Then publish with `publish_trees.py`, as for the baseline.

### 3. Known issues and follow-ups

- **Transport errors restart whole trees.** A request that fails without usage data (for example a connection reset) makes the trajectory `INVALID_USAGE`, and the tree restarts from scratch. This happened to 3 of 1,649 baseline trees. Retrying transient transport errors before invalidating would save that work.
- **Sampling is not reproducible across servers.** Identical c1 requests (same prompt and seed) produced different c1 in different arms on 2–3 of 30 problems, so c1 is only an approximate placebo.
- **Judge-server startup can be very slow on some nodes.** It took 22 minutes on g3-151-187, against about 5 minutes normally.
- **Don't use model probabilities as value labels.** Verifier and planner probabilities have AUROC ≈ 0.5 on ArXivMath.
- **Unpublished README edit.** The rollout explorer README has a local edit (the calibration mention) that was never published.
- **Not started:** training a value model on the collected trees, which is the eventual purpose of the data.

## Reproduce

Build the dataset (download on a `cpu_x86` node, build on the login node):

```bash
srun -p cpu_x86 --cpus-per-task=4 --mem=16G --time=01:00:00 scripts/slurm/run_arxivmath_download.sh
uv run python scripts/prepare_arxivmath_dataset.py build
```

Launch a collection (one harness per launch), then run the pilot report:

```bash
uv run --offline python scripts/submit_gvr_tree_collection.py launch \
  --config experiment_qwen35_9b_gvr_replan_joint_arxivmath.yaml \
  --run-root artifacts/<launch dir> --artifact-root /checkpoint/fort/$USER/value_as_tool/<root> \
  --pilot-gpus 8 --lanes 8 --submit \
  --baseline-manifest artifacts/qwen35_9b_gvr_tree_launch_20260930T191640Z/submission.json
uv run --offline python scripts/submit_gvr_tree_collection.py pilot-report --manifest artifacts/<launch dir>/submission.json
```

Full run (after you write `approve-full.json`), then status:

```bash
uv run --offline python scripts/submit_gvr_tree_collection.py full --manifest artifacts/<launch dir>/submission.json --lanes 8
uv run --offline python scripts/submit_gvr_tree_collection.py status --manifest artifacts/<launch dir>/submission.json
```

Export and compare (run from `scripts/`):

```bash
python export_gvr_tree_dataset.py --artifact-root <root> --output <root>/export
python compare_gvr_tree_variants.py --arm baseline=<baseline root> --arm joint=<joint root> \
  --arm independent=<independent root> \
  --problems-manifest ../artifacts/qwen35_9b_gvr_tree_launch_20260930T191640Z/submission.json \
  --output comparison.json
```

Viewer: build the data, then publish from a `cpu_x86` node with `HF_TOKEN` from `.env`:

```bash
python3 hf_rollout_viewer/build_trees.py --out /checkpoint/fort/$USER/value_as_tool_tree_viewer/data
srun -p cpu_x86 bash /checkpoint/fort/$USER/value_as_tool_rollout_viewer/work/rv_job.sh publish-trees
```
