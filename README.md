# Value as Tool

`value-as-tool` is a standalone, token-accounted evaluation harness for
Qwen-family models on olympiad mathematics. It compares direct generation with
Aletheia-style Generator/Verifier/Reviser loops, value-verifier tools,
fresh-context subagents, reference-assisted ablations, and source-hashed custom
proof harnesses.

The repository does not import or read `../reasoning_bank` at runtime.  Model,
dataset, prompt, sampling, and judging revisions needed to reproduce the direct
protocol are pinned in `experiment.yaml` and in the prompt manifest.

The compatibility path intentionally preserves the prior QED judge request
shape: up to 95,000 output tokens are requested until a rendered proof prompt
exceeds 97,280 tokens, at which point deterministic middle truncation and a
32,768-token output cap are used. On a strict 131,072-token server, prompts in
the intervening range can be rejected; keeping this behavior is necessary for
request-level parity with the direct result rather than silently changing its
evaluation protocol.

## Experiment

The default schedule covers:

- `lm-provers/IMOProofBench` (60 proof problems)
- `lm-provers/ProofBench` (145 proof problems)
- `Hwilner/imo-answerbench` (400 answer problems)

Each applicable condition runs seeds 0, 1, and 2 with a hard cap of 229,376
generated Qwen tokens per trajectory.  The cap is shared by every Generator,
Verifier, Reviser, tool-continuation, and child-agent call.  Prompt tokens and
the independent judge are reported separately.  The reference-assisted arm is
not scheduled for IMO-AnswerBench because that dataset has no reference proof.
Within the shared cap, the initial candidate is capped at 98,304 tokens, each
verifier at 32,768, and later candidate/subagent/verifier calls share a hard
98,304-token correction pool that reserves a viable call for each of the three
configured candidate versions. This allocation leaves enough room for Qwen's
thinking mode to complete the required verifier tool call while keeping the
229,376-token trajectory budget unchanged. The SGLang solver deliberately uses
ordinary decoding rather than NEXTN/EAGLE speculation: in the pinned SGLang
release, speculative verification can bypass the per-request thinking-budget
processor and consume the reserved response tokens as hidden reasoning.
Its verifier returns only a verdict, candidate-local excerpt, short category,
and a 600-character critique; an exact-copy filter removes long spans found
only in the reference. This is a bounded leakage heuristic, not a formal
non-interference guarantee: the ablation intentionally measures the benefit of
reference-informed verification and repair.

The published Aletheia artifacts describe the three-way routing but do not
release the role prompts, orchestration code, attempt limit, or token accounting.
This project therefore uses the precise term **Aletheia-style**, not an exact
reproduction of Aletheia.

### Static proof-harness registry

An experiment may select explicit `module:Class` entries under
`evaluation.harnesses`. The checked-in registry contains the eight existing
methods (`direct`, `gvr`, `gvr_subagents`, `gvr_reference`, `value_tool`, and
their three rationale-score variants) plus `cch_plan_work_review`. Each module
declares whether it is blind or reference-assisted, and its source SHA-256 is
recorded in the schedule so resumed artifacts cannot silently switch policies.
The trusted runtime retains token accounting, model access, and checkpointing;
harness modules control only their orchestration policy.

`cch_plan_work_review` adapts the Plan → Work → fresh Review → bounded Retake
shape from the Claude Code Harness as a proof-agent design. It does not invoke
Claude Code or expose Bash, filesystem, or network tools to a trajectory. The
method is blind: it never receives the benchmark reference proof. It can spend
16,384 tokens planning, 65,536 on the initial proof, 24,576 on each of three
reviews, and up to 40,960 and 32,768 on two repairs. An approving review ends
the trajectory early; the worst-case total remains the shared 229,376-token
cap.

`experiment_value_tool.yaml` defines a separate paired `direct`/`value_tool`
experiment. In the latter condition the solver may call
`query_success_probability` up to three times. Each query gives a fresh-context
Qwen verifier the original problem and the solver's actual partial trace; the
query takes no semantic arguments, and any provider-emitted payload is ignored.
The solver receives only `{"success_probability": p}`. These are model-reported
estimates, not calibrated probabilities. Solver and verifier generations share
the same 229,376-token trajectory cap. Per-query estimates and trace hashes are
stored in each trajectory and exported to `rows.jsonl`; reports also separate
value-verifier token use.

`experiment_rationale_score.yaml` selects the proof benchmarks only and runs
three explicitly versioned conditions: `gvr_rationale_score`,
`value_tool_rationale_score`, and `gvr_reference_rationale_score`. Each verifier
returns a model-reported success probability plus a concise rationale. GVR
retains its categorical verdict solely for routing, while the rationale and
score are passed to the next Generator/Reviser turn. The tool condition returns
the same two fields to the calling solver. Reference-only copied spans are still
removed before reference-verifier feedback reaches the solver.

### Attempt-conditioned verification: 24 arms

`experiment_qwen35_9b_attempt_conditioning.yaml` crosses four factors:

| Factor | Variants |
| --- | --- |
| Prior-attempt evidence | Full solutions and labels; solution summaries and labels; solution-and-thinking summaries and labels |
| Interaction | GVR; solver-controlled value tool |
| Feedback | Legacy; rationale and success probability |
| Reference access | Prior attempts alone; prior attempts plus the benchmark reference solution |

The `legacy` suffix means the existing feedback protocol: GVR returns its
verdict and bounded critique; the value tool returns only a probability.
`rationale` adds the rationale-and-score protocol. This factor does not disable
Qwen's thinking mode. Only verifier calls receive the attempt evidence and,
for `gold` arms, the reference solution. Generator/Reviser turns receive the
verifier's filtered feedback.

The default reuses Direct final responses from Qwen3.5-9B seeds 0–7 in
`artifacts/qwen35_9b_harness_large_budget_v1`. A completed external judge grade
of 7/7 labels an attempt successful; other valid grades label it unsuccessful.
Failed source solves and missing or invalid labels are recorded as exclusions.
Problems with only successful or only unsuccessful attempts stay in the bank.
The same pinned Qwen3.5-9B model produces both summary forms, using resumable
map/reduce tasks with 2,048-token map and 8,192-token final summary targets.
Long material is split and summarized recursively; full-solution packs are
never silently shortened to fit a verifier call.

Evaluation uses fresh seeds 8–15 on both proof benchmarks: 205 problems ×
24 arms × 8 seeds = 39,360 trajectories. Each trajectory has the configured
8,388,608-token shared generation cap and 262,144-token context limit.
Source-attempt, preprocessing, evaluation, and external-judge costs are kept
separate so reused evidence is not counted as fresh generation in each arm.

Build and pin the bank before preparing a launch. Run the build command with a
working copy of the experiment YAML, then set that file's
`conditioning.bank_sha256` to the returned `sha256` value:

```bash
cp experiment_qwen35_9b_attempt_conditioning.yaml experiment.attempt-run.yaml
uv run value-as-tool --config experiment.attempt-run.yaml conditioning build-bank
```

The following dry run creates a source snapshot, copies the pinned local model
manifest, and prepares the preprocessing queue. It queries scheduler capacity
and prints commands without submitting jobs or making model requests:

```bash
uv run python scripts/submit_attempt_conditioning.py launch \
  --config experiment.attempt-run.yaml \
  --control-root artifacts/attempt-launch --dry-run
```

Launch the prepared manifest and inspect its progress with:

```bash
uv run python scripts/submit_attempt_conditioning.py launch \
  --manifest artifacts/attempt-launch/submission.json --submit
uv run python scripts/submit_attempt_conditioning.py status \
  --manifest artifacts/attempt-launch/submission.json
```

The controller runs Qwen summarization first, freezes all evidence packs, writes
`experiment.frozen.yaml` with the manifest digest, and only then creates the
evaluation schedule. It gates the full run on 48 pilot trajectories: all arms
on one problem per benchmark at seed 8. The default pilot problems have median
Direct token cost; `--pilot-problem BENCHMARK=PROBLEM` selects explicit problems
during launch preparation. Workers use one GPU and two concurrent requests,
with current-stage admission capped at 212 high-QoS and 64 shared-QoS GPUs after
accounting for other jobs owned by the same user.

Reusing the same submission manifest preserves completed work; repeating
`launch --manifest ... --submit` does not create a second controller. Slurm
requeues and abandoned worker claims resume from durable checkpoints. A stopped
controller whose manifest remains `running` can be resumed with
`supervise --manifest ...` in the prepared source environment. Failed pilots,
failed preprocessing, changed pinned inputs, and ambiguous submissions stop
the controller with a recorded reason; inspect that reason before recovery.
The final controller stage writes the reports and a complete coverage/context
audit. Model correctness is separate from execution completion.

## Setup

```bash
uv sync --extra dev
cp .env.example .env  # only if an environment file does not already exist
uv run value-as-tool --config experiment.yaml preflight
```

For local serving, install SGLang and vLLM in separate environments because
their dependency sets conflict:

```bash
UV_PROJECT_ENVIRONMENT=.venv-sglang uv sync --extra sglang
UV_PROJECT_ENVIRONMENT=.venv-vllm uv sync --extra vllm
export VALUE_AS_TOOL_PYTHON="$PWD/.venv/bin/python"
export VALUE_AS_TOOL_SGLANG_BIN="$PWD/.venv-sglang/bin/sglang"
export VALUE_AS_TOOL_VLLM_BIN="$PWD/.venv-vllm/bin/vllm"
```

The Slurm launcher forwards those executable overrides while using the base
environment's Python for the orchestration process.

Remote or already-running OpenAI-compatible endpoints can be selected by
overriding the URLs in the YAML configuration.  API key values are read only
from the configured environment-variable names and are never persisted.
For endpoint-only use, `prepare --tokenizers-only` downloads the pinned
datasets and tokenizer/configuration files needed for exact context accounting,
without downloading model weights. Local `serve` intentionally refuses such a
manifest until a full `prepare` has been run.

## Commands

```bash
value-as-tool harness list
value-as-tool --config experiment_qwen38_27b_harness.yaml harness validate
value-as-tool --config experiment.yaml prepare
value-as-tool --config experiment.yaml solve --shard-index 0 --shard-count 16
value-as-tool --config experiment.yaml judge --shard-index 0 --shard-count 16
value-as-tool --config experiment.yaml report
value-as-tool --config experiment.yaml status
```

The value-tool variant can be launched independently with:

```bash
VALUE_AS_TOOL_SLURM_QOS=h200_rsci_bigjob \
VALUE_AS_TOOL_SLURM_PARTITION=h200_rsci \
VALUE_AS_TOOL_SLURM_ACCOUNT=mathsi \
VALUE_AS_TOOL_MAX_CONCURRENT_GPUS=128 \
  ./scripts/submit.sh experiment_value_tool.yaml
```

The nine-method proof-only run uses the pinned `Qwen/Qwen3.8-27B` checkpoint
for every solver, verifier, reviser, and subagent role, while retaining the
pinned `openai/gpt-oss-20b` external judge:

```bash
uv run value-as-tool --config experiment_qwen38_27b_harness.yaml harness validate
VALUE_AS_TOOL_SLURM_QOS=h200_rsci_bigjob \
VALUE_AS_TOOL_SLURM_PARTITION=h200_rsci \
VALUE_AS_TOOL_SLURM_ACCOUNT=mathsi \
VALUE_AS_TOOL_MAX_CONCURRENT_GPUS=128 \
  ./scripts/submit.sh experiment_qwen38_27b_harness.yaml
```

That config evaluates IMO-Proof and ProofBench at seeds 0, 1, and 2: 205
problems × 9 methods × 3 seeds = 5,535 trajectories. It does not schedule the
9B model.

`experiment_qwen35_9b_harness_large_budget.yaml` runs the same nine methods
with pinned `Qwen/Qwen3.5-9B` and **eight samples per problem** (seeds 0–7):
205 problems × 9 methods × 8 seeds = **14,760 trajectories**. It writes to the
separate `artifacts/qwen35_9b_harness_large_budget_v1` directory. Thinking stays
enabled, with no forced thinking cutoff. The shared generation ceiling is
8,388,608 tokens; larger phase, verifier, child, and CCH-stage allowances avoid
the smaller allocations used in the 27B run. Each request still fits within
the native 262,144-token context, including its prompt and 1,024-token headroom,
so native context limits can still truncate a completion.

This run uses 384 solve and judge shards, one GPU per server, and two concurrent
trajectories per server. The launch splits both stages across
`g3_scientific-reasoning_high` (up to 128 GPUs) and `g3_core_shared` (up to 64
GPUs), always with `--segment=1`, for an aggregate cap of 192 GPUs. A 115-cell
pilot covers every method and benchmark before the controller expands the run.
Pilot cells are retained for resume; full judging follows solving. The external
judge remains pinned GPT-OSS-20B at
medium effort with its existing 95,000-token output cap.

```bash
uv run value-as-tool --config experiment_qwen35_9b_harness_large_budget.yaml harness validate
.venv/bin/python scripts/submit_qwen35_large_budget.py launch \
  --config experiment_qwen35_9b_harness_large_budget.yaml \
  --run-root artifacts/qwen35_9b_large_budget_launch_001
```

The dedicated launcher records its immutable snapshot, exact submissions,
pilot audit, and final completion in the run directory. It verifies and reuses
the local pinned models and prepared benchmark data; it refuses nonempty output
directories. Use the snapshot configuration and recorded path overrides for
status or manual recovery.

To expand an already running pilot immediately, while retaining its active
work, use the separate scheduler controller:

```bash
.venv/bin/python scripts/expand_qwen35_large_budget.py \
  --manifest artifacts/qwen35_9b_large_budget_launch_001/submission.json \
  --high-cap 212 --shared-cap 64
```

This overrides the pilot completion gate and permits up to 276 GPUs across the
two QoS pools. It reserves three slots for existing pilot solve or judge jobs,
and makes matching full-array tasks wait for their pilot solve jobs. Full
judging waits for both solve arrays and all pilot judges. The source snapshot,
experiment settings, and final coverage and budget audit remain unchanged.

The general `scripts/submit_slurm.sh` launcher limits aggregate allocation to 128 GPUs by
default and rejects larger values. Set `VALUE_AS_TOOL_MAX_CONCURRENT_GPUS` to a
value from 1 through 128 for a lower hard cap; tensor-parallel solve tasks count
all GPUs they reserve. Solve and judge are dependency-ordered, so their GPU
allocations do not overlap. When multiple tasks share a node, each runner holds
lock-backed reservations for its
HTTP port and, for SGLang, its internal `--nccl-port`; this prevents one model
server's distributed TCPStore from colliding with another server.

To reconstruct only the direct cells in a separate artifact directory, apply
the same two overrides to every stage, for example:

```bash
direct=(--config experiment.yaml \
  --set 'evaluation.conditions=[direct]' \
  --set paths.artifact_root=artifacts/direct-reproduction)
uv run value-as-tool "${direct[@]}" prepare
uv run value-as-tool "${direct[@]}" solve
uv run value-as-tool "${direct[@]}" judge
uv run value-as-tool "${direct[@]}" report
```

This reruns the pinned protocol; historical outputs and headline values are not
bundled or used as acceptance checks.

A nonempty direct response ending with `finish_reason=length` is retained in
the artifact, but this experiment's predeclared policy treats every exhausted
trajectory as a zero and leaves the report incomplete. Therefore exact
headline parity with a legacy runner that judged truncated responses is
conditional on no direct sample reaching the 229,376-token limit.

`value-as-tool all` runs those stages against already-running endpoints.
`value-as-tool smoke` is an explicit live check that runs one reference-bearing
proof problem at seed 0 through every condition in the selected config and then
judges the outputs; both configured endpoints must already be running. It writes ordinary
resumable cells, so a subsequent full run safely reuses them. Use
`--benchmark`, `--problem-id`, or `--seed` to select another scheduled case.
`value-as-tool serve qwen` and `value-as-tool serve judge` start the pinned local
servers.  `scripts/submit_slurm.sh` builds the corresponding Slurm dependency graph.
None of those launch commands run during installation or tests.

## Results

Reports distinguish the legacy direct metric from actual success:

- proof mean grade is the mean 0–7 judge score, normalized to a percentage;
- proof success means an exact 7/7 grade;
- answer success means judged correctness;
- `pass@k` is computed from the configured independent final trajectories,
  for every `k` up to the sample count (including pass@8 for the eight-seed run);
- `best-grade@k` is the expected maximum continuous proof grade.

Only the final trajectory output is an evaluation sample; intermediate
candidates and subagents are diagnostics.  Missing or failed scheduled results
score zero and make the report explicitly incomplete.  `pass@k` and
`best-grade@k` are paired with final-attempt and total incurred `cost@k`, since
`k` samples can consume `k` trajectory budgets. Incurred cost includes
invalidated retries and is marked inexact when an interrupted request requires
an upper bound; external-judge cost remains a separate accounting bucket.

## Development

```bash
uv run pytest
uv run ruff check .
```

Tests use fake endpoints and fixtures.  They do not download datasets or model
weights, start inference servers, or submit scheduler jobs.
