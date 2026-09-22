# Value as Tool

`value-as-tool` is a standalone, token-accounted evaluation harness for
Qwen3.5-9B on olympiad mathematics.  It compares direct generation with an
Aletheia-style Generator/Verifier/Reviser loop, an optional fresh-context
subagent tool, and a reference-assisted verifier ablation.

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
229,376-token trajectory budget unchanged.
Its verifier returns only a verdict, candidate-local excerpt, short category,
and a 600-character critique; an exact-copy filter removes long spans found
only in the reference. This is a bounded leakage heuristic, not a formal
non-interference guarantee: the ablation intentionally measures the benefit of
reference-informed verification and repair.

The published Aletheia artifacts describe the three-way routing but do not
release the role prompts, orchestration code, attempt limit, or token accounting.
This project therefore uses the precise term **Aletheia-style**, not an exact
reproduction of Aletheia.

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
value-as-tool --config experiment.yaml prepare
value-as-tool --config experiment.yaml solve --shard-index 0 --shard-count 16
value-as-tool --config experiment.yaml judge --shard-index 0 --shard-count 16
value-as-tool --config experiment.yaml report
value-as-tool --config experiment.yaml status
```

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
proof problem at seed 0 through all four conditions and then judges the four
outputs; both configured endpoints must already be running. It writes ordinary
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
- `pass@k` is computed from the three independent final trajectories;
- `best-grade@k` is the expected maximum continuous proof grade.

Only the final trajectory output is an evaluation sample; intermediate
candidates and subagents are diagnostics.  Missing or failed scheduled results
score zero and make the report explicitly incomplete.  `pass@k` and
`best-grade@k` are paired with final-attempt and total incurred `cost@k`, since
three samples can consume three trajectory budgets. Incurred cost includes
invalidated retries and is marked inexact when an interrupted request requires
an upper bound; external-judge cost remains a separate accounting bucket.

## Development

```bash
uv run pytest
uv run ruff check .
```

Tests use fake endpoints and fixtures.  They do not download datasets or model
weights, start inference servers, or submit scheduler jobs.
