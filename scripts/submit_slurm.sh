#!/usr/bin/env bash
# Create an immutable source snapshot and submit the full dependency graph.
#
# This script performs no work until invoked by a user. It never copies .env;
# jobs receive only the path to the live credential file.
set -Eeuo pipefail
umask 077

readonly REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
readonly SOURCE_CONFIG=${1:-${REPO_ROOT}/experiment.yaml}
default_python=${REPO_ROOT}/.venv/bin/python
if [[ ! -x $default_python ]]; then
  default_python=$(command -v python3.12 || command -v python3)
fi
readonly PYTHON_BIN=${VALUE_AS_TOOL_PYTHON:-$default_python}
unset default_python
readonly ENV_FILE=${VALUE_AS_TOOL_ENV_FILE:-${REPO_ROOT}/.env}

for command_name in flock sbatch sha256sum tar; do
  command -v "$command_name" >/dev/null || { echo "missing command: $command_name" >&2; exit 2; }
done
[[ -r $SOURCE_CONFIG ]] || { echo "missing config: $SOURCE_CONFIG" >&2; exit 2; }
[[ -x $PYTHON_BIN ]] || { echo "Python 3.12 is required" >&2; exit 2; }

export PYTHONPATH=${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}
readarray -t config_values < <(
  "$PYTHON_BIN" - "$SOURCE_CONFIG" "$REPO_ROOT" <<'PY'
import sys
from pathlib import Path
from value_as_tool.pipeline import load_context

source = Path(sys.argv[1]).resolve()
context = load_context(source)
print(context.artifact_root)
print(context.asset_root)
print(context.config.runtime.solve_shards)
print(context.config.runtime.judge_shards)
print(context.config.runtime.solver_backend)
print(getattr(context.config.runtime, "solver_tensor_parallel_size", 1))
print(context.config_fingerprint)
PY
)
readonly ARTIFACT_ROOT=${config_values[0]}
readonly ASSET_ROOT=${config_values[1]}
readonly SOLVER_BACKEND=${config_values[4]}

# Shard counts are parallelism, not protocol: overriding them here keeps the
# experiment fingerprint stable so a run can be resumed at a different width.
readonly SOLVE_SHARDS=${VALUE_AS_TOOL_SOLVE_SHARDS:-${config_values[2]}}
readonly JUDGE_SHARDS=${VALUE_AS_TOOL_JUDGE_SHARDS:-${config_values[3]}}
readonly MAX_CONCURRENT_GPUS=${VALUE_AS_TOOL_MAX_CONCURRENT_GPUS:-128}
# Full experiment submissions always use the fingerprinted config value. The
# VALUE_AS_TOOL_SOLVER_TENSOR_PARALLEL_SIZE override is reserved for manually
# launched compatibility canaries in run_gpu.sbatch.
readonly SOLVER_TENSOR_PARALLEL_SIZE=${config_values[5]}
readonly CONFIG_FINGERPRINT=${config_values[6]}
for shard_count in "$SOLVE_SHARDS" "$JUDGE_SHARDS"; do
  [[ $shard_count =~ ^[1-9][0-9]*$ ]] || {
    echo "shard counts must be positive integers, got: $shard_count" >&2
    exit 2
  }
done
unset shard_count
[[ $MAX_CONCURRENT_GPUS =~ ^[1-9][0-9]*$ ]] || {
  echo "VALUE_AS_TOOL_MAX_CONCURRENT_GPUS must be a positive integer" >&2
  exit 2
}
[[ $MAX_CONCURRENT_GPUS -le 128 ]] || {
  echo "VALUE_AS_TOOL_MAX_CONCURRENT_GPUS cannot exceed the 128-GPU safety cap" >&2
  exit 2
}
[[ $SOLVER_TENSOR_PARALLEL_SIZE =~ ^[1-9][0-9]*$ ]] || {
  echo "runtime.solver_tensor_parallel_size must be a positive integer" >&2
  exit 2
}
[[ $SOLVER_TENSOR_PARALLEL_SIZE -le $MAX_CONCURRENT_GPUS ]] || {
  echo "runtime.solver_tensor_parallel_size (${SOLVER_TENSOR_PARALLEL_SIZE}) exceeds the GPU cap (${MAX_CONCURRENT_GPUS})" >&2
  exit 2
}
readonly SOLVE_ARRAY_CONCURRENCY=$((MAX_CONCURRENT_GPUS / SOLVER_TENSOR_PARALLEL_SIZE))

vllm_bin=${VALUE_AS_TOOL_VLLM_BIN:-${REPO_ROOT}/.venv-vllm/bin/vllm}
if [[ ! -x $vllm_bin ]]; then
  vllm_bin=$(command -v vllm || true)
fi
[[ -n $vllm_bin && -x $vllm_bin ]] || {
  echo "missing vLLM executable; set VALUE_AS_TOOL_VLLM_BIN" >&2
  exit 2
}
export VALUE_AS_TOOL_VLLM_BIN=$vllm_bin
unset vllm_bin

if [[ $SOLVER_BACKEND == sglang ]]; then
  sglang_bin=${VALUE_AS_TOOL_SGLANG_BIN:-${REPO_ROOT}/.venv-sglang/bin/sglang}
  if [[ ! -x $sglang_bin ]]; then
    sglang_bin=$(command -v sglang || true)
  fi
  [[ -n $sglang_bin && -x $sglang_bin ]] || {
    echo "missing SGLang executable; set VALUE_AS_TOOL_SGLANG_BIN" >&2
    exit 2
  }
  export VALUE_AS_TOOL_SGLANG_BIN=$sglang_bin
  unset sglang_bin
fi
readonly SNAPSHOT_PARENT=${ARTIFACT_ROOT}/source_snapshots
mkdir -p "$SNAPSHOT_PARENT" "$ARTIFACT_ROOT/slurm" "$ASSET_ROOT"
exec 8>"${ARTIFACT_ROOT}/submission.lock"
flock 8

snapshot_staging=$(mktemp -d "${SNAPSHOT_PARENT}/.staging-XXXXXXXX")
cleanup_staging() {
  [[ -z ${snapshot_staging:-} || ! -d $snapshot_staging ]] || rm -rf -- "$snapshot_staging"
}
trap cleanup_staging EXIT

copy_items=()
for item in src scripts prompts tests pyproject.toml README.md uv.lock .gitignore; do
  [[ -e ${REPO_ROOT}/${item} ]] && copy_items+=("$item")
done
(cd "$REPO_ROOT" && tar -cf - "${copy_items[@]}") | tar -xf - -C "$snapshot_staging"
cp -- "$SOURCE_CONFIG" "$snapshot_staging/experiment.source.yaml"

source_digest=$(
  cd "$snapshot_staging"
  find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum
)
source_digest=${source_digest%% *}
readonly source_digest
readonly timestamp=$(date -u +%Y%m%dT%H%M%S.%NZ)
readonly SNAPSHOT=${SNAPSHOT_PARENT}/${timestamp}-${source_digest:0:12}
mv "$snapshot_staging" "$SNAPSHOT"
snapshot_staging=''
"$PYTHON_BIN" - "$SNAPSHOT" <<'PY'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True)
for path in [*paths, root]:
    if not path.is_symlink():
        os.chmod(path, path.stat().st_mode & ~0o222)
PY

git_commit='unavailable'
if command -v git >/dev/null && git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)
fi
manifest=${ARTIFACT_ROOT}/source-snapshot-${timestamp}.json
"$PYTHON_BIN" - "$manifest" "$SNAPSHOT" "$source_digest" "$CONFIG_FINGERPRINT" "$git_commit" <<'PY'
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

destination, snapshot, digest, config_fingerprint, commit = sys.argv[1:]
value = {
    "schema_version": 1,
    "created_at": datetime.now(UTC).isoformat(),
    "snapshot": snapshot,
    "source_sha256": digest,
    "experiment_fingerprint": config_fingerprint,
    "git_commit": commit,
    "credentials": "not copied; VALUE_AS_TOOL_ENV_FILE points to the live protected file",
}
temporary = Path(destination + ".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(destination)
PY

readonly SNAPSHOT_CONFIG=${SNAPSHOT}/experiment.source.yaml
readonly CPU_RUNNER=${SNAPSHOT}/scripts/slurm/run_cpu.sbatch
readonly GPU_RUNNER=${SNAPSHOT}/scripts/slurm/run_gpu.sbatch
readonly LOG_ROOT=${ARTIFACT_ROOT}/slurm
readonly QOS=${VALUE_AS_TOOL_SLURM_QOS:-g3_scientific-reasoning_high}
readonly PARTITION=${VALUE_AS_TOOL_SLURM_PARTITION:-g3}
readonly ACCOUNT=${VALUE_AS_TOOL_SLURM_ACCOUNT:-scientific-reasoning}
exports=VALUE_AS_TOOL_REPO_ROOT=${SNAPSHOT},VALUE_AS_TOOL_ENV_FILE=${ENV_FILE},VALUE_AS_TOOL_PYTHON=${PYTHON_BIN},VALUE_AS_TOOL_ARTIFACT_ROOT=${ARTIFACT_ROOT},VALUE_AS_TOOL_ASSET_ROOT=${ASSET_ROOT}
for executable_variable in \
  VALUE_AS_TOOL_SGLANG_BIN \
  VALUE_AS_TOOL_VLLM_BIN \
  VALUE_AS_TOOL_SERVER_STARTUP_TIMEOUT_SECONDS; do
  if [[ -n ${!executable_variable:-} ]]; then
    exports+=",${executable_variable}=${!executable_variable}"
  fi
done
readonly EXPORTS=$exports

submit() {
  sbatch --parsable --export="$EXPORTS" --partition="$PARTITION" --account="$ACCOUNT" \
    --qos="$QOS" "$@"
}

prepare_job=$(submit --job-name=vat-prepare \
  --output="${LOG_ROOT}/prepare-%j.out" --error="${LOG_ROOT}/prepare-%j.err" \
  "$CPU_RUNNER" "$SNAPSHOT_CONFIG" prepare)
prepare_job=${prepare_job%%;*}

solve_job=$(submit --job-name=vat-solve --dependency="afterok:${prepare_job}" \
  --gpus-per-node="$SOLVER_TENSOR_PARALLEL_SIZE" \
  --array="0-$((SOLVE_SHARDS - 1))%${SOLVE_ARRAY_CONCURRENCY}" \
  --output="${LOG_ROOT}/solve-%A_%a.out" --error="${LOG_ROOT}/solve-%A_%a.err" \
  "$GPU_RUNNER" "$SNAPSHOT_CONFIG" solve \
  --shard-count "$SOLVE_SHARDS")
solve_job=${solve_job%%;*}

# afterany intentionally allows judging/reporting partial schedules. Missing or
# failed cells remain explicit zeros and the resulting report is incomplete.
judge_job=$(submit --job-name=vat-judge --dependency="afterany:${solve_job}" \
  --gpus-per-node=1 \
  --array="0-$((JUDGE_SHARDS - 1))%${MAX_CONCURRENT_GPUS}" \
  --output="${LOG_ROOT}/judge-%A_%a.out" --error="${LOG_ROOT}/judge-%A_%a.err" \
  "$GPU_RUNNER" "$SNAPSHOT_CONFIG" judge \
  --shard-count "$JUDGE_SHARDS")
judge_job=${judge_job%%;*}

report_job=$(submit --job-name=vat-report --dependency="afterany:${judge_job}" \
  --output="${LOG_ROOT}/report-%j.out" --error="${LOG_ROOT}/report-%j.err" \
  "$CPU_RUNNER" "$SNAPSHOT_CONFIG" report)
report_job=${report_job%%;*}

printf 'snapshot=%s\nsource_sha256=%s\ngpu_concurrency_cap=%s\nsolver_tensor_parallel_size=%s\nsolve_array_concurrency=%s\nprepare=%s\nsolve=%s\njudge=%s\nreport=%s\n' \
  "$SNAPSHOT" "$source_digest" "$MAX_CONCURRENT_GPUS" \
  "$SOLVER_TENSOR_PARALLEL_SIZE" "$SOLVE_ARRAY_CONCURRENCY" "$prepare_job" \
  "$solve_job" "$judge_job" "$report_job"
