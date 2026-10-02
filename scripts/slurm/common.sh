#!/usr/bin/env bash
# Shared Slurm runtime setup. This file is sourced by the stage runners.

set -Eeuo pipefail
umask 077

[[ $# -ge 2 ]] || { echo "usage: <runner> CONFIG STAGE [ARGS ...]" >&2; exit 2; }

readonly EXPERIMENT_CONFIG=$(readlink -f -- "$1")
readonly EXPERIMENT_STAGE=$2
shift 2
readonly STAGE_ARGS=("$@")
readonly REPO_ROOT=${VALUE_AS_TOOL_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}
default_python=${REPO_ROOT}/.venv/bin/python
if [[ ! -x $default_python ]]; then
  default_python=$(command -v python3.12 || command -v python3)
fi
readonly PYTHON_BIN=${VALUE_AS_TOOL_PYTHON:-$default_python}
unset default_python
readonly ENV_FILE=${VALUE_AS_TOOL_ENV_FILE:-${REPO_ROOT}/.env}

[[ -r $EXPERIMENT_CONFIG ]] || { echo "missing config: $EXPERIMENT_CONFIG" >&2; exit 2; }
[[ -x $PYTHON_BIN ]] || { echo "Python 3.12 is required: $PYTHON_BIN" >&2; exit 2; }

# Slurm starts batch scripts with a sanitized PATH. The model server entry
# points are absolute, but both SGLang and vLLM invoke sibling build tools such
# as ninja by name while compiling first-run kernels.
if [[ $EXPERIMENT_STAGE == solve || $EXPERIMENT_STAGE == judge || $EXPERIMENT_STAGE == judge-nodes ]]; then
  for server_binary_variable in VALUE_AS_TOOL_SGLANG_BIN VALUE_AS_TOOL_VLLM_BIN; do
    server_binary=${!server_binary_variable:-}
    if [[ -n $server_binary ]]; then
      PATH=$(dirname -- "$server_binary"):${PATH}
    fi
  done
  export PATH
  unset server_binary server_binary_variable
  command -v ninja >/dev/null || {
    echo "missing ninja executable beside the configured model server binaries" >&2
    exit 2
  }
fi

export PYTHONPATH=${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONDONTWRITEBYTECODE=1

# Read only the three known credentials. The .env file is never sourced or
# copied into the immutable source snapshot.
read_env_value() {
  local key=$1 raw=''
  [[ -r $ENV_FILE ]] || return 0
  raw=$(awk -v key="$key" '
    $0 ~ "^[[:space:]]*(export[[:space:]]+)?" key "[[:space:]]*=" {
      sub("^[[:space:]]*(export[[:space:]]+)?" key "[[:space:]]*=[[:space:]]*", "")
      sub(/\r$/, "")
      print
      exit
    }
  ' "$ENV_FILE")
  if [[ ${#raw} -ge 2 ]] && {
    [[ ${raw:0:1} == '"' && ${raw: -1} == '"' ]] \
      || [[ ${raw:0:1} == "'" && ${raw: -1} == "'" ]];
  }; then
    raw=${raw:1:${#raw}-2}
  fi
  printf '%s' "$raw"
}

for secret_name in MODEL_API_KEY LLAMA_API_KEY HF_TOKEN; do
  if [[ -z ${!secret_name:-} ]]; then
    secret_value=$(read_env_value "$secret_name")
    if [[ -n $secret_value ]]; then
      printf -v "$secret_name" '%s' "$secret_value"
      export "$secret_name"
    fi
  fi
done
unset secret_name secret_value

readarray -t configured_paths < <(
  "$PYTHON_BIN" - "$EXPERIMENT_CONFIG" <<'PY'
import sys
from pathlib import Path
from value_as_tool.pipeline import load_context

source = Path(sys.argv[1]).resolve()
context = load_context(source)
print(context.artifact_root)
print(context.asset_root)
print(context.config.runtime.solver_backend)
print(getattr(context.config.runtime, "solver_tensor_parallel_size", 1))
PY
)
readonly ARTIFACT_ROOT=${configured_paths[0]}
readonly ASSET_ROOT=${configured_paths[1]}
readonly SOLVER_BACKEND=${configured_paths[2]}
readonly CONFIGURED_SOLVER_TENSOR_PARALLEL_SIZE=${configured_paths[3]}
readonly SOLVER_TENSOR_PARALLEL_SIZE=${VALUE_AS_TOOL_SOLVER_TENSOR_PARALLEL_SIZE:-${CONFIGURED_SOLVER_TENSOR_PARALLEL_SIZE}}
[[ $SOLVER_TENSOR_PARALLEL_SIZE =~ ^[1-9][0-9]*$ ]] || {
  echo "solver tensor-parallel size must be a positive integer" >&2
  exit 2
}
readonly RUNTIME_KEY=${SLURM_JOB_ID:-manual}-${SLURM_ARRAY_TASK_ID:-0}-${EXPERIMENT_STAGE}
readonly LOCAL_RUNTIME_ROOT=/tmp/value-as-tool-${UID}/${RUNTIME_KEY}

export HF_HOME=${ASSET_ROOT}/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${ASSET_ROOT}/datasets
export TRANSFORMERS_CACHE=${ASSET_ROOT}/transformers
export TORCH_HOME=${ASSET_ROOT}/torch
export UV_CACHE_DIR=${ASSET_ROOT}/uv
export TRITON_CACHE_DIR=${LOCAL_RUNTIME_ROOT}/triton
export SGLANG_CACHE_DIR=${LOCAL_RUNTIME_ROOT}/sglang
export XDG_CACHE_HOME=${LOCAL_RUNTIME_ROOT}/xdg
export TMPDIR=${LOCAL_RUNTIME_ROOT}/tmp
export VLLM_RPC_BASE_PATH=${LOCAL_RUNTIME_ROOT}/vllm-rpc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$ARTIFACT_ROOT/logs" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" \
    "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$UV_CACHE_DIR" "$TRITON_CACHE_DIR" \
    "$SGLANG_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR" "$VLLM_RPC_BASE_PATH"
chmod 700 "$LOCAL_RUNTIME_ROOT" "$TRITON_CACHE_DIR" "$SGLANG_CACHE_DIR" \
  "$XDG_CACHE_HOME" "$TMPDIR" "$VLLM_RPC_BASE_PATH"

# Prevent duplicate execution of the same array cell while still permitting
# different stages and shard IDs to run concurrently.
lock_material=$(printf '%s\0' "$EXPERIMENT_CONFIG" "$EXPERIMENT_STAGE" \
  "${STAGE_ARGS[@]}" "${SLURM_ARRAY_TASK_ID:-0}" | sha256sum)
lock_material=${lock_material%% *}
exec 9>"${ARTIFACT_ROOT}/stage-${lock_material:0:20}.lock"
flock -n 9 || { echo "an identical stage is already active" >&2; exit 3; }

child_pid=''
server_pid=''
stop_children() {
  local signal_name=${1:-TERM}
  [[ -z $child_pid ]] || kill -"$signal_name" "$child_pid" 2>/dev/null || true
  [[ -z $server_pid ]] || kill -"$signal_name" "$server_pid" 2>/dev/null || true
}
on_term() {
  stop_children TERM
}
on_requeue() {
  trap - USR1
  echo "wall-time signal received; stopping at the durable checkpoint boundary" >&2
  stop_children TERM
  [[ -z $child_pid ]] || wait "$child_pid" 2>/dev/null || true
  [[ -z $server_pid ]] || wait "$server_pid" 2>/dev/null || true
  if [[ -n ${SLURM_JOB_ID:-} ]] && command -v scontrol >/dev/null; then
    scontrol requeue "$SLURM_JOB_ID"
  fi
  exit 0
}
trap on_term TERM INT
trap on_requeue USR1
