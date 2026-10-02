#!/usr/bin/env bash
#SBATCH --job-name=value-as-tool-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append

set -Eeuo pipefail
source "${VALUE_AS_TOOL_REPO_ROOT:?VALUE_AS_TOOL_REPO_ROOT is required}/scripts/slurm/common.sh" "$@"

cleanup_server() {
  if [[ -n ${server_pid:-} ]]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=''
  fi
}
trap cleanup_server EXIT

[[ $EXPERIMENT_STAGE == solve || $EXPERIMENT_STAGE == judge || $EXPERIMENT_STAGE == judge-nodes ]] || {
  echo "GPU runner supports only solve, judge, or judge-nodes" >&2
  exit 2
}
server_role=qwen
model_role=solver
server_tensor_parallel_size=$SOLVER_TENSOR_PARALLEL_SIZE
if [[ $EXPERIMENT_STAGE == judge || $EXPERIMENT_STAGE == judge-nodes ]]; then
  server_role=judge
  model_role=judge
  server_tensor_parallel_size=1
fi
readonly server_role model_role server_tensor_parallel_size
command -v nvidia-smi >/dev/null || {
  echo "nvidia-smi is unavailable in the GPU allocation" >&2
  exit 2
}
allocated_gpu_count=0
if [[ -n ${CUDA_VISIBLE_DEVICES:-} && $CUDA_VISIBLE_DEVICES != NoDevFiles ]]; then
  IFS=',' read -r -a allocated_gpu_ids <<<"$CUDA_VISIBLE_DEVICES"
  allocated_gpu_count=${#allocated_gpu_ids[@]}
  unset allocated_gpu_ids
else
  gpu_inventory=$(nvidia-smi -L 2>/dev/null || true)
  allocated_gpu_count=$(awk '/^GPU / { count++ } END { print count + 0 }' <<<"$gpu_inventory")
  unset gpu_inventory
fi
[[ $allocated_gpu_count -ge $server_tensor_parallel_size ]] || {
  echo "GPU allocation has ${allocated_gpu_count} devices; ${server_tensor_parallel_size} required" >&2
  exit 2
}
readonly allocated_gpu_count

# SGLang independently chooses an HTTP port and an internal torch.distributed
# TCPStore port. Serialize selection and retain one advisory lock per selected
# port for the server's entire lifetime, so co-located array cells cannot pick
# either port during the bind gap. The kernel probe also rejects ports already
# occupied by processes outside this launcher.
readonly PORT_LOCK_ROOT=/tmp/value-as-tool-ports-${UID}
mkdir -p "$PORT_LOCK_ROOT"
chmod 700 "$PORT_LOCK_ROOT"
exec {selection_fd}>"${PORT_LOCK_ROOT}/selection.lock"
flock "$selection_fd"

RESERVED_PORTS=()
RESERVED_PORT_FDS=()
reserve_tcp_port() {
  local attempt candidate candidate_fd
  for ((attempt = 1; attempt <= 256; attempt++)); do
    candidate=$(
      "$PYTHON_BIN" - <<'PY'
import secrets
import socket
from pathlib import Path

# Binding port 0 selects from the kernel's ephemeral client-port range. During
# a long model startup another process can legitimately claim that port for an
# outbound connection before the server binds it. Select a currently free port
# outside that range instead; the surrounding flock prevents our own colocated
# array tasks from choosing the same candidate.
try:
    low, high = map(
        int,
        Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split(),
    )
except (OSError, ValueError):
    low, high = 32768, 60999
ranges = []
if low > 10240:
    ranges.append((10240, low - 1))
if high < 65535:
    ranges.append((high + 1, 65535))
candidates = [
    port
    for start, stop in ranges
    for port in range(start, stop + 1)
]
for _ in range(min(512, len(candidates))):
    port = candidates.pop(secrets.randbelow(len(candidates)))
    try:
        with socket.socket() as sock:
            sock.bind(("0.0.0.0", port))
    except OSError:
        continue
    print(port)
    break
else:
    raise SystemExit("could not locate a free non-ephemeral TCP port")
PY
    )
    exec {candidate_fd}>"${PORT_LOCK_ROOT}/${candidate}.lock"
    if flock -n "$candidate_fd"; then
      RESERVED_PORTS+=("$candidate")
      RESERVED_PORT_FDS+=("$candidate_fd")
      return 0
    fi
    exec {candidate_fd}>&-
  done
  echo "could not reserve a free TCP port after 256 attempts" >&2
  return 1
}

reserve_tcp_port
if [[ $model_role == solver && $SOLVER_BACKEND == sglang ]]; then
  reserve_tcp_port
fi
flock -u "$selection_fd"
exec {selection_fd}>&-
readonly -a RESERVED_PORTS RESERVED_PORT_FDS
readonly SERVER_PORT=${RESERVED_PORTS[0]}

server_cli_args=()
if [[ $model_role == solver ]]; then
  export VALUE_AS_TOOL_SOLVER_BASE_URL=http://127.0.0.1:${SERVER_PORT}/v1
  if [[ $SOLVER_BACKEND == sglang ]]; then
    readonly SGLANG_NCCL_PORT=${RESERVED_PORTS[1]}
    server_cli_args+=(
      --extra-arg=--nccl-port
      "--extra-arg=${SGLANG_NCCL_PORT}"
    )
  fi
else
  export VALUE_AS_TOOL_JUDGE_BASE_URL=http://127.0.0.1:${SERVER_PORT}/v1
fi
readonly -a server_cli_args
readonly SERVER_LOG=${ARTIFACT_ROOT}/logs/server-${RUNTIME_KEY}.log
"$PYTHON_BIN" -m value_as_tool --config "$EXPERIMENT_CONFIG" serve "$server_role" \
  --tensor-parallel-size "$server_tensor_parallel_size" \
  "${server_cli_args[@]}" \
  >>"$SERVER_LOG" 2>&1 &
server_pid=$!

readonly SERVER_STARTUP_TIMEOUT_SECONDS=${VALUE_AS_TOOL_SERVER_STARTUP_TIMEOUT_SECONDS:-3600}
[[ $SERVER_STARTUP_TIMEOUT_SECONDS =~ ^[1-9][0-9]*$ ]] || {
  echo "VALUE_AS_TOOL_SERVER_STARTUP_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
}
readonly SERVER_STARTUP_POLLS=$(((SERVER_STARTUP_TIMEOUT_SECONDS + 1) / 2))
server_ready=0
for ((startup_poll = 0; startup_poll < SERVER_STARTUP_POLLS; startup_poll++)); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "model server exited during startup; see $SERVER_LOG" >&2
    wait "$server_pid"
    exit 1
  fi
  if "$PYTHON_BIN" - "$EXPERIMENT_CONFIG" "$model_role" <<'PY'
import os
import sys
from value_as_tool.config import load_config
from value_as_tool.server import endpoint_ready

config = load_config(sys.argv[1])
model = getattr(config.models, sys.argv[2])
base_url = config.models.operational_base_url(sys.argv[2])
raise SystemExit(
    0 if endpoint_ready(
        base_url,
        model.name,
        api_key=os.environ.get(model.api_key_env, "EMPTY"),
        timeout_seconds=2,
    ) else 1
)
PY
  then
    server_ready=1
    break
  fi
  sleep 2
done
[[ $server_ready == 1 ]] || { echo "model server startup timed out: $SERVER_LOG" >&2; exit 1; }

if [[ -n ${VALUE_AS_TOOL_GPU_COMMAND:-} ]]; then
  # A one-off job against the started server, e.g. a judge-agreement study.
  bash -c "$VALUE_AS_TOOL_GPU_COMMAND" &
elif [[ -n ${VALUE_AS_TOOL_ATTEMPT_MANIFEST:-} ]]; then
  "$PYTHON_BIN" "${REPO_ROOT}/scripts/attempt_conditioning_worker.py" \
    --config "$EXPERIMENT_CONFIG" --manifest "$VALUE_AS_TOOL_ATTEMPT_MANIFEST" \
    --stage "${VALUE_AS_TOOL_ATTEMPT_STAGE:?}" --queue "${VALUE_AS_TOOL_ATTEMPT_QUEUE:?}" &
else
  runtime_stage_args=("${STAGE_ARGS[@]}")
  if [[ -n ${SLURM_ARRAY_TASK_ID:-} ]]; then
    runtime_stage_args+=(--shard-index "$SLURM_ARRAY_TASK_ID")
  fi
  "$PYTHON_BIN" -m value_as_tool --config "$EXPERIMENT_CONFIG" \
    "$EXPERIMENT_STAGE" "${runtime_stage_args[@]}" &
fi
child_pid=$!
set +e
wait "$child_pid"
stage_status=$?
set -e
child_pid=''
kill -TERM "$server_pid" 2>/dev/null || true
wait "$server_pid" 2>/dev/null || true
server_pid=''
exit "$stage_status"
