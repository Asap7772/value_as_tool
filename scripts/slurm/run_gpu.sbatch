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

[[ $EXPERIMENT_STAGE == solve || $EXPERIMENT_STAGE == judge ]] || {
  echo "GPU runner supports only solve or judge" >&2
  exit 2
}
command -v nvidia-smi >/dev/null && [[ $(nvidia-smi -L | wc -l) -ge 1 ]] || {
  echo "GPU allocation is missing" >&2
  exit 2
}

# A per-user advisory lock closes the gap between selecting an unused port and
# the model server binding it when multiple array cells share a host.
readonly PORT_LOCK_ROOT=/tmp/value-as-tool-ports-${UID}
mkdir -p "$PORT_LOCK_ROOT"
chmod 700 "$PORT_LOCK_ROOT"
exec {selection_fd}>"${PORT_LOCK_ROOT}/selection.lock"
flock "$selection_fd"
SERVER_PORT=$(
  "$PYTHON_BIN" - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)
readonly SERVER_PORT
exec {port_fd}>"${PORT_LOCK_ROOT}/${SERVER_PORT}.lock"
flock -n "$port_fd" || { echo "could not reserve selected port $SERVER_PORT" >&2; exit 3; }
flock -u "$selection_fd"
exec {selection_fd}>&-
server_role=qwen
model_role=solver
if [[ $EXPERIMENT_STAGE == judge ]]; then
  server_role=judge
  model_role=judge
fi
readonly server_role model_role
if [[ $model_role == solver ]]; then
  export VALUE_AS_TOOL_SOLVER_BASE_URL=http://127.0.0.1:${SERVER_PORT}/v1
else
  export VALUE_AS_TOOL_JUDGE_BASE_URL=http://127.0.0.1:${SERVER_PORT}/v1
fi
readonly SERVER_LOG=${ARTIFACT_ROOT}/logs/server-${RUNTIME_KEY}.log
"$PYTHON_BIN" -m value_as_tool --config "$EXPERIMENT_CONFIG" serve "$server_role" \
  >>"$SERVER_LOG" 2>&1 &
server_pid=$!

server_ready=0
for _ in {1..450}; do
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

runtime_stage_args=("${STAGE_ARGS[@]}")
if [[ -n ${SLURM_ARRAY_TASK_ID:-} ]]; then
  runtime_stage_args+=(--shard-index "$SLURM_ARRAY_TASK_ID")
fi
"$PYTHON_BIN" -m value_as_tool --config "$EXPERIMENT_CONFIG" \
  "$EXPERIMENT_STAGE" "${runtime_stage_args[@]}" &
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
