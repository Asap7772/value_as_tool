#!/usr/bin/env bash
#SBATCH --job-name=value-as-tool-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append

set -Eeuo pipefail
source "${VALUE_AS_TOOL_REPO_ROOT:?VALUE_AS_TOOL_REPO_ROOT is required}/scripts/slurm/common.sh" "$@"

"$PYTHON_BIN" -m value_as_tool --config "$EXPERIMENT_CONFIG" \
  "$EXPERIMENT_STAGE" "${STAGE_ARGS[@]}" &
child_pid=$!
wait "$child_pid"
child_pid=''
