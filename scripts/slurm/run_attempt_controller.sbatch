#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
set -Eeuo pipefail
[[ $# == 2 ]] || { echo 'usage: PREPROCESS_CONFIG MANIFEST' >&2; exit 2; }
readonly attempt_manifest=$2
source "${VALUE_AS_TOOL_REPO_ROOT:?}/scripts/slurm/common.sh" "$1" report \
  --attempt-controller "$attempt_manifest"
"$PYTHON_BIN" "${REPO_ROOT}/scripts/submit_attempt_conditioning.py" supervise \
  --manifest "$attempt_manifest" &
child_pid=$!
set +e
wait "$child_pid"
status=$?
set -e
child_pid=''
exit "$status"
