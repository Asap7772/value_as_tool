#!/usr/bin/env bash
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
[[ $# == 4 ]] || { echo 'usage: CONFIG STAGE MANIFEST QUEUE' >&2; exit 2; }
readonly attempt_config=$1
export VALUE_AS_TOOL_ATTEMPT_STAGE=$2
export VALUE_AS_TOOL_ATTEMPT_MANIFEST=$3
export VALUE_AS_TOOL_ATTEMPT_QUEUE=$4
role=solve
[[ $VALUE_AS_TOOL_ATTEMPT_STAGE != *judge ]] || role=judge
exec bash "${VALUE_AS_TOOL_REPO_ROOT:?}/scripts/slurm/run_gpu.sbatch" \
  "$attempt_config" "$role" --attempt-queue "$VALUE_AS_TOOL_ATTEMPT_QUEUE" \
  --attempt-worker "${SLURM_JOB_ID:-manual}"
