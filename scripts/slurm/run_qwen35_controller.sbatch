#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --open-mode=append

set -Eeuo pipefail
readonly controller_config=$1
readonly controller_mode=$2
readonly controller_manifest=$3
source "${VALUE_AS_TOOL_REPO_ROOT:?}/scripts/slurm/common.sh" \
  "$controller_config" report "$controller_mode"
exec "$PYTHON_BIN" "${REPO_ROOT}/scripts/submit_qwen35_large_budget.py" \
  "$controller_mode" --manifest "$controller_manifest"
