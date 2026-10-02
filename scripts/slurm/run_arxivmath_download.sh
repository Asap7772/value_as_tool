#!/usr/bin/env bash
# Download the pinned ArXivMath outputs dataset where huggingface.co is reachable:
#   srun -p cpu_x86 --cpus-per-task=4 --mem=16G --time=01:00:00 \
#     scripts/slurm/run_arxivmath_download.sh
# HF_TOKEN is read from the repo's .env by prepare_arxivmath_dataset.py.
set -Eeuo pipefail
# The submitting shell's sandbox proxy and aarch64 PATH entries do not apply on x86 nodes.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy ALL_PROXY all_proxy
export PATH=/usr/local/bin:/usr/bin:/bin
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
PYTHONPATH=${VALUE_AS_TOOL_X86_PYDEPS:-/checkpoint/fort/$USER/value_as_tool_rollout_viewer/work/pydeps_x86} \
  /usr/bin/python3 "$repo/scripts/prepare_arxivmath_dataset.py" download "$@"
