#!/usr/bin/env bash
set -Eeuo pipefail

exec "$(dirname "${BASH_SOURCE[0]}")/submit_slurm.sh" "$@"
