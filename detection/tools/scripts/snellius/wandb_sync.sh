#!/usr/bin/env bash
#
# Upload wandb runs that were logged offline (no internet on the compute node, or not logged in yet).
# Run on a Snellius login node, which has internet:
#
#   bash detection/tools/scripts/snellius/wandb_sync.sh                    # every offline run under detection/output
#   bash detection/tools/scripts/snellius/wandb_sync.sh <run output dir>   # just one run
#
# Safe to repeat: wandb skips data it has already uploaded.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ENV_DIR="${ENV_DIR:-$HOME/envs/dsert-lidar}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniforge3}"
SEARCH_DIR="${1:-$REPO_ROOT/detection/output}"

# shellcheck disable=SC1091
source "$CONDA_DIR/bin/activate" "$ENV_DIR"

if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
    echo "Not logged in to wandb. Run 'wandb login' first (key: https://wandb.ai/authorize)." >&2
    exit 1
fi

mapfile -t runs < <(find "$SEARCH_DIR" -type d -path "*/wandb/offline-run-*" -prune 2>/dev/null | sort)
if [ "${#runs[@]}" -eq 0 ]; then
    echo "No offline wandb runs under $SEARCH_DIR"
    exit 0
fi
echo "==> uploading ${#runs[@]} offline run(s)"
wandb sync "${runs[@]}"
