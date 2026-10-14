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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
SEARCH_DIR="${1:-$REPO_ROOT/detection/output}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/snellius_env.sh"
activate_env

if ! wandb_logged_in; then
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
