#!/bin/bash
#
# Evaluate every saved checkpoint of a finished (or running) run on the validation split, one after another, and
# print the overall AP block per epoch. Training keeps the last 5 checkpoints, so this covers epochs 16-20 of a
# 20-epoch run. Submit from the repository root:
#
#   sbatch --export=ALL,DATA_ROOT=$HOME/data/dsert_roll,TAG=livox_pretrain,SENSOR=livox detection/tools/scripts/snellius/eval_epochs.sbatch
#
# Optional: EVAL_EPOCHS="17 20" to evaluate only those epochs.
# Results: detection/output/det_model_cfgs/dsert/lidar_only/<TAG>/eval/epoch_<N>/val/log_eval_*.txt
#
#SBATCH --job-name=dsert-eval
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
TAG="${TAG:?set TAG to the run to evaluate, e.g. livox_pretrain}"
SENSOR="${SENSOR:?set SENSOR to the sensor the run was trained on (livox | radar)}"
CFG="${CFG:-cfgs/det_model_cfgs/dsert/lidar_only.yaml}"
EVAL_EPOCHS="${EVAL_EPOCHS:-}"

SNELLIUS_DIR="$REPO_ROOT/detection/tools/scripts/snellius"
# shellcheck disable=SC1091
source "$SNELLIUS_DIR/snellius_env.sh"
load_cuda_module
activate_env
export PYTHONWARNINGS='ignore:`torch.cuda.amp.custom_:FutureWarning'
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

RUN_DIR="$REPO_ROOT/detection/output/$(dirname "${CFG#cfgs/}")/$(basename "$CFG" .yaml)/$TAG"
cd "$REPO_ROOT/detection/tools"

if [ -z "$EVAL_EPOCHS" ]; then
    EVAL_EPOCHS=$(ls "$RUN_DIR/ckpt/" | sed -n 's/^checkpoint_epoch_\([0-9]*\)\.pth$/\1/p' | sort -n | tr '\n' ' ')
fi
[ -n "$EVAL_EPOCHS" ] || { echo "no checkpoints in $RUN_DIR/ckpt" >&2; exit 1; }
echo "==> $TAG ($SENSOR): evaluating epochs $EVAL_EPOCHS"

for e in $EVAL_EPOCHS; do
    echo "==> epoch $e"
    python test.py --cfg_file "$CFG" --ckpt "$RUN_DIR/ckpt/checkpoint_epoch_$e.pth" --extra_tag "$TAG" \
        --batch_size 16 --workers 14 \
        --set DATA_CONFIG.DATA_PATH "$DATA_ROOT" DATA_CONFIG.SENSOR.NAME "$SENSOR"
done

echo ""
echo "==================== $TAG: overall validation results per epoch ===================="
for e in $EVAL_EPOCHS; do
    log=$(ls -t "$RUN_DIR/eval/epoch_$e/val/"log_eval_*.txt 2>/dev/null | head -1 || true)
    echo "---- epoch $e  ($log)"
    if [ -n "$log" ]; then
        sed -n '/==== Overall ====/,/==== Per Weather ====/p' "$log" | grep -v "==== Per Weather" || true
    fi
done
