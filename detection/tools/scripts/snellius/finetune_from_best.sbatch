#!/bin/bash
#
# Run 3 in one job: evaluate the pretraining run's saved checkpoints on the validation split, pick the one with the
# highest mean Level-1 AP (Vehicle, Pedestrian, Cyclist), and fine-tune on radar from it.
#
# Submit from the repository root, after (or waiting for) the pretraining job:
#
#   sbatch --dependency=afterany:<pretrain jobid> --export=ALL,DATA_ROOT=$HOME/data/dsert_roll \
#       detection/tools/scripts/snellius/finetune_from_best.sbatch
#
# afterany (not afterok): the checkpoints are what matter, so this also proceeds if only the pretraining job's own
# final evaluation failed. It stops if the final epoch's checkpoint is missing (pretraining did not finish).
#
# Settings: SOURCE_TAG (livox_pretrain), SOURCE_SENSOR (livox), TARGET_TAG (radar_ft_from_livox),
# TARGET_SENSOR (radar), REQUIRE_EPOCH (20). Training settings (AMP, COMPILE, PRETRAINED_SCOPE, ...) are passed through
# to train_single_sensor.sbatch. The choice is recorded in <source run>/best_checkpoint.json.
#
#SBATCH --job-name=radar_ft
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
export REPO_ROOT
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
SOURCE_TAG="${SOURCE_TAG:-livox_pretrain}"
SOURCE_SENSOR="${SOURCE_SENSOR:-livox}"
TARGET_TAG="${TARGET_TAG:-radar_ft_from_livox}"
TARGET_SENSOR="${TARGET_SENSOR:-radar}"
REQUIRE_EPOCH="${REQUIRE_EPOCH:-20}"
CFG="${CFG:-cfgs/det_model_cfgs/dsert/lidar_only.yaml}"

SNELLIUS_DIR="$REPO_ROOT/detection/tools/scripts/snellius"
SOURCE_DIR="$REPO_ROOT/detection/output/$(dirname "${CFG#cfgs/}")/$(basename "$CFG" .yaml)/$SOURCE_TAG"
[ -f "$SOURCE_DIR/ckpt/checkpoint_epoch_${REQUIRE_EPOCH}.pth" ] || {
    echo "ERROR: $SOURCE_DIR/ckpt/checkpoint_epoch_${REQUIRE_EPOCH}.pth missing: pretraining did not finish" >&2; exit 1; }
EVAL_LIST=$(ls "$SOURCE_DIR/ckpt/" | sed -n 's/^checkpoint_epoch_\([0-9]*\)\.pth$/\1/p' | sort -n | tr '\n' ' ')

# ---- 1) evaluate the pretraining checkpoints on val -------------------------------------------------------
echo "==> [1/3] evaluating $SOURCE_TAG epochs $EVAL_LIST on val ($SOURCE_SENSOR)"
DATA_ROOT="$DATA_ROOT" TAG="$SOURCE_TAG" SENSOR="$SOURCE_SENSOR" EVAL_EPOCHS="$EVAL_LIST" CFG="$CFG" \
    bash "$SNELLIUS_DIR/eval_epochs.sbatch"

# ---- 2) pick the best ---------------------------------------------------------------------------------------
echo "==> [2/3] selecting the best checkpoint"
# shellcheck disable=SC1091
source "$SNELLIUS_DIR/snellius_env.sh"
activate_env
# shellcheck disable=SC2086
BEST=$(python "$REPO_ROOT/detection/tools/select_best_checkpoint.py" --run_dir "$SOURCE_DIR" --epochs $EVAL_LIST | tee /dev/stderr | tail -1)
[ -f "$BEST" ] || { echo "ERROR: selection did not return a checkpoint" >&2; exit 1; }
echo "==> selected: $BEST"

# ---- 3) fine-tune on the target sensor from it -------------------------------------------------------------
echo "==> [3/3] fine-tuning $TARGET_TAG on $TARGET_SENSOR from $BEST"
unset EPOCHS EVAL_EPOCHS
DATA_ROOT="$DATA_ROOT" SENSOR="$TARGET_SENSOR" TAG="$TARGET_TAG" PRETRAINED="$BEST" CFG="$CFG" \
    bash "$SNELLIUS_DIR/train_single_sensor.sbatch"
