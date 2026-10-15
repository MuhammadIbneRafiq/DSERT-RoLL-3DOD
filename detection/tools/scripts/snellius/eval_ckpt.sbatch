#!/bin/bash
#
# Evaluate one checkpoint on the validation split with any sensor as input, on one H100, and log the results
# (overall / per weather / per light AP) to Weights & Biases as their own run. Submit from the repository root:
#
#   # Livox-pretrained model fed radar, no fine-tuning (how big is the LiDAR -> radar gap?)
#   sbatch --export=ALL,DATA_ROOT=$HOME/data/dsert_roll,SENSOR=radar,NAME=livox_e20_on_radar,\
#   CKPT=$HOME/DSERT-RoLL-3DOD/detection/output/det_model_cfgs/dsert/lidar_only/livox_pretrain/ckpt/checkpoint_epoch_20.pth \
#       detection/tools/scripts/snellius/eval_ckpt.sbatch
#
# NAME is the wandb run name and the output folder: detection/output/det_model_cfgs/dsert/lidar_only/<NAME>/eval/.
#
#SBATCH --job-name=dsert-evalckpt
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
CKPT="${CKPT:?set CKPT to the checkpoint to evaluate}"
SENSOR="${SENSOR:?set SENSOR to the input sensor (livox | radar)}"
NAME="${NAME:?set NAME, e.g. livox_e20_on_radar}"
CFG="${CFG:-cfgs/det_model_cfgs/dsert/lidar_only.yaml}"
WANDB_PROJECT="${WANDB_PROJECT-dsert-single-sensor}"

SNELLIUS_DIR="$REPO_ROOT/detection/tools/scripts/snellius"
# shellcheck disable=SC1091
source "$SNELLIUS_DIR/snellius_env.sh"
load_cuda_module
activate_env
export PYTHONWARNINGS='ignore:`torch.cuda.amp.custom_:FutureWarning'
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1
if [ -n "$WANDB_PROJECT" ] && ! wandb_logged_in; then
    echo "WARNING: wandb is not logged in; logging offline (upload later with wandb_sync.sh)." >&2
    export WANDB_MODE=offline
fi

[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$REPO_ROOT/detection/tools"

WANDB_ARGS=()
[ -n "$WANDB_PROJECT" ] && WANDB_ARGS=(--wandb_project "$WANDB_PROJECT")
python test.py --cfg_file "$CFG" --ckpt "$CKPT" --extra_tag "$NAME" --batch_size 16 \
    --workers $(( ${SLURM_CPUS_PER_TASK:-16} - 2 )) "${WANDB_ARGS[@]}" \
    --set DATA_CONFIG.DATA_PATH "$DATA_ROOT" DATA_CONFIG.SENSOR.NAME "$SENSOR"

OUT_DIR="$REPO_ROOT/detection/output/$(dirname "${CFG#cfgs/}")/$(basename "$CFG" .yaml)/$NAME"
LOG=$(ls -t "$OUT_DIR"/eval/epoch_*/val/log_eval_*.txt | head -1)
echo ""
echo "==================== $NAME: $(basename "$CKPT") evaluated on $SENSOR ===================="
sed -n '/==== Overall ====/,/==== Per Light ====/p' "$LOG" | grep -E "====|N=|LEVEL_1/AP:" || true
echo "full log: $LOG"
