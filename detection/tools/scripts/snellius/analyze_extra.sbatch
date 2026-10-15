#!/bin/bash
#
# Extras for an existing analysis folder (from analyze.sbatch), ~10 min on 1 H100:
#   1. features of an untrained, randomly initialised encoder (seed 0) on radar and on Livox: the floor for the probe
#      metrics, i.e. how separable the classes are from geometry alone, before any learning
#   2. Grad-CAM of livox_pretrain on Livox (where the LiDAR model looks when it works well), per weather
#   3. probe metrics + recall tables recomputed with all models, upload to wandb
#
#   sbatch --export=ALL,DATA_ROOT=$HOME/data/dsert_roll,OUT=$HOME/analysis/<date_time> \
#       detection/tools/scripts/snellius/analyze_extra.sbatch
#
#SBATCH --job-name=dsert-analyze-extra
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:59:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
OUT="${OUT:?set OUT to the analysis folder written by analyze.sbatch, e.g. \$HOME/analysis/20261015_1950}"
LIDAR_CKPT="${LIDAR_CKPT:-$REPO_ROOT/detection/output/det_model_cfgs/dsert/lidar_only/livox_pretrain/ckpt/checkpoint_epoch_20.pth}"
MAX_FRAMES="${MAX_FRAMES:-100000}"
GRADCAM_WEATHERS="${GRADCAM_WEATHERS:-Clear Fog Light_Rain Heavy_Rain Light_Snow Heavy_Snow}"
GRADCAM_FRAMES="${GRADCAM_FRAMES:-3}"
WANDB_PROJECT="${WANDB_PROJECT-dsert-single-sensor}"

SNELLIUS_DIR="$REPO_ROOT/detection/tools/scripts/snellius"
# shellcheck disable=SC1091
source "$SNELLIUS_DIR/snellius_env.sh"
load_cuda_module
activate_env
export PYTHONWARNINGS='ignore:`torch.cuda.amp.custom_:FutureWarning'
export PYTHONUNBUFFERED=1
if [ -n "$WANDB_PROJECT" ] && ! wandb_logged_in; then export WANDB_MODE=offline; fi
[ -d "$OUT" ] || { echo "ERROR: $OUT does not exist" >&2; exit 1; }
A="$REPO_ROOT/detection/tools/analysis"

# ---- 1) random-init encoder features -------------------------------------------------------------------------
for sensor in radar livox; do
    echo "==> [1/3] features: random_init_on_$sensor"
    python "$A/extract_features.py" --ckpt random --sensor "$sensor" --data_root "$DATA_ROOT" \
        --name "random_init_on_$sensor" --out "$OUT/feats_random_init_on_$sensor.npz" --max_frames "$MAX_FRAMES"
done

# ---- 2) Grad-CAM of the LiDAR model on LiDAR -------------------------------------------------------------------
for w in $GRADCAM_WEATHERS; do
    echo "==> [2/3] Grad-CAM: livox_pretrain on livox, $w"
    python "$A/gradcam_bev.py" --ckpt "$LIDAR_CKPT" --sensor livox --data_root "$DATA_ROOT" --name livox_pretrain \
        --out_dir "$OUT/gradcam/$w" --weather "$w" --max_frames "$GRADCAM_FRAMES" || echo "   (no frames for $w)"
done

# ---- 3) metrics with all models, upload ------------------------------------------------------------------------
echo "==> [3/3] probe metrics + recall, all models"
python "$A/plot_tsne.py" --features "$OUT"/feats_*.npz --out_dir "$OUT" --no_tsne
if [ -n "$WANDB_PROJECT" ]; then
    python "$A/log_to_wandb.py" --dir "$OUT" --project "$WANDB_PROJECT" --name "analysis_$(basename "$OUT")_full" \
        || echo "WARNING: wandb upload failed; results are still in $OUT"
fi
echo "==> done: $OUT"
