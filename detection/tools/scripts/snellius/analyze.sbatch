#!/bin/bash
#
# GPU-only analysis in one job (1 H100, roughly 30-45 min). Only work that needs the GPU runs here:
#   1. zero-shot evaluation: the Livox-pretrained checkpoint fed radar, no fine-tuning (AP, logged to wandb)
#   2. encoder features on the full validation split for: radar_scratch, radar_ft_from_livox, livox_e20_on_radar
#      (Livox model fed radar) and livox_pretrain (Livox model fed Livox, as reference)
#   3. BEV Grad-CAM images for the radar-input models, a few frames per weather
# Everything CPU-side (t-SNE, probe metrics, failure_stats.py, plots, wandb upload of the folder) runs afterwards on
# Kaggle / a laptop from the .npz files (see analysis/transfer_diagnosis.ipynb), or for the upload on a login node:
#   python detection/tools/analysis/log_to_wandb.py --dir <OUT> --project dsert-single-sensor --name analysis_<date>
#
# Submit from the repository root:
#
#   sbatch --export=ALL,DATA_ROOT=$HOME/data/dsert_roll detection/tools/scripts/snellius/analyze.sbatch
#
#SBATCH --job-name=dsert-analyze
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
export REPO_ROOT
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
RUNS_DIR="$REPO_ROOT/detection/output/det_model_cfgs/dsert/lidar_only"
LIDAR_CKPT="${LIDAR_CKPT:-$RUNS_DIR/livox_pretrain/ckpt/checkpoint_epoch_20.pth}"
SCRATCH_CKPT="${SCRATCH_CKPT:-$RUNS_DIR/radar_scratch/ckpt/checkpoint_epoch_20.pth}"
FT_CKPT="${FT_CKPT:-$RUNS_DIR/radar_ft_from_livox/ckpt/checkpoint_epoch_20.pth}"
OUT="${OUT:-$HOME/analysis/$(date +%Y%m%d_%H%M)}"
MAX_FRAMES="${MAX_FRAMES:-100000}"                                  # default: the whole validation split
GRADCAM_WEATHERS="${GRADCAM_WEATHERS:-Clear Fog Light_Rain Heavy_Rain Light_Snow Heavy_Snow}"
GRADCAM_FRAMES="${GRADCAM_FRAMES:-3}"                               # frames per weather and model
ZERO_SHOT_EVAL="${ZERO_SHOT_EVAL:-1}"                               # 0 = skip step 1 (its AP is already on wandb)
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
    echo "WARNING: wandb is not logged in; logging offline." >&2
    export WANDB_MODE=offline
fi
for c in "$LIDAR_CKPT" "$SCRATCH_CKPT" "$FT_CKPT"; do
    [ -f "$c" ] || { echo "ERROR: checkpoint not found: $c" >&2; exit 1; }
done
mkdir -p "$OUT"
A="$REPO_ROOT/detection/tools/analysis"
echo "==> outputs: $OUT"

# ---- 1) zero-shot evaluation (AP), logged to wandb as lidar_only/livox_e20_on_radar ------------------------
if [ "$ZERO_SHOT_EVAL" = "1" ]; then
    echo "==> [1/3] Livox epoch-20 model evaluated on radar, no fine-tuning"
    DATA_ROOT="$DATA_ROOT" CKPT="$LIDAR_CKPT" SENSOR=radar NAME=livox_e20_on_radar WANDB_PROJECT="$WANDB_PROJECT" \
        bash "$SNELLIUS_DIR/eval_ckpt.sbatch"
fi

# ---- 2) encoder features (name : checkpoint : input sensor) -------------------------------------------------
MODELS="radar_scratch:$SCRATCH_CKPT:radar radar_ft_from_livox:$FT_CKPT:radar livox_e20_on_radar:$LIDAR_CKPT:radar livox_pretrain:$LIDAR_CKPT:livox"
for spec in $MODELS; do
    IFS=: read -r name ckpt sensor <<< "$spec"
    echo "==> [2/3] features: $name ($sensor)"
    python "$A/extract_features.py" --ckpt "$ckpt" --sensor "$sensor" --data_root "$DATA_ROOT" --name "$name" \
        --out "$OUT/feats_$name.npz" --max_frames "$MAX_FRAMES"
done

# ---- 3) Grad-CAM for the radar-input models, per weather ---------------------------------------------------
for spec in $MODELS; do
    IFS=: read -r name ckpt sensor <<< "$spec"
    [ "$sensor" = radar ] || continue
    for w in $GRADCAM_WEATHERS; do
        echo "==> [3/3] Grad-CAM: $name, $w"
        python "$A/gradcam_bev.py" --ckpt "$ckpt" --sensor "$sensor" --data_root "$DATA_ROOT" --name "$name" \
            --out_dir "$OUT/gradcam/$w" --weather "$w" --max_frames "$GRADCAM_FRAMES" || echo "   (no frames for $w)"
    done
done

echo ""
echo "==> done: $OUT ($(du -sh "$OUT" | cut -f1))"
