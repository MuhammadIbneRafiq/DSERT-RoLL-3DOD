#!/bin/bash
#
# Submit the TruckScenes VoxelNeXt experiments (cfgs/det_model_cfgs/truckscenes/voxelnext.yaml), mirroring the
# DSERT-RoLL VoxelNeXt runs (submit_voxelnext.sh), one train_truckscenes.sbatch job each:
#
#   vnx_ts_lidar                 the 6 LiDARs from scratch (= LiDAR pretraining)
#   vnx_ts_radar                 the 6 radars from scratch
#   vnx_ts_radar_ft_from_lidar   radar, initialised from vnx_ts_lidar's final checkpoint (epoch 20); waits for it.
#                                PRETRAINED_SCOPE=encoder (VFE + sparse backbone; head fresh), as on DSERT-RoLL.
#
# All are scored by the devkit's official evaluation (detection_cvpr_2024) on the official val split.
#
# Run from the repository root on a login node, after prepare_truckscenes.sbatch:
#
#   DATA_ROOT=/scratch-shared/$USER/truckscenes/man-truckscenes bash detection/tools/scripts/snellius/submit_truckscenes.sh
#   SMOKE=1 DATA_ROOT=... bash ...submit_truckscenes.sh          # every 50th sample, 2 epochs; FT from the smoke LiDAR run
#   SEEDS="1 2 3" DATA_ROOT=... bash ...submit_truckscenes.sh    # 3 seeded runs per setup
#   ONLY="lidar radar" ...                                       # a subset of: lidar radar ft
#
# Before the first full run, check the model once against OpenPCDet (1 GPU, no dataset; 7 heads with velocity):
#   CFG=cfgs/det_model_cfgs/truckscenes/voxelnext.yaml sbatch detection/tools/scripts/snellius/check_voxelnext.sbatch
#
# Other settings pass through to train_truckscenes.sbatch (BATCH, LR, AMP, INPUT_FEATURE, NSWEEPS, EXTRA_SETS, ...).
# TIME / FT_TIME set the job time limits.

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
SNELLIUS_DIR="$REPO_ROOT/detection/tools/scripts/snellius"
[ -f "$SNELLIUS_DIR/train_truckscenes.sbatch" ] || { echo "Run from the repository root" >&2; exit 1; }
: "${DATA_ROOT:?set DATA_ROOT to the TruckScenes dataroot}"
export DATA_ROOT REPO_ROOT

CFG=cfgs/det_model_cfgs/truckscenes/voxelnext.yaml
ONLY="${ONLY:-lidar radar ft}"
SEEDS="${SEEDS:-}"
SMOKE="${SMOKE:-0}"
EPOCHS_FULL=20                     # OPTIMIZATION.NUM_EPOCHS; the FT starts from this epoch's checkpoint
if [ "$SMOKE" = "1" ]; then
    TIME="${TIME:-03:00:00}"; FT_TIME="${FT_TIME:-$TIME}"; LAST_EPOCH=2
else
    TIME="${TIME:-1-00:00:00}"; FT_TIME="${FT_TIME:-1-00:00:00}"; LAST_EPOCH=$EPOCHS_FULL
fi
OUT_ROOT="$REPO_ROOT/detection/output/det_model_cfgs/truckscenes/voxelnext"

want() { [[ " $ONLY " == *" $1 "* ]]; }

submit_seed() {
    local seed="$1" sfx="" smoke_sfx=""
    [ -n "$seed" ] && sfx="_s${seed}"
    [ "$SMOKE" = "1" ] && smoke_sfx="_smoke"   # train_truckscenes.sbatch appends _smoke to the tag itself
    local lidar_tag="vnx_ts_lidar${sfx}" radar_tag="vnx_ts_radar${sfx}" ft_tag="vnx_ts_radar_ft_from_lidar${sfx}"
    local lidar_job="" j

    if want lidar; then
        lidar_job=$(sbatch --parsable --time="$TIME" --job-name="$lidar_tag" \
            --export=ALL,CFG="$CFG",SENSOR=lidar,TAG="$lidar_tag",SEED="$seed",SMOKE="$SMOKE" \
            "$SNELLIUS_DIR/train_truckscenes.sbatch")
        echo "submitted $lidar_tag$smoke_sfx: job $lidar_job"
    fi
    if want radar; then
        j=$(sbatch --parsable --time="$TIME" --job-name="$radar_tag" \
            --export=ALL,CFG="$CFG",SENSOR=radar,TAG="$radar_tag",SEED="$seed",SMOKE="$SMOKE" \
            "$SNELLIUS_DIR/train_truckscenes.sbatch")
        echo "submitted $radar_tag$smoke_sfx: job $j"
    fi
    if want ft; then
        local ckpt="$OUT_ROOT/${lidar_tag}${smoke_sfx}/ckpt/checkpoint_epoch_${LAST_EPOCH}.pth"
        local dep=()
        # afterok: fine-tune only from a LiDAR run that finished; without a LiDAR job now, the checkpoint must exist.
        if [ -n "$lidar_job" ]; then
            dep=(--dependency=afterok:"$lidar_job")
        elif [ ! -f "$ckpt" ]; then
            echo "skipped $ft_tag$smoke_sfx: $ckpt not found (submit with lidar in ONLY, or wait for that run)" >&2
            return
        fi
        j=$(sbatch --parsable --time="$FT_TIME" --job-name="$ft_tag" "${dep[@]}" \
            --export=ALL,CFG="$CFG",SENSOR=radar,TAG="$ft_tag",SEED="$seed",SMOKE="$SMOKE",PRETRAINED="$ckpt" \
            "$SNELLIUS_DIR/train_truckscenes.sbatch")
        echo "submitted $ft_tag$smoke_sfx: job $j${lidar_job:+ (after $lidar_job)}, from $ckpt"
    fi
}

if [ -z "$SEEDS" ]; then
    submit_seed ""
else
    for s in $SEEDS; do submit_seed "$s"; done
fi
echo "outputs: $OUT_ROOT/<tag>/"
