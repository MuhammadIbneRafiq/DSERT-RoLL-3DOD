#!/bin/bash
#
# Submit the VoxelNeXt experiments (cfgs/det_model_cfgs/dsert/voxelnext.yaml), one sbatch job each, reusing the same
# job scripts as the CenterPoint + LoGoHead runs:
#
#   vnx_livox                  LiDAR (Livox) from scratch      -> compare with the paper's VoxelNeXt L row
#   vnx_radar                  radar from scratch              -> compare with the paper's VoxelNeXt 4R row
#   vnx_radar_ft_from_livox    radar, initialised from vnx_livox (finetune_from_best.sbatch: waits for vnx_livox,
#                              evaluates its checkpoints, starts from the best one, as radar_ft_from_livox did)
#
# Run from the repository root on a login node:
#
#   DATA_ROOT=$HOME/data/dsert_roll bash detection/tools/scripts/snellius/submit_voxelnext.sh
#   SMOKE=1 DATA_ROOT=... bash detection/tools/scripts/snellius/submit_voxelnext.sh        # ~1% of data, 2 epochs
#   SEEDS="1 2 3" DATA_ROOT=... bash detection/tools/scripts/snellius/submit_voxelnext.sh  # 3 seeded runs per setup
#   ONLY="livox radar" ...                                                                  # a subset of the three
#
# Before the first full run, check the model once (1 GPU, < 1 h, no dataset; compares it with OpenPCDet's VoxelNeXt):
#   sbatch detection/tools/scripts/snellius/check_voxelnext.sbatch
#
# Other settings pass through to train_single_sensor.sbatch (AMP, PRETRAINED_SCOPE, INPUT_FEATURE, EXTRA_SETS, ...).
# PRETRAINED_SCOPE defaults to encoder (vfe + sparse backbone; head starts fresh), as in radar_ft_from_livox;
# PRETRAINED_SCOPE=all also copies the head.

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
SNELLIUS_DIR="$REPO_ROOT/detection/tools/scripts/snellius"
[ -f "$SNELLIUS_DIR/train_single_sensor.sbatch" ] || { echo "Run from the repository root" >&2; exit 1; }
: "${DATA_ROOT:?set DATA_ROOT to the DSERT-RoLL root}"
export DATA_ROOT REPO_ROOT

CFG=cfgs/det_model_cfgs/dsert/voxelnext.yaml
ONLY="${ONLY:-livox radar ft}"
SEEDS="${SEEDS:-}"          # empty = one unseeded run per setup
SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then TIME="${TIME:-02:00:00}"; else TIME="${TIME:-1-00:00:00}"; fi
FT_TIME="${FT_TIME:-1-12:00:00}"   # finetune job also evaluates every vnx_livox checkpoint first

want() { [[ " $ONLY " == *" $1 "* ]]; }

submit_seed() {
    local seed="$1" sfx="" smoke_sfx=""
    [ -n "$seed" ] && sfx="_s${seed}"
    [ "$SMOKE" = "1" ] && smoke_sfx="_smoke"   # train_single_sensor.sbatch appends _smoke to the tag itself
    local livox_tag="vnx_livox${sfx}" radar_tag="vnx_radar${sfx}" ft_tag="vnx_radar_ft_from_livox${sfx}"
    local livox_job=""

    if want livox; then
        livox_job=$(sbatch --parsable --time="$TIME" --job-name="$livox_tag" \
            --export=ALL,CFG="$CFG",SENSOR=livox,TAG="$livox_tag",SEED="$seed",SMOKE="$SMOKE" \
            "$SNELLIUS_DIR/train_single_sensor.sbatch")
        echo "submitted $livox_tag$smoke_sfx: job $livox_job"
    fi
    if want radar; then
        local j
        j=$(sbatch --parsable --time="$TIME" --job-name="$radar_tag" \
            --export=ALL,CFG="$CFG",SENSOR=radar,TAG="$radar_tag",SEED="$seed",SMOKE="$SMOKE" \
            "$SNELLIUS_DIR/train_single_sensor.sbatch")
        echo "submitted $radar_tag$smoke_sfx: job $j"
    fi
    if want ft; then
        if [ "$SMOKE" = "1" ]; then
            echo "skipped $ft_tag in SMOKE mode (finetune_from_best needs a full pretraining run)"
            return
        fi
        local dep=()
        [ -n "$livox_job" ] && dep=(--dependency=afterany:"$livox_job")
        local j
        j=$(sbatch --parsable --time="$FT_TIME" --job-name="$ft_tag" "${dep[@]}" \
            --export=ALL,CFG="$CFG",SOURCE_TAG="$livox_tag",SOURCE_SENSOR=livox,TARGET_TAG="$ft_tag",TARGET_SENSOR=radar,SEED="$seed" \
            "$SNELLIUS_DIR/finetune_from_best.sbatch")
        echo "submitted $ft_tag: job $j${livox_job:+ (after $livox_job)}"
    fi
}

if [ -z "$SEEDS" ]; then
    submit_seed ""
else
    for s in $SEEDS; do submit_seed "$s"; done
fi
echo "outputs: detection/output/det_model_cfgs/dsert/voxelnext/<tag>/"
