#!/bin/bash
#
# The reverse of finetune_from_best.sbatch: pretrain on radar, fine-tune on LiDAR, and evaluate before and after.
# One job, five steps. Every step is skipped if its result is already on disk, so re-submitting the same command
# after a time-out (or with a radar run that already finished) continues where it stopped.
#
#   1) radar pretraining            -> <SOURCE_TAG>/ckpt/                  skipped if checkpoint_epoch_<REQUIRE_EPOCH>.pth exists
#   2) evaluate its checkpoints on radar, pick the best (mean Level-1 AP)  -> <SOURCE_TAG>/best_checkpoint.json
#   3) BEFORE fine-tuning: that radar checkpoint evaluated on LiDAR as is  -> <SOURCE_SENSOR>_e<N>_on_<TARGET_SENSOR>/eval/
#   4) fine-tune on LiDAR from that checkpoint                             -> <TARGET_TAG>/ckpt/
#   5) AFTER fine-tuning: evaluate its checkpoints on LiDAR, pick the best -> <TARGET_TAG>/best_checkpoint.json
#
# and a three-line summary at the end of the slurm-*.out file. All folders are under
# detection/output/det_model_cfgs/dsert/lidar_only/.
#
# Submit from the repository root:
#
#   sbatch --export=ALL,DATA_ROOT=$HOME/data/dsert_roll detection/tools/scripts/snellius/radar_to_lidar.sbatch
#
#   # a radar run that already finished under another tag (step 1 is skipped, so a shorter limit is enough)
#   sbatch --time=16:00:00 --export=ALL,DATA_ROOT=$HOME/data/dsert_roll,SOURCE_TAG=<tag> detection/tools/scripts/snellius/radar_to_lidar.sbatch
#
#   # extra fine-tuning seeds (works in either direction; steps 1-3 are already on disk, so only 4-5 run).
#   # SEED=1 goes to <TARGET_SENSOR>_ft_from_<SOURCE_SENSOR>_s1; the unseeded first run keeps its old tag.
#   sbatch --time=16:00:00 --job-name=ft_radar2livox_s1 --export=ALL,DATA_ROOT=$HOME/data/dsert_roll,SEED=1 detection/tools/scripts/snellius/radar_to_lidar.sbatch
#   sbatch --time=16:00:00 --job-name=ft_livox2radar_s1 --export=ALL,DATA_ROOT=$HOME/data/dsert_roll,SEED=1,SOURCE_TAG=livox_pretrain,SOURCE_SENSOR=livox,TARGET_SENSOR=radar detection/tools/scripts/snellius/radar_to_lidar.sbatch
#
# Settings: SOURCE_TAG (radar_scratch), SOURCE_SENSOR (radar), TARGET_SENSOR (livox), TARGET_TAG
# (<TARGET_SENSOR>_ft_from_<SOURCE_SENSOR>[_s<SEED>]), SEED (unset; fine-tuning only), REQUIRE_EPOCH (20). Training
# settings (AMP, COMPILE, PRETRAINED_SCOPE, WANDB_PROJECT, ...) are passed through to train_single_sensor.sbatch.
#
# Time: two 20-epoch trainings plus 11 evaluations. Snellius reserves (time limit x 192 SBU/h) when the job starts,
# so lower --time when step 1 is already done.
#
#SBATCH --job-name=radar2lidar
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint=scratch-node
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
export REPO_ROOT
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
SOURCE_TAG="${SOURCE_TAG:-radar_scratch}"
SOURCE_SENSOR="${SOURCE_SENSOR:-radar}"
TARGET_SENSOR="${TARGET_SENSOR:-livox}"
TARGET_TAG="${TARGET_TAG:-${TARGET_SENSOR}_ft_from_${SOURCE_SENSOR}${SEED:+_s$SEED}}"
REQUIRE_EPOCH="${REQUIRE_EPOCH:-20}"
CFG="${CFG:-cfgs/det_model_cfgs/dsert/lidar_only.yaml}"

SNELLIUS_DIR="$REPO_ROOT/detection/tools/scripts/snellius"
[ -f "$SNELLIUS_DIR/snellius_env.sh" ] || { echo "Submit from the repository root (REPO_ROOT=$REPO_ROOT)" >&2; exit 1; }
OUT_BASE="$REPO_ROOT/detection/output/$(dirname "${CFG#cfgs/}")/$(basename "$CFG" .yaml)"
SOURCE_DIR="$OUT_BASE/$SOURCE_TAG"
TARGET_DIR="$OUT_BASE/$TARGET_TAG"

# These belong to the individual steps; a value exported at submit time must not leak into all of them.
unset EPOCHS EVAL_EPOCHS TAG SENSOR NAME CKPT SMOKE
FT_PRETRAINED_SCOPE="${PRETRAINED_SCOPE:-encoder}"
# The seed is for the fine-tuning only (step 4); the pretraining run in step 1 stays the shared, unseeded one.
FT_SEED="${SEED:-}"
unset SEED

# Epoch numbers of the checkpoints saved in <run dir>/ckpt, ascending, space separated.
saved_epochs() {
    { ls "$1/ckpt/" 2>/dev/null || true; } | sed -n 's/^checkpoint_epoch_\([0-9]*\)\.pth$/\1/p' | sort -n | tr '\n' ' '
}
# Of the given epochs, those without a finished validation log (one that contains AP values) in <run dir>.
not_evaluated() {
    local dir=$1 e; shift
    for e in "$@"; do
        grep -qs "LEVEL_1/AP" "$dir/eval/epoch_$e/val/"log_eval_*.txt || printf '%s ' "$e"
    done
}
# Evaluate the saved checkpoints of run <tag> on <sensor> (only the ones not evaluated yet) and print the best one.
eval_and_select() {
    local tag=$1 sensor=$2 dir="$OUT_BASE/$1" epochs todo
    epochs=$(saved_epochs "$dir")
    # shellcheck disable=SC2086
    todo=$(not_evaluated "$dir" $epochs)
    if [ -n "$todo" ]; then
        DATA_ROOT="$DATA_ROOT" TAG="$tag" SENSOR="$sensor" EVAL_EPOCHS="$todo" CFG="$CFG" \
            bash "$SNELLIUS_DIR/eval_epochs.sbatch" >&2
    else
        echo "    epochs $epochs already evaluated" >&2
    fi
    # shellcheck disable=SC2086
    python "$REPO_ROOT/detection/tools/select_best_checkpoint.py" --run_dir "$dir" --epochs $epochs | tee /dev/stderr | tail -1
}

# shellcheck disable=SC1091
source "$SNELLIUS_DIR/snellius_env.sh"
activate_env

# ---- 1) pretrain on the source sensor -----------------------------------------------------------------------
if [ -f "$SOURCE_DIR/ckpt/checkpoint_epoch_${REQUIRE_EPOCH}.pth" ]; then
    echo "==> [1/5] $SOURCE_TAG already has epoch $REQUIRE_EPOCH: skipping pretraining"
else
    echo "==> [1/5] pretraining $SOURCE_TAG on $SOURCE_SENSOR from scratch"
    DATA_ROOT="$DATA_ROOT" SENSOR="$SOURCE_SENSOR" TAG="$SOURCE_TAG" PRETRAINED="" CFG="$CFG" \
        bash "$SNELLIUS_DIR/train_single_sensor.sbatch"
    [ -f "$SOURCE_DIR/ckpt/checkpoint_epoch_${REQUIRE_EPOCH}.pth" ] || {
        echo "ERROR: $SOURCE_DIR/ckpt/checkpoint_epoch_${REQUIRE_EPOCH}.pth missing after pretraining" >&2; exit 1; }
fi

# ---- 2) evaluate the pretraining checkpoints on the source sensor, pick the best ---------------------------
echo "==> [2/5] evaluating $SOURCE_TAG on val ($SOURCE_SENSOR) and selecting the best checkpoint"
BEST=$(eval_and_select "$SOURCE_TAG" "$SOURCE_SENSOR")
[ -f "$BEST" ] || { echo "ERROR: selection did not return a checkpoint" >&2; exit 1; }
BEST_EPOCH=$(basename "$BEST" .pth); BEST_EPOCH="${BEST_EPOCH##*_}"
echo "==> selected: $BEST"

# ---- 3) before fine-tuning: the source checkpoint fed the target sensor ------------------------------------
ZERO_NAME="${SOURCE_SENSOR}_e${BEST_EPOCH}_on_${TARGET_SENSOR}"
ZERO_DIR="$OUT_BASE/$ZERO_NAME"
if [ -z "$(not_evaluated "$ZERO_DIR" "$BEST_EPOCH")" ]; then
    echo "==> [3/5] $ZERO_NAME already evaluated: skipping"
else
    echo "==> [3/5] before fine-tuning: $SOURCE_TAG epoch $BEST_EPOCH evaluated on $TARGET_SENSOR"
    DATA_ROOT="$DATA_ROOT" SENSOR="$TARGET_SENSOR" NAME="$ZERO_NAME" CKPT="$BEST" CFG="$CFG" \
        bash "$SNELLIUS_DIR/eval_ckpt.sbatch"
fi

# ---- 4) fine-tune on the target sensor ----------------------------------------------------------------------
if [ -f "$TARGET_DIR/ckpt/checkpoint_epoch_${REQUIRE_EPOCH}.pth" ]; then
    echo "==> [4/5] $TARGET_TAG already has epoch $REQUIRE_EPOCH: skipping fine-tuning"
else
    echo "==> [4/5] fine-tuning $TARGET_TAG on $TARGET_SENSOR from $BEST (scope: $FT_PRETRAINED_SCOPE, seed: ${FT_SEED:-unset})"
    DATA_ROOT="$DATA_ROOT" SENSOR="$TARGET_SENSOR" TAG="$TARGET_TAG" PRETRAINED="$BEST" SEED="$FT_SEED" \
        PRETRAINED_SCOPE="$FT_PRETRAINED_SCOPE" CFG="$CFG" bash "$SNELLIUS_DIR/train_single_sensor.sbatch"
    [ -f "$TARGET_DIR/ckpt/checkpoint_epoch_${REQUIRE_EPOCH}.pth" ] || {
        echo "ERROR: $TARGET_DIR/ckpt/checkpoint_epoch_${REQUIRE_EPOCH}.pth missing after fine-tuning" >&2; exit 1; }
fi

# ---- 5) after fine-tuning: evaluate the fine-tuned checkpoints on the target sensor -------------------------
echo "==> [5/5] after fine-tuning: evaluating $TARGET_TAG on val ($TARGET_SENSOR)"
FT_BEST=$(eval_and_select "$TARGET_TAG" "$TARGET_SENSOR")
[ -f "$FT_BEST" ] || { echo "ERROR: selection did not return a checkpoint" >&2; exit 1; }

# ---- summary ------------------------------------------------------------------------------------------------
cd "$REPO_ROOT/detection/tools"
python - "$SOURCE_DIR" "$ZERO_DIR" "$TARGET_DIR" "$BEST_EPOCH" "$SOURCE_SENSOR" "$TARGET_SENSOR" <<'PY'
import glob, json, os, sys
from select_best_checkpoint import AP_KEY, overall_block

source_dir, zero_dir, target_dir, epoch, src, tgt = sys.argv[1:]
classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']

def row(label, aps):
    mean = sum(aps.values()) / len(aps) if aps else float('nan')
    print('%-46s' % label + ''.join('%12s' % ('%.4f' % aps[c] if c in aps else '-') for c in classes) + '%12.4f' % mean)

def from_json(run_dir):
    best = json.load(open(os.path.join(run_dir, 'best_checkpoint.json')))
    return best['best_epoch'], best['per_class_ap']

def from_log(run_dir, epoch):
    logs = sorted(glob.glob(os.path.join(run_dir, 'eval', 'epoch_%s' % epoch, 'val', 'log_eval_*.txt')), key=os.path.getmtime)
    return {AP_KEY.search(k).group(1): v for k, v in overall_block(logs[-1]).items() if AP_KEY.search(k)}

print('\n==================== %s -> %s: Level-1 AP on val ====================' % (src, tgt))
print('%-46s' % '' + ''.join('%12s' % c for c in classes) + '%12s' % 'mean')
e, aps = from_json(source_dir)
row('%s model on %s (epoch %d)' % (src, src, e), aps)
row('before fine-tuning: %s model on %s' % (src, tgt), from_log(zero_dir, epoch))
e, aps = from_json(target_dir)
row('after fine-tuning on %s (epoch %d)' % (tgt, e), aps)
PY
