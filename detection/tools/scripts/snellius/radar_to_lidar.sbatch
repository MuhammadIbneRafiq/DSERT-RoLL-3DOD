#!/bin/bash
#
# The reverse of finetune_from_best.sbatch: take the finished radar run, fine-tune it on LiDAR, and evaluate before
# and after. One job, four steps. Every step is skipped if its result is already on disk, so re-submitting the same
# command after a time-out continues where it stopped. The radar run itself is not trained here: the job stops if
# <SOURCE_TAG>/ckpt/checkpoint_epoch_<REQUIRE_EPOCH>.pth is missing.
#
#   1) evaluate the radar checkpoints on radar, pick the best (mean Level-1 AP) -> <SOURCE_TAG>/best_checkpoint.json
#   2) BEFORE fine-tuning: that radar checkpoint evaluated on LiDAR as is  -> <SOURCE_SENSOR>_e<N>_on_<TARGET_SENSOR>/eval/
#   3) fine-tune on LiDAR from that checkpoint                             -> <TARGET_TAG>/ckpt/
#   4) AFTER fine-tuning: evaluate its checkpoints on LiDAR, pick the best -> <TARGET_TAG>/best_checkpoint.json
#
# and a three-line summary at the end of the slurm-*.out file. All folders are under
# detection/output/det_model_cfgs/dsert/lidar_only/.
#
# Submit from the repository root:
#
#   sbatch --export=ALL,DATA_ROOT=$HOME/data/dsert_roll detection/tools/scripts/snellius/radar_to_lidar.sbatch
#
#   # a radar run saved under another tag
#   sbatch --export=ALL,DATA_ROOT=$HOME/data/dsert_roll,SOURCE_TAG=<tag> detection/tools/scripts/snellius/radar_to_lidar.sbatch
#
#   # extra fine-tuning seeds (works in either direction; steps 1-2 are already on disk, so only 3-4 run).
#   # SEED=1 goes to <TARGET_SENSOR>_ft_from_<SOURCE_SENSOR>_s1; the unseeded first run keeps its old tag.
#   sbatch --time=16:00:00 --job-name=ft_radar2livox_s1 --export=ALL,DATA_ROOT=$HOME/data/dsert_roll,SEED=1 detection/tools/scripts/snellius/radar_to_lidar.sbatch
#   sbatch --time=16:00:00 --job-name=ft_livox2radar_s1 --export=ALL,DATA_ROOT=$HOME/data/dsert_roll,SEED=1,SOURCE_TAG=livox_pretrain,SOURCE_SENSOR=livox,TARGET_SENSOR=radar detection/tools/scripts/snellius/radar_to_lidar.sbatch
#
#   # geometry only: the target sensor's 4th channel set to zeros (run names get _xyz)
#   sbatch ... --export=ALL,DATA_ROOT=$HOME/data/dsert_roll,SEED=1,INPUT_FEATURE=none,SOURCE_TAG=livox_pretrain,SOURCE_SENSOR=livox,TARGET_SENSOR=radar ...
#
#   # radar -> LiDAR from a radar run trained without Doppler (run names get radarxyz, e.g. livox_ft_from_radarxyz_s1)
#   sbatch ... --export=ALL,DATA_ROOT=$HOME/data/dsert_roll,SEED=1,SOURCE_TAG=radar_scratch_xyz_s1,SOURCE_FEATURE=none ...
#
# Settings: SOURCE_TAG (radar_scratch), SOURCE_SENSOR (radar), SOURCE_FEATURE (empty = default; none = the source run
# was trained x/y/z only), TARGET_SENSOR (livox), TARGET_TAG
# (<TARGET_SENSOR>_ft_from_<SOURCE_SENSOR>[xyz][_xyz][_s<SEED>]), SEED (unset; fine-tuning only), INPUT_FEATURE (empty =
# sensor default; none = x/y/z only; target sensor only), STOP_AFTER_ZERO (0; 1 = steps 1-2 only), SKIP_ZERO (0; 1 = skip step 2), REQUIRE_EPOCH (20). Training
# settings (AMP, COMPILE, PRETRAINED_SCOPE, WANDB_PROJECT, ...) are passed through to train_single_sensor.sbatch.
#
# Time: one 20-epoch training plus up to 11 evaluations. Snellius reserves (time limit x 192 SBU/h) when the job
# starts, so keep --time realistic.
#
#SBATCH --job-name=radar2lidar
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint=scratch-node
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
export REPO_ROOT
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
SOURCE_TAG="${SOURCE_TAG:-radar_scratch}"
SOURCE_SENSOR="${SOURCE_SENSOR:-radar}"
TARGET_SENSOR="${TARGET_SENSOR:-livox}"
# 4th point channel of the TARGET sensor (steps 2-4 only; the source model is always evaluated with its own default):
# empty = default (radar doppler, LiDAR intensity), none = zeros (x/y/z only), or a stored field such as power.
# A non-default feature is part of the run names (e.g. radar_ft_from_livox_xyz_s1, livox_e20_on_radar_xyz).
FT_FEATURE="${INPUT_FEATURE:-}"
unset INPUT_FEATURE
case "$(echo "$FT_FEATURE" | tr '[:upper:]' '[:lower:]')" in
    "") FEAT_SUFFIX="" ;;
    none|zeros|xyz) FEAT_SUFFIX="_xyz" ;;
    *) FEAT_SUFFIX="_$FT_FEATURE" ;;
esac
# 4th point channel the SOURCE run was trained with (step 1 only; must match its training): empty = default,
# none = zeros (e.g. SOURCE_TAG=radar_scratch_xyz_s1). A non-default source is part of the run names
# (radarxyz: livox_ft_from_radarxyz_s1, radarxyz_e20_on_livox).
SRC_FEATURE="${SOURCE_FEATURE:-}"
case "$(echo "$SRC_FEATURE" | tr '[:upper:]' '[:lower:]')" in
    "") SRC_SUFFIX="" ;;
    none|zeros|xyz) SRC_SUFFIX="xyz" ;;
    *) SRC_SUFFIX="$SRC_FEATURE" ;;
esac
TARGET_TAG="${TARGET_TAG:-${TARGET_SENSOR}_ft_from_${SOURCE_SENSOR}${SRC_SUFFIX}${FEAT_SUFFIX}${SEED:+_s$SEED}}"
REQUIRE_EPOCH="${REQUIRE_EPOCH:-20}"
STOP_AFTER_ZERO="${STOP_AFTER_ZERO:-0}"      # 1 = only steps 1-2 (select the source checkpoint, zero-shot evaluation)
SKIP_ZERO="${SKIP_ZERO:-0}"                  # 1 = skip step 2 (the zero-shot evaluation) and go straight to fine-tuning
CFG="${CFG:-cfgs/det_model_cfgs/dsert/lidar_only.yaml}"

SNELLIUS_DIR="$REPO_ROOT/detection/tools/scripts/snellius"
[ -f "$SNELLIUS_DIR/snellius_env.sh" ] || { echo "Submit from the repository root (REPO_ROOT=$REPO_ROOT)" >&2; exit 1; }
OUT_BASE="$REPO_ROOT/detection/output/$(dirname "${CFG#cfgs/}")/$(basename "$CFG" .yaml)"
SOURCE_DIR="$OUT_BASE/$SOURCE_TAG"
TARGET_DIR="$OUT_BASE/$TARGET_TAG"

# These belong to the individual steps; a value exported at submit time must not leak into all of them.
unset EPOCHS EVAL_EPOCHS TAG SENSOR NAME CKPT SMOKE
FT_PRETRAINED_SCOPE="${PRETRAINED_SCOPE:-encoder}"
# The seed is for the fine-tuning only (step 3); the pretraining run stays the shared, unseeded one.
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
# Evaluate the saved checkpoints of run <tag> on <sensor> with 4th channel <feature> (only the ones not evaluated
# yet) and print the best one.
eval_and_select() {
    local tag=$1 sensor=$2 feature=$3 dir="$OUT_BASE/$1" epochs todo
    epochs=$(saved_epochs "$dir")
    # shellcheck disable=SC2086
    todo=$(not_evaluated "$dir" $epochs)
    if [ -n "$todo" ]; then
        DATA_ROOT="$DATA_ROOT" TAG="$tag" SENSOR="$sensor" EVAL_EPOCHS="$todo" CFG="$CFG" INPUT_FEATURE="$feature" \
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

[ -f "$SOURCE_DIR/ckpt/checkpoint_epoch_${REQUIRE_EPOCH}.pth" ] || {
    echo "ERROR: $SOURCE_DIR/ckpt/checkpoint_epoch_${REQUIRE_EPOCH}.pth missing: set SOURCE_TAG to the finished $SOURCE_SENSOR run" >&2; exit 1; }

# ---- 1) evaluate the pretraining checkpoints on the source sensor, pick the best ---------------------------
echo "==> [1/4] evaluating $SOURCE_TAG on val ($SOURCE_SENSOR) and selecting the best checkpoint"
BEST=$(eval_and_select "$SOURCE_TAG" "$SOURCE_SENSOR" "$SRC_FEATURE")
[ -f "$BEST" ] || { echo "ERROR: selection did not return a checkpoint" >&2; exit 1; }
BEST_EPOCH=$(basename "$BEST" .pth); BEST_EPOCH="${BEST_EPOCH##*_}"
echo "==> selected: $BEST"

# ---- 2) before fine-tuning: the source checkpoint fed the target sensor ------------------------------------
ZERO_NAME="${SOURCE_SENSOR}${SRC_SUFFIX}_e${BEST_EPOCH}_on_${TARGET_SENSOR}${FEAT_SUFFIX}"
ZERO_DIR="$OUT_BASE/$ZERO_NAME"
if [ "$SKIP_ZERO" = "1" ]; then
    echo "==> [2/4] SKIP_ZERO=1: skipping the before-fine-tuning evaluation"
elif [ -z "$(not_evaluated "$ZERO_DIR" "$BEST_EPOCH")" ]; then
    echo "==> [2/4] $ZERO_NAME already evaluated: skipping"
else
    echo "==> [2/4] before fine-tuning: $SOURCE_TAG epoch $BEST_EPOCH evaluated on $TARGET_SENSOR (4th channel: ${FT_FEATURE:-default})"
    DATA_ROOT="$DATA_ROOT" SENSOR="$TARGET_SENSOR" NAME="$ZERO_NAME" CKPT="$BEST" CFG="$CFG" INPUT_FEATURE="$FT_FEATURE" \
        bash "$SNELLIUS_DIR/eval_ckpt.sbatch"
fi
if [ "$STOP_AFTER_ZERO" = "1" ]; then
    echo "==> STOP_AFTER_ZERO=1: done after step 2 ($ZERO_DIR)"
    exit 0
fi

# ---- 3) fine-tune on the target sensor ----------------------------------------------------------------------
if [ -f "$TARGET_DIR/ckpt/checkpoint_epoch_${REQUIRE_EPOCH}.pth" ]; then
    echo "==> [3/4] $TARGET_TAG already has epoch $REQUIRE_EPOCH: skipping fine-tuning"
else
    echo "==> [3/4] fine-tuning $TARGET_TAG on $TARGET_SENSOR from $BEST (scope: $FT_PRETRAINED_SCOPE, seed: ${FT_SEED:-unset}, 4th channel: ${FT_FEATURE:-default})"
    DATA_ROOT="$DATA_ROOT" SENSOR="$TARGET_SENSOR" TAG="$TARGET_TAG" PRETRAINED="$BEST" SEED="$FT_SEED" \
        PRETRAINED_SCOPE="$FT_PRETRAINED_SCOPE" CFG="$CFG" INPUT_FEATURE="$FT_FEATURE" \
        bash "$SNELLIUS_DIR/train_single_sensor.sbatch"
    [ -f "$TARGET_DIR/ckpt/checkpoint_epoch_${REQUIRE_EPOCH}.pth" ] || {
        echo "ERROR: $TARGET_DIR/ckpt/checkpoint_epoch_${REQUIRE_EPOCH}.pth missing after fine-tuning" >&2; exit 1; }
fi

# ---- 4) after fine-tuning: evaluate the fine-tuned checkpoints on the target sensor -------------------------
echo "==> [4/4] after fine-tuning: evaluating $TARGET_TAG on val ($TARGET_SENSOR)"
FT_BEST=$(eval_and_select "$TARGET_TAG" "$TARGET_SENSOR" "$FT_FEATURE")
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
    if not logs:  # step 2 skipped (SKIP_ZERO=1) or failed
        return {}
    return {AP_KEY.search(k).group(1): v for k, v in overall_block(logs[-1]).items() if AP_KEY.search(k)}

print('\n==================== %s -> %s: Level-1 AP on val ====================' % (src, tgt))
print('%-46s' % '' + ''.join('%12s' % c for c in classes) + '%12s' % 'mean')
e, aps = from_json(source_dir)
row('%s model on %s (epoch %d)' % (src, src, e), aps)
row('before fine-tuning: %s model on %s' % (src, tgt), from_log(zero_dir, epoch))
e, aps = from_json(target_dir)
row('after fine-tuning on %s (epoch %d)' % (tgt, e), aps)
PY
