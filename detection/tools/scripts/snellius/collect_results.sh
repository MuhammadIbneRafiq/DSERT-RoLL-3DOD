#!/bin/bash
#
# Pack every run's epoch-20 predictions and logs into one archive to download, in the same layout as the
# dsert_results/ folder on the laptop:
#
#   dsert_results/<RUN>.pkl          result.pkl of epoch 20 (what frame_inspector.ipynb reads)
#   dsert_results/<RUN>/log_*.txt    the run's training log(s) and evaluation log(s), and best_checkpoint.json if any
#
# Only copies files, so it runs on a login node. Run it from the repository root:
#
#   bash detection/tools/scripts/snellius/collect_results.sh              # list found / missing, then pack
#   SUBMIT=1 bash detection/tools/scripts/snellius/collect_results.sh     # also submit a 1 h evaluation for every
#                                                                         # run whose epoch 20 has no result.pkl
#
# After the submitted evaluations finish, run it again (without SUBMIT) to pack.
# A run missing result.pkl is evaluated with the sensor and 4th channel it was trained with, read from its name:
# radar_* / livox_* / ouster_* = the sensor, _xyz = x/y/z only (4th channel zeros). A zero-shot run
# <A>_e<N>_on_<B>[_xyz] is checkpoint N of A's pretraining run (livox -> livox_pretrain, radar -> radar_scratch,
# radarxyz -> radar_scratch_xyz_s1) fed sensor B. Runs whose training has not reached epoch 20 are listed and skipped.
#
# Settings: EPOCH (20), OUT ($HOME/dsert_results_export), DATA_ROOT ($HOME/data/dsert_roll, for SUBMIT=1),
# CFG (cfgs/det_model_cfgs/dsert/lidar_only.yaml).

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
SNELLIUS_DIR="$REPO_ROOT/detection/tools/scripts/snellius"
[ -f "$SNELLIUS_DIR/snellius_env.sh" ] || { echo "Run from the repository root (REPO_ROOT=$REPO_ROOT)" >&2; exit 1; }
EPOCH="${EPOCH:-20}"
OUT="${OUT:-$HOME/dsert_results_export}"
DATA_ROOT="${DATA_ROOT:-$HOME/data/dsert_roll}"
SUBMIT="${SUBMIT:-0}"
CFG="${CFG:-cfgs/det_model_cfgs/dsert/lidar_only.yaml}"
RUNS_DIR="$REPO_ROOT/detection/output/$(dirname "${CFG#cfgs/}")/$(basename "$CFG" .yaml)"

# the source run behind each zero-shot name prefix (radarxyz = the radar run trained without Doppler)
declare -A PRETRAIN_TAG=([livox]=livox_pretrain [radar]=radar_scratch [ouster]=ouster_pretrain [radarxyz]=radar_scratch_xyz_s1)

# result.pkl of <run dir> at <epoch>: the evaluation at the end of training first (what the existing .pkl files are),
# then a separate evaluation (eval_epochs.sbatch / eval_ckpt.sbatch).
find_pkl() {
    local p
    for p in "$1/eval/eval_with_train/epoch_$2/val/result.pkl" "$1/eval/epoch_$2/val/result.pkl"; do
        [ -f "$p" ] && { echo "$p"; return; }
    done
    return 0
}

EXPORT="$OUT/dsert_results"
rm -rf "$EXPORT"
mkdir -p "$EXPORT"
n_ok=0; n_missing=0; n_running=0

for dir in "$RUNS_DIR"/*/; do
    dir="${dir%/}"; tag=$(basename "$dir")
    case "$tag" in *_smoke) continue ;; esac
    feature=""
    case "$tag" in *_xyz|*_xyz_*) feature=none ;; esac

    if [[ "$tag" =~ ^((livox|radar|ouster)(xyz)?)_e([0-9]+)_on_(livox|radar|ouster)(_xyz)?$ ]]; then
        kind=zeroshot; epoch=${BASH_REMATCH[4]}; sensor=${BASH_REMATCH[5]}
        ckpt="$RUNS_DIR/${PRETRAIN_TAG[${BASH_REMATCH[1]}]}/ckpt/checkpoint_epoch_$epoch.pth"
    elif [[ "$tag" =~ ^(livox|radar|ouster)_ ]]; then
        kind=train; epoch=$EPOCH; sensor=${BASH_REMATCH[1]}
        ckpt="$dir/ckpt/checkpoint_epoch_$epoch.pth"
    else
        echo "  ?        $tag (name not recognised, skipped)"; continue
    fi

    pkl=$(find_pkl "$dir" "$epoch")
    if [ -n "$pkl" ]; then
        cp "$pkl" "$EXPORT/$tag.pkl"
        mkdir -p "$EXPORT/$tag"
        for f in "$dir"/log_train_*.txt "$dir"/eval/*/val/log_eval_*.txt "$dir"/best_checkpoint.json; do
            [ -f "$f" ] && cp "$f" "$EXPORT/$tag/"
        done
        echo "  ok       $tag  <- ${pkl#"$RUNS_DIR/"}"
        n_ok=$((n_ok + 1))
    elif [ -f "$ckpt" ]; then
        echo "  MISSING  $tag: epoch $epoch has a checkpoint but no result.pkl (sensor $sensor, 4th channel ${feature:-default})"
        n_missing=$((n_missing + 1))
        if [ "$SUBMIT" = "1" ]; then
            if [ "$kind" = train ]; then
                (cd "$REPO_ROOT" && sbatch --time=01:00:00 --job-name="eval_$tag" \
                    --export=ALL,DATA_ROOT="$DATA_ROOT",TAG="$tag",SENSOR="$sensor",EVAL_EPOCHS="$epoch",INPUT_FEATURE="$feature",CFG="$CFG" \
                    "$SNELLIUS_DIR/eval_epochs.sbatch")
            else
                (cd "$REPO_ROOT" && sbatch --time=01:00:00 --job-name="eval_$tag" \
                    --export=ALL,DATA_ROOT="$DATA_ROOT",NAME="$tag",SENSOR="$sensor",CKPT="$ckpt",INPUT_FEATURE="$feature",CFG="$CFG" \
                    "$SNELLIUS_DIR/eval_ckpt.sbatch")
            fi
        fi
    else
        echo "  running  $tag: no checkpoint_epoch_$epoch yet (training not finished, or timed out: resubmit it)"
        n_running=$((n_running + 1))
    fi
done

tar -czf "$OUT/dsert_results.tar.gz" -C "$OUT" dsert_results
echo ""
echo "==> $n_ok runs packed, $n_missing missing result.pkl, $n_running without epoch $EPOCH"
echo "==> $OUT/dsert_results.tar.gz ($(du -h "$OUT/dsert_results.tar.gz" | cut -f1))"
[ "$n_missing" -gt 0 ] && [ "$SUBMIT" != "1" ] && echo "==> rerun with SUBMIT=1 to evaluate the missing ones"
exit 0
