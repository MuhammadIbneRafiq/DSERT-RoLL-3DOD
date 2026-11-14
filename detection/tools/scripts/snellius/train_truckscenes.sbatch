#!/bin/bash
#
# Train one VoxelNeXt run on MAN TruckScenes, one modality (SENSOR=lidar: the 6 LiDARs, SENSOR=radar: the 6 radars),
# and score it with the devkit's official evaluation on the val split at the end. Normally submitted by
# submit_truckscenes.sh; run prepare_truckscenes.sbatch once first (devkit install, info files, data checks).
#
#   sbatch --export=ALL,DATA_ROOT=/scratch-shared/$USER/truckscenes/man-truckscenes,SENSOR=radar,SMOKE=1 \
#       detection/tools/scripts/snellius/train_truckscenes.sbatch
#
# Data staging (STAGE=auto, on a node with local NVMe): copies only the files this run reads, i.e. for both splits the
# keyframe + the first NSWEEPS sweeps of each channel of this modality (listed from the info files), plus the devkit
# tables (the evaluation needs them) and the info files. Radar stages ~10 GB, LiDAR (3 sweeps) about half of the 459 GB
# LiDAR folder.
#
# Batch and LR: 1 GPU x BATCH (default 16, the DSERT-RoLL VoxelNeXt runs' batch). LR defaults to OpenPCDet's nuScenes
# VoxelNeXt LR (0.001 at a global batch of 32) scaled linearly: 0.001 x BATCH / 32, i.e. 0.0005 at 16 (the same rule
# the DSERT-RoLL VoxelNeXt config used: 0.003 @ 32 -> 0.0015 @ 16). Set LR=... to override.
#
# Re-submitting with the same TAG resumes from the newest checkpoint. wandb: as train_single_sensor.sbatch.
# Keep --time realistic (Snellius reserves time limit x SBU/h from the budget when the job starts).
#
#SBATCH --job-name=ts-voxelnext
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint=scratch-node
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# ---- settings (override with sbatch --export=ALL,NAME=value,...) ------------------------------------------
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the TruckScenes dataroot (the folder containing samples/, sweeps/, v1.2-trainval/)}"
VERSION="${VERSION:-v1.2-trainval}"
SENSOR="${SENSOR:-lidar}"                        # lidar | radar
CFG="${CFG:-cfgs/det_model_cfgs/truckscenes/voxelnext.yaml}"
TAG="${TAG:-ts_${SENSOR}_$(date +%Y%m%d)}"
BATCH="${BATCH:-16}"                             # per GPU = global (1 GPU)
LR="${LR:-}"                                     # empty = 0.001 x BATCH / 32
NSWEEPS="${NSWEEPS:-}"                           # empty = config (lidar 3, radar 6)
AMP="${AMP:-fp16}"                               # fp16 | bf16 | none (training only; evaluation runs fp32)
COMPILE="${COMPILE:-1}"
EPOCHS="${EPOCHS:-}"                             # empty = OPTIMIZATION.NUM_EPOCHS (20)
PRETRAINED="${PRETRAINED:-}"                     # checkpoint to initialise from (the LiDAR run's last epoch)
PRETRAINED_SCOPE="${PRETRAINED_SCOPE:-encoder}"  # encoder = VFE + sparse backbone (head fresh) | all
SMOKE="${SMOKE:-0}"                              # 1 = every 50th sample of train and val, 2 epochs
SKIP_EVAL="${SKIP_EVAL:-0}"
STAGE="${STAGE:-auto}"                           # auto = stage to node-local NVMe when the node has it | 1 | 0
WANDB_PROJECT="${WANDB_PROJECT-truckscenes-single-sensor}"
SEED="${SEED:-}"
INPUT_FEATURE="${INPUT_FEATURE:-}"               # 4th channel: empty = intensity (LiDAR) / doppler (radar) | rcs | none
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_SETS="${EXTRA_SETS:-}"                     # extra "KEY value KEY value ..." config overrides

SNELLIUS_DIR="$REPO_ROOT/detection/tools/scripts/snellius"
[ -f "$SNELLIUS_DIR/snellius_env.sh" ] || { echo "Submit from the repository root (REPO_ROOT=$REPO_ROOT)" >&2; exit 1; }
# shellcheck disable=SC1091
source "$SNELLIUS_DIR/snellius_env.sh"
case "$SENSOR" in lidar|radar) ;; *) echo "SENSOR must be lidar or radar, got $SENSOR" >&2; exit 1 ;; esac

# ---- environment ------------------------------------------------------------------------------------------
load_cuda_module
activate_env
python -c "import truckscenes" 2>/dev/null || {
    echo "ERROR: truckscenes devkit not installed in the env. Run prepare_truckscenes.sbatch first." >&2; exit 1; }
[ -f "$DATA_ROOT/al3d_infos/$VERSION/truckscenes_infos_train.pkl" ] || {
    echo "ERROR: no info files in $DATA_ROOT/al3d_infos/$VERSION. Run prepare_truckscenes.sbatch first." >&2; exit 1; }

NGPUS="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
if [ "$NGPUS" != "1" ]; then
    echo "ERROR: this job is set up for exactly 1 GPU, got $NGPUS (the runs compare at 1 GPU x BATCH)." >&2
    exit 1
fi
CPUS="${SLURM_CPUS_PER_TASK:-16}"
WORKERS="${WORKERS:-$(( CPUS - 2 ))}"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}"
export PYTHONWARNINGS='ignore:`torch.cuda.amp.custom_:FutureWarning'
export TF_CPP_MIN_LOG_LEVEL=2
if [ -n "$WANDB_PROJECT" ] && ! wandb_logged_in; then
    echo "WARNING: wandb is not logged in for $USER. Run 'wandb login' once on a login node. Logging offline for now." >&2
    export WANDB_MODE=offline
fi

[ -z "$LR" ] && LR=$(python -c "print(0.001 * $BATCH / 32)")
if [ "$SMOKE" = "1" ]; then
    EPOCHS="${EPOCHS:-2}"
    TAG="${TAG}_smoke"
fi

cd "$REPO_ROOT/detection/tools"
echo "==> host $(hostname) | job ${SLURM_JOB_ID:-} | ${NGPUS} GPU | ${CPUS} CPUs | ${WORKERS} workers | $SENSOR | batch $BATCH | LR $LR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# ---- stage this modality's files to node-local NVMe ---------------------------------------------------------
RUN_DATA="$DATA_ROOT"
ON_NVME=0
case "${TMPDIR:-}" in /scratch-node/*) ON_NVME=1 ;; esac
if [ "$STAGE" = "1" ] || { [ "$STAGE" = "auto" ] && [ "$ON_NVME" = "1" ]; }; then
    if [ "$ON_NVME" != "1" ]; then
        echo "WARNING: no node-local scratch on this node (TMPDIR=${TMPDIR:-unset}); reading data in place." >&2
    else
        RUN_DATA="$TMPDIR/man-truckscenes"
        mkdir -p "$RUN_DATA"
        start=$(date +%s)
        rsync -a "$DATA_ROOT/$VERSION" "$RUN_DATA/"
        mkdir -p "$RUN_DATA/al3d_infos"
        rsync -a "$DATA_ROOT/al3d_infos/$VERSION" "$RUN_DATA/al3d_infos/"
        LIST="$TMPDIR/staged_files.txt"
        # The files the dataset will read: train + val, this modality, first NSWEEPS sweeps per channel.
        python - "$RUN_DATA/al3d_infos/$VERSION" "$SENSOR" "$CFG" "${NSWEEPS}" > "$LIST" <<'EOF'
import pickle, sys, yaml, os
info_dir, sensor, cfg, nsw = sys.argv[1:5]
if not nsw:
    base = yaml.safe_load(open(cfg))['DATA_CONFIG']['_BASE_CONFIG_']
    nsw = yaml.safe_load(open(base))['SENSOR']['NSWEEPS'][sensor]
files = set()
for split in ('train', 'val'):
    with open(os.path.join(info_dir, 'truckscenes_infos_%s.pkl' % split), 'rb') as f:
        for info in pickle.load(f)['infos']:
            for chain in info['sweeps'][sensor].values():
                files.update(sw['filename'] for sw in chain[:int(nsw)])
print('\n'.join(sorted(files)))
EOF
        echo "==> staging $(wc -l < "$LIST") $SENSOR files + $VERSION tables + infos to $RUN_DATA"
        rsync -a --files-from="$LIST" "$DATA_ROOT/" "$RUN_DATA/"
        echo "==> staged $(du -sh "$RUN_DATA" | cut -f1) in $(( $(date +%s) - start ))s"
    fi
fi

# ---- config overrides -------------------------------------------------------------------------------------
SETS=(DATA_CONFIG.DATA_PATH "$RUN_DATA" DATA_CONFIG.VERSION "$VERSION" DATA_CONFIG.SENSOR.NAME "$SENSOR"
      OPTIMIZATION.LR "$LR")
[ -n "$NSWEEPS" ] && SETS+=(DATA_CONFIG.SENSOR.NSWEEPS."$SENSOR" "$NSWEEPS")
[ -n "$INPUT_FEATURE" ] && SETS+=(DATA_CONFIG.SENSOR.FEATURE "$INPUT_FEATURE")
# ~2% of samples for train and val: exercises data, AMP, compile, checkpointing and the official evaluation (which then
# reports "NOT official": the other val samples have no predictions).
[ "$SMOKE" = "1" ] && SETS+=(DATA_CONFIG.SAMPLED_INTERVAL.train 50 DATA_CONFIG.SAMPLED_INTERVAL.test 50)
# shellcheck disable=SC2206
if [ -n "$EXTRA_SETS" ]; then set -f; SETS+=($EXTRA_SETS); set +f; fi
TRAIN_ARGS=(--launcher pytorch --cfg_file "$CFG" --extra_tag "$TAG" --workers "$WORKERS" --batch_size "$BATCH"
            --amp "$AMP" --find_unused_parameters --max_ckpt_save_num 5)
[ "$COMPILE" = "1" ] && TRAIN_ARGS+=(--compile)
[ -n "$EPOCHS" ] && TRAIN_ARGS+=(--epochs "$EPOCHS")
[ -n "$PRETRAINED" ] && TRAIN_ARGS+=(--pretrained_model "$PRETRAINED" --pretrained_scope "$PRETRAINED_SCOPE")
[ "$SKIP_EVAL" = "1" ] && TRAIN_ARGS+=(--skip_final_eval)
[ -n "$SEED" ] && TRAIN_ARGS+=(--seed "$SEED")
[ -n "$WANDB_PROJECT" ] && TRAIN_ARGS+=(--wandb_project "$WANDB_PROJECT")

echo "==> torchrun train_mm.py ${TRAIN_ARGS[*]} ${EXTRA_ARGS} --set ${SETS[*]}"
# shellcheck disable=SC2086
torchrun --standalone --nnodes=1 --nproc_per_node=1 train_mm.py "${TRAIN_ARGS[@]}" $EXTRA_ARGS --set "${SETS[@]}"

OUT_DIR="$REPO_ROOT/detection/output/$(dirname "${CFG#cfgs/}")/$(basename "$CFG" .yaml)/$TAG"
echo "==> done. Outputs: $OUT_DIR"
if compgen -G "$OUT_DIR/wandb/offline-run-*" > /dev/null; then
    echo "==> wandb logged offline. To upload, on a login node run:"
    echo "    bash $SNELLIUS_DIR/wandb_sync.sh $OUT_DIR"
fi
