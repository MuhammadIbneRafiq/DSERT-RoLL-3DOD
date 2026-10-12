#!/bin/bash
#
# Single-sensor (LiDAR or radar) CenterPoint + LoGoHead on DSERT-RoLL, 1 x H100 (quarter node), Snellius.
# No cameras, no radar branch. Mixed precision + torch.compile on the dense layers + TF32.
#
# Submit from the repository root (one-time setup first: detection/tools/scripts/snellius/setup_env.sh):
#
#   # 1) smoke test: ~1% of the data, 2 epochs, ~20 min. Do this first and note the it/s it reports.
#   sbatch --time=01:00:00 --export=ALL,DATA_ROOT=/path/to/dsert_roll,SMOKE=1 detection/tools/scripts/snellius/train_single_sensor.sbatch
#
#   # 2) full Ouster pretraining run. hours ~= 20 epochs * steps per epoch / (it/s * 3600); set --time to ~1.5x that.
#   sbatch --time=12:00:00 --export=ALL,DATA_ROOT=/path/to/dsert_roll,TAG=ouster_pretrain,STAGE=1 detection/tools/scripts/snellius/train_single_sensor.sbatch
#
#   # 3) later: radar fine-tuning from the Ouster checkpoint, and the radar-from-scratch baseline
#   sbatch --export=ALL,DATA_ROOT=...,SENSOR=radar,TAG=radar_ft_from_ouster,PRETRAINED=<ckpt.pth> ...
#   sbatch --export=ALL,DATA_ROOT=...,SENSOR=radar,TAG=radar_scratch ...
#
# Re-submitting with the same TAG resumes from the newest checkpoint in that run's ckpt/ folder, and continues
# the same wandb run. wandb logs to project WANDB_PROJECT (default dsert-single-sensor; set it empty to turn off)
# after a one-time `wandb login` on a login node. If the node cannot reach wandb it logs offline; upload with
# detection/tools/scripts/snellius/wandb_sync.sh from a login node.
# One GPU matches the config's global batch of 16 at LR 0.001; with more GPUs, rescale LR in the config.
#
#SBATCH --job-name=dsert-single
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# ---- settings (override with sbatch --export=ALL,NAME=value,...) ------------------------------------------
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the DSERT-RoLL root (the folder containing ImageSets/ and processed_data/)}"
ENV_DIR="${ENV_DIR:-$HOME/envs/dsert-lidar}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniforge3}"
MODULE_STACK="${MODULE_STACK:-2023}"
CUDA_MODULE="${CUDA_MODULE:-CUDA/12.1.1}"

SENSOR="${SENSOR:-ouster}"                       # ouster | livox | radar
CFG="${CFG:-cfgs/det_model_cfgs/dsert/lidar_only.yaml}"
TAG="${TAG:-${SENSOR}_$(date +%Y%m%d)}"
AMP="${AMP:-fp16}"                               # fp16 (GradScaler) | bf16 | none
COMPILE="${COMPILE:-1}"                          # 1 = torch.compile the dense layers
EPOCHS="${EPOCHS:-}"                             # empty = OPTIMIZATION.NUM_EPOCHS from the config
BATCH_PER_GPU="${BATCH_PER_GPU:-}"               # empty = OPTIMIZATION.BATCH_SIZE_PER_GPU from the config
PRETRAINED="${PRETRAINED:-}"                     # checkpoint to initialise from (e.g. the Ouster run's last epoch)
PRETRAINED_SCOPE="${PRETRAINED_SCOPE:-encoder}"  # encoder = copy 3D/BEV backbones only (heads fresh) | all
SMOKE="${SMOKE:-0}"                              # 1 = tiny subset, 2 epochs: checks the whole pipeline end to end
PREFLIGHT="${PREFLIGHT:-1}"                      # 1 = run preflight_single_sensor.py first and stop if it fails
SKIP_EVAL="${SKIP_EVAL:-0}"                      # 1 = skip the Waymo-AP evaluation after training
STAGE="${STAGE:-0}"                              # 1 = copy labels + this sensor's frames to node-local $TMPDIR first
WANDB_PROJECT="${WANDB_PROJECT-dsert-single-sensor}"   # Weights & Biases project; WANDB_PROJECT= (empty) turns it off
EXTRA_ARGS="${EXTRA_ARGS:-}"                     # anything else for train_mm.py (before --set)

# ---- environment ------------------------------------------------------------------------------------------
module purge
module load "$MODULE_STACK"
module load "$CUDA_MODULE"
# shellcheck disable=SC1091
source "$CONDA_DIR/bin/activate" "$ENV_DIR"

NGPUS="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
CPUS="${SLURM_CPUS_PER_TASK:-16}"
WORKERS="${WORKERS:-$(( CPUS / NGPUS - 2 ))}"    # dataloader workers per GPU process; leave 2 cores per rank
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}"   # reuse compiled kernels across jobs
export NCCL_ASYNC_ERROR_HANDLING=1

# wandb reads its API key from ~/.netrc (written by `wandb login` on a login node; home is shared with
# compute nodes) or from WANDB_API_KEY. Without either, log offline rather than failing the job.
if [ -n "$WANDB_PROJECT" ] && [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
    echo "WARNING: wandb is not logged in for $USER. Run 'wandb login' once on a login node. Logging offline for now." >&2
    export WANDB_MODE=offline
fi

cd "$REPO_ROOT/detection/tools"
[ -f train_mm.py ] || { echo "REPO_ROOT=$REPO_ROOT does not look like the repo root" >&2; exit 1; }

echo "==> host $(hostname) | ${NGPUS} GPU(s) | ${CPUS} CPUs | ${WORKERS} workers/GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# ---- optional: stage data to node-local disk ------------------------------------------------------------
case "$SENSOR" in
    ouster) SENSOR_DIR=LIDAR_OUSTER_Tilted_90_degree ;;
    livox)  SENSOR_DIR=LIDAR_LIVOX_Tilted ;;
    radar)  SENSOR_DIR=RADAR_Tilted ;;
    *) echo "unknown SENSOR=$SENSOR" >&2; exit 1 ;;
esac
RUN_DATA="$DATA_ROOT"
if [ "$STAGE" = "1" ]; then
    RUN_DATA="${TMPDIR:?no TMPDIR on this node}/dsert-roll"
    echo "==> staging labels + ${SENSOR_DIR} to $RUN_DATA"
    mkdir -p "$RUN_DATA"
    rsync -a "$DATA_ROOT/ImageSets" "$RUN_DATA/"
    rsync -a --prune-empty-dirs --include='*/' --include='label.pkl' --include="${SENSOR_DIR}/***" --exclude='*' \
        "$DATA_ROOT/processed_data/" "$RUN_DATA/processed_data/"
    du -sh "$RUN_DATA"
fi

# ---- config overrides -------------------------------------------------------------------------------------
SETS=(DATA_CONFIG.DATA_PATH "$RUN_DATA" DATA_CONFIG.SENSOR.NAME "$SENSOR")
if [ "$SMOKE" = "1" ]; then
    # ~1% of frames for train and val, 2 epochs: exercises data, AMP, compile, checkpointing and evaluation.
    SETS+=(DATA_CONFIG.SAMPLED_INTERVAL.train 100 DATA_CONFIG.SAMPLED_INTERVAL.test 100)
    EPOCHS="${EPOCHS:-2}"
    TAG="${TAG}_smoke"
fi
TRAIN_ARGS=(--launcher pytorch --cfg_file "$CFG" --extra_tag "$TAG" --workers "$WORKERS"
            --amp "$AMP" --find_unused_parameters --max_ckpt_save_num 5)
[ "$COMPILE" = "1" ] && TRAIN_ARGS+=(--compile)
[ -n "$EPOCHS" ] && TRAIN_ARGS+=(--epochs "$EPOCHS")
[ -n "$BATCH_PER_GPU" ] && TRAIN_ARGS+=(--batch_size "$(( BATCH_PER_GPU * NGPUS ))")
[ -n "$PRETRAINED" ] && TRAIN_ARGS+=(--pretrained_model "$PRETRAINED" --pretrained_scope "$PRETRAINED_SCOPE")
[ "$SKIP_EVAL" = "1" ] && TRAIN_ARGS+=(--skip_final_eval)
[ -n "$WANDB_PROJECT" ] && TRAIN_ARGS+=(--wandb_project "$WANDB_PROJECT")

# ---- preflight (CPU): sensor key, calibration, point/box alignment -------------------------------------
if [ "$PREFLIGHT" = "1" ]; then
    echo "==> preflight"
    python preflight_single_sensor.py --cfg_file "$CFG" --num_frames 20 \
        --plot_dir "../output/preflight/${TAG}" --set "${SETS[@]}"
fi

# ---- train ------------------------------------------------------------------------------------------------
echo "==> torchrun train_mm.py ${TRAIN_ARGS[*]} ${EXTRA_ARGS} --set ${SETS[*]}"
# shellcheck disable=SC2086
torchrun --standalone --nproc_per_node="$NGPUS" train_mm.py "${TRAIN_ARGS[@]}" $EXTRA_ARGS --set "${SETS[@]}"

OUT_DIR="$REPO_ROOT/detection/output/$(dirname "${CFG#cfgs/}")/$(basename "$CFG" .yaml)/$TAG"
echo "==> done. Outputs: $OUT_DIR"
if compgen -G "$OUT_DIR/wandb/offline-run-*" > /dev/null; then
    echo "==> wandb logged offline. To upload, on a login node run:"
    echo "    bash $REPO_ROOT/detection/tools/scripts/snellius/wandb_sync.sh $OUT_DIR"
fi
