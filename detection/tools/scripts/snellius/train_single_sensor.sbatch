#!/bin/bash
#
# Single-sensor (LiDAR or radar) CenterPoint + LoGoHead on DSERT-RoLL. 1 node, 1 H100 (a quarter node:
# 16 cores, 180 GiB RAM). No cameras, no radar branch. Mixed precision + torch.compile on the dense layers + TF32.
#
# One-time setup first: setup_env.sh on a login node, then `sbatch .../build_ops.sbatch`.
# Submit from the repository root:
#
#   # 1) smoke test: ~1% of the data, 2 epochs. Do this first and note the it/s it reports.
#   #    (a time limit <= 1 hour puts it on the nodes reserved for short jobs, so it starts sooner)
#   sbatch --time=01:00:00 --export=ALL,DATA_ROOT=/path/to/dsert_roll,SMOKE=1 detection/tools/scripts/snellius/train_single_sensor.sbatch
#
#   # 2) full pretraining run (24 h limit by default; a full 20-epoch run takes roughly 7-9 h on one H100)
#   sbatch --export=ALL,DATA_ROOT=/path/to/dsert_roll,TAG=ouster_pretrain detection/tools/scripts/snellius/train_single_sensor.sbatch
#
#   # 3) later: radar fine-tuning from the Ouster checkpoint, and the radar-from-scratch baseline
#   sbatch --export=ALL,DATA_ROOT=...,SENSOR=radar,TAG=radar_ft_from_ouster,PRETRAINED=<ckpt.pth> ...
#   sbatch --export=ALL,DATA_ROOT=...,SENSOR=radar,TAG=radar_scratch ...
#
# Data: keep the dataset permanently in your home or project space (NOT /scratch-shared or /scratch-local, which
# delete files after 14 / 6 days). The job asks for a node with local NVMe (--constraint=scratch-node; only some
# H100 nodes have one), copies the labels and this sensor's frames there, and trains from that copy. The copy
# is deleted automatically when the job ends; checkpoints and logs are written to detection/output/ in the repo.
#
# Re-submitting with the same TAG resumes from the newest checkpoint and continues the same wandb run.
# wandb logs to WANDB_PROJECT (default dsert-single-sensor; empty turns it off) after a one-time `wandb login` on a
# login node. If the node cannot reach wandb it logs offline; upload with wandb_sync.sh from a login node.
#
# Single GPU only: the config's batch 16 at LR 0.001 is the paper's global batch, and the comparison between
# runs relies on every run using the same setup.
#
# Keep --time realistic: Snellius reserves (time limit x 192 SBU/h) from the budget when a job starts and cancels it
# ("CANCELLED by 0", AdminComment reason=budget) if running jobs' reservations would exceed the budget. A 5-day limit
# reserves 23,040 SBU even if the run takes 8 h. If a run ever needs longer, resubmit with the same TAG to resume.
#
#SBATCH --job-name=dsert-single
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
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the DSERT-RoLL root (the folder containing ImageSets/ and processed_data/)}"

SENSOR="${SENSOR:-ouster}"                       # ouster | livox | radar
CFG="${CFG:-cfgs/det_model_cfgs/dsert/lidar_only.yaml}"
TAG="${TAG:-${SENSOR}_$(date +%Y%m%d)}"
AMP="${AMP:-fp16}"                               # fp16 (GradScaler) | bf16 | none
COMPILE="${COMPILE:-1}"                          # 1 = torch.compile the dense layers
EPOCHS="${EPOCHS:-}"                             # empty = OPTIMIZATION.NUM_EPOCHS from the config
PRETRAINED="${PRETRAINED:-}"                     # checkpoint to initialise from (e.g. the Ouster run's last epoch)
PRETRAINED_SCOPE="${PRETRAINED_SCOPE:-encoder}"  # encoder = copy 3D/BEV backbones only (heads fresh) | all
SMOKE="${SMOKE:-0}"                              # 1 = tiny subset, 2 epochs: checks the whole pipeline end to end
PREFLIGHT="${PREFLIGHT:-1}"                      # 1 = run preflight_single_sensor.py first and stop if it fails
SKIP_EVAL="${SKIP_EVAL:-0}"                      # 1 = skip the Waymo-AP evaluation after training
STAGE="${STAGE:-auto}"                           # auto = copy data to node-local NVMe when the node has it | 1 | 0
WANDB_PROJECT="${WANDB_PROJECT-dsert-single-sensor}"   # Weights & Biases project; WANDB_PROJECT= (empty) turns it off
EXTRA_ARGS="${EXTRA_ARGS:-}"                     # anything else for train_mm.py (before --set)

SNELLIUS_DIR="$REPO_ROOT/detection/tools/scripts/snellius"
[ -f "$SNELLIUS_DIR/snellius_env.sh" ] || { echo "Submit from the repository root (REPO_ROOT=$REPO_ROOT)" >&2; exit 1; }
# shellcheck disable=SC1091
source "$SNELLIUS_DIR/snellius_env.sh"

# ---- environment ------------------------------------------------------------------------------------------
load_cuda_module
activate_env

NGPUS="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
if [ "$NGPUS" != "1" ]; then
    echo "ERROR: this job is set up for exactly 1 GPU, got $NGPUS. Submit without changing --gpus-per-node." >&2
    exit 1
fi
CPUS="${SLURM_CPUS_PER_TASK:-16}"
WORKERS="${WORKERS:-$(( CPUS - 2 ))}"            # dataloader workers; leave 2 cores for the training process
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}"   # reuse compiled kernels across jobs
# spconv 2.3 calls the deprecated torch.cuda.amp.custom_fwd/bwd inside its own package (harmless, but printed by every
# dataloader worker). Silence exactly that message; inherited by all processes of the job. TF: hide C++ info logs.
export PYTHONWARNINGS='ignore:`torch.cuda.amp.custom_:FutureWarning'
export TF_CPP_MIN_LOG_LEVEL=2

# wandb reads its API key from ~/.netrc (written by `wandb login` on a login node; home is shared with
# compute nodes) or from WANDB_API_KEY. Without either, log offline rather than failing the job.
if [ -n "$WANDB_PROJECT" ] && ! wandb_logged_in; then
    echo "WARNING: wandb is not logged in for $USER. Run 'wandb login' once on a login node. Logging offline for now." >&2
    export WANDB_MODE=offline
fi

cd "$REPO_ROOT/detection/tools"
echo "==> host $(hostname) | job ${SLURM_JOB_ID:-} | ${NGPUS} GPU | ${CPUS} CPUs | ${WORKERS} dataloader workers"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

case "$SENSOR" in
    ouster) SENSOR_DIR=LIDAR_OUSTER_Tilted_90_degree ;;
    livox)  SENSOR_DIR=LIDAR_LIVOX_Tilted ;;
    radar)  SENSOR_DIR=RADAR_Tilted ;;
    *) echo "unknown SENSOR=$SENSOR" >&2; exit 1 ;;
esac
if [ "$SMOKE" = "1" ]; then
    EPOCHS="${EPOCHS:-2}"
    TAG="${TAG}_smoke"
fi

# ---- preflight (CPU), on the original data: sensor key, calibration, point/box alignment ------------------
if [ "$PREFLIGHT" = "1" ]; then
    echo "==> preflight"
    python preflight_single_sensor.py --cfg_file "$CFG" --num_frames 20 --plot_dir "../output/preflight/${TAG}" \
        --set DATA_CONFIG.DATA_PATH "$DATA_ROOT" DATA_CONFIG.SENSOR.NAME "$SENSOR"
fi

# ---- stage data to node-local NVMe ($TMPDIR is on /scratch-node only with --constraint=scratch-node) ----
RUN_DATA="$DATA_ROOT"
ON_NVME=0
case "${TMPDIR:-}" in /scratch-node/*) ON_NVME=1 ;; esac
if [ "$STAGE" = "1" ] || { [ "$STAGE" = "auto" ] && [ "$ON_NVME" = "1" ]; }; then
    if [ "$ON_NVME" != "1" ]; then
        # Without node-local scratch, $TMPDIR is /scratch-local, which is the same shared GPFS as the source.
        echo "WARNING: no node-local scratch on this node (TMPDIR=${TMPDIR:-unset}); reading data in place." >&2
    else
        RUN_DATA="$TMPDIR/dsert-roll"
        echo "==> staging labels + ${SENSOR_DIR} to $RUN_DATA"
        mkdir -p "$RUN_DATA"
        start=$(date +%s)
        rsync -a "$DATA_ROOT/ImageSets" "$RUN_DATA/"
        rsync -a --prune-empty-dirs --include='*/' --include='label.pkl' --include="${SENSOR_DIR}/***" --exclude='*' \
            "$DATA_ROOT/processed_data/" "$RUN_DATA/processed_data/"
        echo "==> staged $(du -sh "$RUN_DATA" | cut -f1) in $(( $(date +%s) - start ))s"
    fi
fi

# ---- config overrides -------------------------------------------------------------------------------------
SETS=(DATA_CONFIG.DATA_PATH "$RUN_DATA" DATA_CONFIG.SENSOR.NAME "$SENSOR")
# ~1% of frames for train and val: exercises data, AMP, compile, checkpointing and evaluation.
[ "$SMOKE" = "1" ] && SETS+=(DATA_CONFIG.SAMPLED_INTERVAL.train 100 DATA_CONFIG.SAMPLED_INTERVAL.test 100)
TRAIN_ARGS=(--launcher pytorch --cfg_file "$CFG" --extra_tag "$TAG" --workers "$WORKERS"
            --amp "$AMP" --find_unused_parameters --max_ckpt_save_num 5)
[ "$COMPILE" = "1" ] && TRAIN_ARGS+=(--compile)
[ -n "$EPOCHS" ] && TRAIN_ARGS+=(--epochs "$EPOCHS")
[ -n "$PRETRAINED" ] && TRAIN_ARGS+=(--pretrained_model "$PRETRAINED" --pretrained_scope "$PRETRAINED_SCOPE")
[ "$SKIP_EVAL" = "1" ] && TRAIN_ARGS+=(--skip_final_eval)
[ -n "$WANDB_PROJECT" ] && TRAIN_ARGS+=(--wandb_project "$WANDB_PROJECT")

# ---- train (single process, single GPU) -------------------------------------------------------------------
echo "==> torchrun train_mm.py ${TRAIN_ARGS[*]} ${EXTRA_ARGS} --set ${SETS[*]}"
# shellcheck disable=SC2086
torchrun --standalone --nnodes=1 --nproc_per_node=1 train_mm.py "${TRAIN_ARGS[@]}" $EXTRA_ARGS --set "${SETS[@]}"

OUT_DIR="$REPO_ROOT/detection/output/$(dirname "${CFG#cfgs/}")/$(basename "$CFG" .yaml)/$TAG"
echo "==> done. Outputs: $OUT_DIR"
if compgen -G "$OUT_DIR/wandb/offline-run-*" > /dev/null; then
    echo "==> wandb logged offline. To upload, on a login node run:"
    echo "    bash $SNELLIUS_DIR/wandb_sync.sh $OUT_DIR"
fi
