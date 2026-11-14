#!/bin/bash
#
# One-time MAN TruckScenes preparation (CPU only), after download_truckscenes.sbatch:
#   1) install the official devkit (git submodule truckscenes-devkit) into the training env;
#   2) build the info files (tools/create_truckscenes_infos.py: official train/val split, sweep chains, GT boxes from
#      the devkit) into <DATA_ROOT>/al3d_infos/<VERSION>/;
#   3) run the data checks (tools/check_truckscenes_data.py) against the devkit; the job fails if any check fails.
#
#   git submodule update --init truckscenes-devkit        # once, in the repository on Snellius
#   sbatch --export=ALL,DATA_ROOT=/scratch-shared/$USER/truckscenes/man-truckscenes \
#       detection/tools/scripts/snellius/prepare_truckscenes.sbatch
#
# DATA_ROOT is the devkit dataroot: the folder holding samples/, sweeps/ and v1.2-trainval/.
# Re-running rebuilds the info files (a few minutes). CHECK_SAMPLES / CHECK_ITEMS set how many samples checks 1-2 and
# 4 look at; check 3 (GT through the official evaluation) always uses the whole val split.
#
#SBATCH --job-name=truckscenes-prep
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the TruckScenes dataroot (the folder containing samples/, sweeps/, v1.2-trainval/)}"
VERSION="${VERSION:-v1.2-trainval}"
INFO_DIR="${INFO_DIR:-$DATA_ROOT/al3d_infos/$VERSION}"
CHECK_SAMPLES="${CHECK_SAMPLES:-40}"
CHECK_ITEMS="${CHECK_ITEMS:-16}"

SNELLIUS_DIR="$REPO_ROOT/detection/tools/scripts/snellius"
[ -f "$SNELLIUS_DIR/snellius_env.sh" ] || { echo "Submit from the repository root (REPO_ROOT=$REPO_ROOT)" >&2; exit 1; }
[ -f "$REPO_ROOT/truckscenes-devkit/setup.cfg" ] || {
    echo "ERROR: truckscenes-devkit submodule is empty. Run: git submodule update --init truckscenes-devkit" >&2; exit 1; }
[ -d "$DATA_ROOT/$VERSION" ] || { echo "ERROR: $DATA_ROOT/$VERSION not found (wrong DATA_ROOT?)" >&2; exit 1; }
# shellcheck disable=SC1091
source "$SNELLIUS_DIR/snellius_env.sh"
activate_env
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4

# ---- 1) devkit --------------------------------------------------------------------------------------------
# --no-deps + its three small dependencies: the devkit does not pin numpy, so pip must not touch the env's numpy/torch.
echo "==> [1/3] installing truckscenes-devkit (editable) into $(python -c 'import sys; print(sys.prefix)')"
python -m pip install --quiet --no-deps -e "$REPO_ROOT/truckscenes-devkit"
python -m pip install --quiet "pyquaternion>=0.9.5" pypcd4 tqdm
python -c "import truckscenes, numpy; print('truckscenes devkit at', truckscenes.__path__[0], '| numpy', numpy.__version__)"

# ---- 2) info files ----------------------------------------------------------------------------------------
echo "==> [2/3] info files -> $INFO_DIR"
cd "$REPO_ROOT/detection/tools"
python create_truckscenes_infos.py --dataroot "$DATA_ROOT" --version "$VERSION" --out_dir "$INFO_DIR"
ls -la "$INFO_DIR"

# ---- 3) data checks against the devkit --------------------------------------------------------------------
echo "==> [3/3] data checks"
python check_truckscenes_data.py --dataroot "$DATA_ROOT" --version "$VERSION" --info_dir "$INFO_DIR" \
    --num_samples "$CHECK_SAMPLES" --num_items "$CHECK_ITEMS"
echo "==> done. Train with detection/tools/scripts/snellius/submit_truckscenes.sh (SMOKE=1 first)."
