#!/usr/bin/env bash
#
# Step 1 of 2 of the one-time setup: download and install the Python environment. Run on a LOGIN node,
# because it needs internet. It is download-bound (a few GB of wheels) and does no compiling, so it stays well
# under the login-node process limits. If it is interrupted, just run it again: finished steps are skipped.
#
#   bash detection/tools/scripts/snellius/setup_env.sh
#
# Step 2 compiles the repo's CUDA ops on an H100 node as a short batch job (compiling on a login node would
# run into the limits on long-running processes there):
#
#   sbatch detection/tools/scripts/snellius/build_ops.sbatch
#
# Stack: Python 3.10 (Miniforge, conda-forge only), PyTorch 2.4.1 + CUDA 12.1, spconv 2.3 (cu120 wheel),
# torch_scatter. The camera branch (mmcv-full 1.4 / mmdet / deformable attention) is not installed: it needs
# PyTorch 1.10. Paths are set in snellius_env.sh (ENV_DIR, CONDA_DIR).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/snellius_env.sh"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "NOTE: running inside a Slurm job. This step needs internet; if downloads fail, run it on a login node." >&2
fi
echo "==> env: $ENV_DIR"

# --- Python 3.10 via Miniforge ------------------------------------------------------------------------------
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    echo "==> installing Miniforge into $CONDA_DIR"
    installer="$(mktemp -d)/miniforge.sh"
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O "$installer"
    bash "$installer" -b -p "$CONDA_DIR"
fi
if [ ! -d "$ENV_DIR" ]; then
    "$CONDA_DIR/bin/conda" create -y -p "$ENV_DIR" python=3.10
fi
activate_env

# --- Python packages ----------------------------------------------------------------------------------------
# numpy is pinned to what TensorFlow 2.12 (Waymo eval) accepts; torch 2.4 and spconv work with it.
pip install --upgrade pip wheel ninja "setuptools<70"
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy==1.23.5"
pip install spconv-cu120
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install numba tensorboardX "wandb>=0.22.3,<0.27" easydict pyyaml scikit-image tqdm pandas tabulate kornia scipy matplotlib \
    scikit-learn statsmodels \
    "opencv-python-headless<4.10" "numpy==1.23.5"

# Waymo AP for the evaluation at the end of training. The wheel pins numpy==1.23 exactly, which pip treats as
# conflicting with 1.23.5, so install its only runtime dependency the metric needs (TensorFlow 2.12, CPU build:
# the metric runs on CPU) and then the wheel itself without dependencies. If this fails, train with SKIP_EVAL=1.
# TensorFlow 2.12 needs protobuf < 5; wandb 0.22.3-0.26 accepts protobuf 4 and the long API keys (0.28.1+ needs protobuf 5).
pip install "tensorflow-cpu==2.12.0" "numpy==1.23.5" "protobuf>=4.25,<5" "wandb>=0.22.3,<0.27" \
    && pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.4 \
    && python -c "from waymo_open_dataset.metrics.python import detection_metrics" \
    || echo "WARNING: waymo-open-dataset did not install; final evaluation will fail (use SKIP_EVAL=1)."

python -c "import torch, spconv, torch_scatter, wandb; print('==> torch', torch.__version__, '| cuda', torch.version.cuda, '| spconv', spconv.__version__, '| wandb', wandb.__version__)"

# --- wandb login (one time, here on the login node) ---------------------------------------------------------
# The key lands in ~/.netrc, which compute nodes share, so jobs log online without further setup.
if wandb_logged_in; then
    echo "==> wandb: logged in"
else
    echo ""
    echo "==> wandb: not logged in yet. Copy your API key from https://wandb.ai/authorize, then run:"
    echo "    source $CONDA_DIR/bin/activate $ENV_DIR && wandb login"
fi

echo ""
echo "==> Step 1 done. Next, from the repository root:  sbatch detection/tools/scripts/snellius/build_ops.sbatch"
