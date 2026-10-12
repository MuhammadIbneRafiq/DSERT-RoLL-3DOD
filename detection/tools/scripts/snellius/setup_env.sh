#!/usr/bin/env bash
#
# One-time Python environment for LiDAR/radar-only training on Snellius H100 nodes.
# Run from anywhere on a login node (no GPU needed to compile; it takes ~20-30 min):
#
#   bash detection/tools/scripts/snellius/setup_env.sh
#
# Stack: Python 3.10 (Miniforge), PyTorch 2.4.1 + CUDA 12.1, spconv 2.3 (cu120 wheel), torch_scatter.
# The camera branch (mmcv-full 1.4 / mmdet / deformable attention) is NOT installed: it needs PyTorch 1.10.
#
# Override any of these with environment variables, e.g. ENV_DIR=/projects/0/<proj>/envs/dsert bash setup_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ENV_DIR="${ENV_DIR:-$HOME/envs/dsert-lidar}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniforge3}"
MODULE_STACK="${MODULE_STACK:-2023}"         # `module avail` on Snellius lists the yearly stacks
CUDA_MODULE="${CUDA_MODULE:-CUDA/12.1.1}"    # nvcc for building the repo's CUDA ops; must be CUDA 12.x

echo "==> repo: $REPO_ROOT"
echo "==> env:  $ENV_DIR"

module purge
module load "$MODULE_STACK"
module load "$CUDA_MODULE" || { echo "Could not load $CUDA_MODULE. Run 'module spider CUDA' and set CUDA_MODULE." >&2; exit 1; }
nvcc --version | tail -n 2

# --- Python 3.10 via Miniforge (conda-forge only) ---------------------------------------------------------
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    echo "==> installing Miniforge into $CONDA_DIR"
    installer="$(mktemp -d)/miniforge.sh"
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O "$installer"
    bash "$installer" -b -p "$CONDA_DIR"
fi
if [ ! -d "$ENV_DIR" ]; then
    "$CONDA_DIR/bin/conda" create -y -p "$ENV_DIR" python=3.10
fi
# shellcheck disable=SC1091
source "$CONDA_DIR/bin/activate" "$ENV_DIR"

# --- Python packages ----------------------------------------------------------------------------------------
# numpy is pinned to what TensorFlow 2.12 (Waymo eval) accepts; torch 2.4 and spconv work with it.
pip install --upgrade pip wheel ninja "setuptools<70"
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy==1.23.5"
pip install spconv-cu120
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install numba tensorboardX easydict pyyaml scikit-image tqdm pandas tabulate kornia scipy matplotlib \
    "opencv-python-headless<4.10" "numpy==1.23.5"

# Waymo AP for the evaluation at the end of training. If this fails, train with SKIP_EVAL=1 and evaluate elsewhere.
pip install waymo-open-dataset-tf-2-12-0==1.6.4 "numpy==1.23.5" \
    || echo "WARNING: waymo-open-dataset did not install; final evaluation will fail (use SKIP_EVAL=1)."

# --- Repo CUDA ops ------------------------------------------------------------------------------------------
export TORCH_CUDA_ARCH_LIST="9.0"   # H100 only; add "8.0" for A100 nodes
export MAX_JOBS="${MAX_JOBS:-8}"
echo "==> building al3d_utils CUDA ops"
( cd "$REPO_ROOT/utils" && python setup.py develop )
echo "==> installing al3d_det"
( cd "$REPO_ROOT/detection" && python setup.py develop )

# --- Import check (CPU) -------------------------------------------------------------------------------------
python - <<'EOF'
import torch, spconv, torch_scatter
from al3d_utils.ops.iou3d_nms import iou3d_nms_utils
from al3d_utils.ops.pointnet2.pointnet2_stack import pointnet2_utils
from al3d_det.models import build_network, __all__ as models
from al3d_det.datasets import DSERTSingleSensorDataset
assert 'CenterPointSingle' in models
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| spconv', spconv.__version__)
try:
    import waymo_open_dataset  # noqa: F401
    print('waymo eval: available')
except ImportError:
    print('waymo eval: NOT available (use SKIP_EVAL=1)')
print('Install OK')
EOF
