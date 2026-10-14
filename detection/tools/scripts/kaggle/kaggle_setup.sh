#!/usr/bin/env bash
#
# Install the single-sensor model on a Kaggle GPU notebook (T4 recommended; Internet must be ON in the notebook
# settings). Uses Kaggle's preinstalled PyTorch, adds spconv, compiles the repo's CUDA ops for the notebook's GPU.
# Takes ~5-10 min. Run from a notebook cell:
#
#   !bash /kaggle/working/DSERT-RoLL-3DOD/detection/tools/scripts/kaggle/kaggle_setup.sh
#
# The camera branch and the Waymo-AP evaluation are not installed; the analysis scripts do not need them.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

read -r TORCH_VER CUDA_VER GPU_CC GPU_NAME < <(python -c "
import torch
assert torch.cuda.is_available(), 'No GPU: set Accelerator to GPU T4 in the notebook settings'
print(torch.__version__.split('+')[0], torch.version.cuda, '%d.%d' % torch.cuda.get_device_capability(), torch.cuda.get_device_name(0).replace(' ', '_'))")
echo "==> torch $TORCH_VER | cuda $CUDA_VER | GPU $GPU_NAME (sm_${GPU_CC/./})"

pip install -q "setuptools<70" ninja easydict numba tensorboardX scikit-image tqdm pyyaml

# spconv ships prebuilt kernels per CUDA major version.
case "$CUDA_VER" in
    12.*) pip install -q spconv-cu120 ;;
    11.8) pip install -q spconv-cu118 ;;
    *)    echo "ERROR: no spconv wheel chosen for CUDA $CUDA_VER" >&2; exit 1 ;;
esac

# torch_scatter is optional (the model falls back to a pure-PyTorch mean); take a prebuilt wheel only.
PYG_TORCH="$(echo "$TORCH_VER" | cut -d. -f1,2).0"
PYG_CU="cu$(echo "$CUDA_VER" | tr -d .)"
pip install -q --only-binary=:all: torch_scatter -f "https://data.pyg.org/whl/torch-${PYG_TORCH}+${PYG_CU}.html" \
    || echo "==> no torch_scatter wheel for torch ${PYG_TORCH}+${PYG_CU}; using the built-in fallback"

# Compile the repo's CUDA ops (box IoU/NMS, ROI pooling, pointnet2) for this GPU only.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
command -v nvcc > /dev/null || { echo "ERROR: nvcc not found under $CUDA_HOME" >&2; exit 1; }
export TORCH_CUDA_ARCH_LIST="$GPU_CC"
export MAX_JOBS=4
( cd "$REPO_ROOT/utils" && python setup.py develop > /tmp/build_utils.log 2>&1 ) \
    || { tail -40 /tmp/build_utils.log; exit 1; }
( cd "$REPO_ROOT/detection" && python setup.py develop > /tmp/build_det.log 2>&1 ) \
    || { tail -40 /tmp/build_det.log; exit 1; }

python -c "
import spconv
from al3d_utils.ops.iou3d_nms import iou3d_nms_utils
from al3d_det.models import __all__ as models
assert 'CenterPointSingle' in models
print('==> setup OK | spconv', spconv.__version__)"
