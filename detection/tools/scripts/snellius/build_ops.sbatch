#!/bin/bash
#
# Step 2 of 2 of the one-time setup: compile the repo's CUDA ops for H100 and self-test the model on the GPU.
# Run after setup_env.sh, from the repository root:
#
#   sbatch detection/tools/scripts/snellius/build_ops.sbatch
#
# Kept under 1 hour so Slurm can place it on the nodes reserved for short jobs (shorter queue). Re-run it
# whenever the C++/CUDA sources under utils/al3d_utils/ops change.
#
#SBATCH --job-name=dsert-build
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:59:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
SNELLIUS_DIR="$REPO_ROOT/detection/tools/scripts/snellius"
[ -f "$SNELLIUS_DIR/snellius_env.sh" ] || { echo "Submit from the repository root (REPO_ROOT=$REPO_ROOT)" >&2; exit 1; }
# shellcheck disable=SC1091
source "$SNELLIUS_DIR/snellius_env.sh"

load_cuda_module
activate_env
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export TORCH_CUDA_ARCH_LIST="9.0"                  # H100
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-16}"
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"

echo "==> building al3d_utils CUDA ops (iou3d_nms, roiaware/roipoint pool, pointnet2)"
( cd "$REPO_ROOT/utils" && python setup.py develop )
echo "==> installing al3d_det"
( cd "$REPO_ROOT/detection" && python setup.py develop )

echo "==> import check"
python - <<'EOF'
import torch, spconv, torch_scatter
from al3d_utils.ops.iou3d_nms import iou3d_nms_utils
from al3d_utils.ops.pointnet2.pointnet2_stack import pointnet2_utils
from al3d_det.models import __all__ as models
from al3d_det.datasets import DSERTSingleSensorDataset  # noqa: F401
assert 'CenterPointSingle' in models
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| spconv', spconv.__version__)
try:
    import waymo_open_dataset  # noqa: F401
    print('waymo eval: available')
except ImportError:
    print('waymo eval: NOT available (train with SKIP_EVAL=1)')
EOF

echo "==> GPU self-test: training steps with fp16 AMP + torch.compile, then inference, on random points"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}"
cd "$REPO_ROOT/detection/tools"
python gpu_selftest.py --cfg_file cfgs/det_model_cfgs/dsert/lidar_only.yaml --amp fp16 --compile

if wandb_logged_in; then echo "==> wandb: logged in"; else echo "==> wandb: NOT logged in (run 'wandb login' on a login node)"; fi
echo "==> Setup complete. Next: the smoke test (see train_single_sensor.sbatch)."
