#!/bin/bash
#
# Two checks of the VoxelNeXt model before any real run, on 1 H100, no dataset needed (about 15-30 min):
#
#   [1/2] gpu_selftest.py on voxelnext.yaml: a few fp16 training steps at batch 16 and one inference pass on random
#         points. Shows the model runs in this environment and prints peak GPU memory.
#   [2/2] check_voxelnext_vs_openpcdet.py: our VoxelNeXt vs OpenPCDet's VoxelNeXt (commit below), same weights and
#         inputs, comparing backbone output, head outputs, training targets, every loss term and the final boxes.
#
# OpenPCDet is not installed into the environment. Its source is cloned to OPENPCDET_REF once, and only its iou3d_nms
# CUDA op is compiled, in place inside that folder. If the compute node cannot reach GitHub, clone on a login node:
#   git clone https://github.com/open-mmlab/OpenPCDet $HOME/.cache/openpcdet_ref
#
# Submit from the repository root:
#   sbatch detection/tools/scripts/snellius/check_voxelnext.sbatch
# The last lines of slurm-vnx-check-<jobid>.out say PASSED or FAILED for each check.
#
#SBATCH --job-name=vnx-check
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

# The OpenPCDet commit the port was made from (VoxelNeXt files unchanged on master as of 2026-11-14).
OPENPCDET_COMMIT="${OPENPCDET_COMMIT:-233f849829b6ac19afb8af8837a0246890908755}"
OPENPCDET_REF="${OPENPCDET_REF:-$HOME/.cache/openpcdet_ref}"
CFG="cfgs/det_model_cfgs/dsert/voxelnext.yaml"

load_cuda_module
activate_env
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-16}"
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
export PYTHONWARNINGS='ignore:`torch.cuda.amp.custom_:FutureWarning'
export TF_CPP_MIN_LOG_LEVEL=2

# ---- OpenPCDet source + its iou3d_nms op ----------------------------------------------------------------------
if [ ! -d "$OPENPCDET_REF/.git" ]; then
    echo "==> cloning OpenPCDet to $OPENPCDET_REF"
    git clone -q https://github.com/open-mmlab/OpenPCDet "$OPENPCDET_REF" || {
        echo "ERROR: clone failed (no internet on this node?). On a login node run:" >&2
        echo "  git clone https://github.com/open-mmlab/OpenPCDet $OPENPCDET_REF" >&2
        exit 1; }
fi
git -C "$OPENPCDET_REF" -c advice.detachedHead=false checkout -q "$OPENPCDET_COMMIT"
echo "==> OpenPCDet at $(git -C "$OPENPCDET_REF" rev-parse HEAD)"

IOU_DIR="$OPENPCDET_REF/pcdet/ops/iou3d_nms"
if ! compgen -G "$IOU_DIR/iou3d_nms_cuda*.so" > /dev/null; then
    echo "==> building OpenPCDet's iou3d_nms op in $IOU_DIR"
    BUILD_PY="${TMPDIR:-/tmp}/build_ref_iou3d_$$.py"
    cat > "$BUILD_PY" <<'EOF'
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(name='openpcdet_ref_iou3d_nms', cmdclass={'build_ext': BuildExtension},
      ext_modules=[CUDAExtension('iou3d_nms_cuda', ['src/iou3d_cpu.cpp', 'src/iou3d_nms_api.cpp',
                                                    'src/iou3d_nms.cpp', 'src/iou3d_nms_kernel.cu'])])
EOF
    ( cd "$IOU_DIR" && python "$BUILD_PY" build_ext --inplace > "$IOU_DIR/build.log" 2>&1 ) || {
        echo "ERROR: build failed, see $IOU_DIR/build.log" >&2; tail -30 "$IOU_DIR/build.log" >&2; exit 1; }
    rm -f "$BUILD_PY"
fi

cd "$REPO_ROOT/detection/tools"
set +e
echo
echo "==> [1/2] GPU self-test of VoxelNeXt (random points, batch 16, fp16)"
python gpu_selftest.py --cfg_file "$CFG" --amp fp16 --batch_size 16 --steps 6
SELFTEST=$?
echo
echo "==> [2/2] our VoxelNeXt vs OpenPCDet's"
python check_voxelnext_vs_openpcdet.py --cfg "$CFG" --ref_root "$OPENPCDET_REF" --trials 3
EQUIV=$?
set -e

echo
echo "==> self-test:              $([ $SELFTEST = 0 ] && echo PASSED || echo "FAILED (exit $SELFTEST)")"
echo "==> matches OpenPCDet:      $([ $EQUIV = 0 ] && echo PASSED || echo "FAILED (exit $EQUIV)")"
[ $SELFTEST = 0 ] && [ $EQUIV = 0 ]
