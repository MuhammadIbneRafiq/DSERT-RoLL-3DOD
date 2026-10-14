# Shared Snellius settings, sourced by setup_env.sh, build_ops.sbatch, train_single_sensor.sbatch and wandb_sync.sh.
# Override any of these by exporting them before running a script (or via sbatch --export=ALL,NAME=value).

ENV_DIR="${ENV_DIR:-$HOME/envs/dsert-lidar}"     # conda env with PyTorch etc.
CONDA_DIR="${CONDA_DIR:-$HOME/miniforge3}"       # Miniforge install (conda-forge only)

# Module stack + CUDA toolkit providing nvcc for the repo's CUDA ops. Any CUDA 12.x works with the cu121 PyTorch
# wheel. If CUDA_MODULE is set, only that is tried; otherwise the candidates below are tried in order.
SNELLIUS_CUDA_CANDIDATES=("2023 CUDA/12.1.1" "2024 CUDA/12.6.0" "2025 CUDA/12.8.0")

load_cuda_module() {
    local candidates=("${SNELLIUS_CUDA_CANDIDATES[@]}")
    [ -n "${CUDA_MODULE:-}" ] && candidates=("${MODULE_STACK:-2023} $CUDA_MODULE")
    local pair stack cuda
    for pair in "${candidates[@]}"; do
        read -r stack cuda <<< "$pair"
        module purge > /dev/null 2>&1
        if module load "$stack" > /dev/null 2>&1 && module load "$cuda" > /dev/null 2>&1; then
            echo "==> modules: $stack $cuda ($(nvcc --version | grep -o 'release [0-9.]*'))"
            return 0
        fi
    done
    echo "ERROR: none of the CUDA modules loaded (${candidates[*]})." >&2
    echo "       Run 'module avail CUDA' (after 'module load 2024' or similar) and set MODULE_STACK and CUDA_MODULE." >&2
    return 1
}

activate_env() {
    if [ ! -f "$CONDA_DIR/bin/activate" ] || [ ! -d "$ENV_DIR" ]; then
        echo "ERROR: environment $ENV_DIR not found. Run detection/tools/scripts/snellius/setup_env.sh on a login node first." >&2
        return 1
    fi
    # shellcheck disable=SC1091
    source "$CONDA_DIR/bin/activate" "$ENV_DIR"
}

wandb_logged_in() {
    [ -n "${WANDB_API_KEY:-}" ] || grep -qs "api.wandb.ai" "$HOME/.netrc"
}
