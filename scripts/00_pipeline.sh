#!/bin/bash
#SBATCH --job-name=00_pipeline
#SBATCH --output=log/00_pipeline_%j.out
#SBATCH --error=log/00_pipeline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=mps:l40:2
#SBATCH --time=48:00:00

# Full MCQGen Pipeline: 01+02 CPU → gen_wrapper (P1→P2→P3→P4→P5-P8) → eval_wrapper (eval_overall→eval_iwf)
# Steps 01-02: CPU (indexing + retrieval)
# gen_wrapper:   GPU L40s — Qwen2.5-14B-Instruct (model loaded ONCE)
# eval_wrapper: GPU L40s — Gemma-3-12b-it      (model loaded ONCE)

set -euo pipefail

PROJECT_ROOT="/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen"
REQUIRED_VRAM=30000

cleanup() {
    local rc=$?
    echo "[INFO] cleanup rc=$rc at $(date)"
    if [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ]; then
        rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" 2>/dev/null || true
    fi
    if [ -n "${CUDA_MPS_LOG_DIRECTORY:-}" ]; then
        rm -rf "${CUDA_MPS_LOG_DIRECTORY}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[INFO] start at $(date)"
echo "[INFO] hostname=$(hostname)"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"

module clear -f
module load slurm/slurm/24.11
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

source /datastore/uittogether/tools/miniconda3/etc/profile.d/conda.sh
set +u
conda activate /datastore/uittogether/tools/miniconda3/envs/cs431mcq
set -u

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

# ── GPU check ─────────────────────────────────────────────────────
unset CUDA_VISIBLE_DEVICES

set +e
CHECK_OUT=$("$PROJECT_ROOT/scripts/gpu_check.sh" "$REQUIRED_VRAM" "$SLURM_JOB_ID" 2>&1)
EXIT_CODE=$?
set -e

echo "[INFO] gpu_check exit_code=$EXIT_CODE"
echo "[INFO] gpu_check output=$CHECK_OUT"

if [ "$EXIT_CODE" -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0
elif [ "$EXIT_CODE" -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1
elif [ "$EXIT_CODE" -ne 0 ]; then
    echo "[ERROR] gpu_check.sh returned unexpected: $EXIT_CODE"
    exit "$EXIT_CODE"
fi

BEST_GPU="$CHECK_OUT"
echo "[INFO] BEST_GPU=$BEST_GPU"

# ── MPS setup ────────────────────────────────────────────────────
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-job${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-job${SLURM_JOB_ID}"
rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
export CUDA_VISIBLE_DEVICES="${BEST_GPU}"

echo "[INFO] launching pipeline at $(date)"

# # ── Step 01: Indexing (CPU) ──────────────────────────────────────
# echo "[INFO] Step 01 — Indexing | CPU"
# python -u src/gen/indexing.py
# echo "[INFO] Step 01 done"

# # ── Step 02: Retrieval (CPU) ──────────────────────────────────────
# echo "[INFO] Step 02 — Retrieval | CPU"
# python -u src/gen/retrieval.py
# echo "[INFO] Step 02 done"

# # ── Steps 03-06: gen_wrapper (Qwen2.5-14B, model loaded ONCE) ────
# echo "[INFO] Steps 03-06 — gen_wrapper (P1→P2→P3→P4→P5-P8) | GPU $BEST_GPU"
# python -u src/gen/gen_wrapper.py
echo "[INFO] Steps 03-06 done"

# ── Steps 07-08: eval_wrapper (Gemma-3-12b-it, model loaded ONCE) ──
echo "[INFO] Steps 07-08 — eval_wrapper (eval_overall→eval_iwf) | GPU $BEST_GPU"
python -u src/eval/eval_wrapper.py
echo "[INFO] Steps 07-08 done"

echo "[INFO] pipeline complete at $(date)"
