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

# Full MCQGen Pipeline — 8 bước chạy tuần tự
# Mỗi script Python tự cleanup VRAM sau khi chạy xong (gc.collect + torch.cuda.empty_cache)
# gpu_memory_utilization=0.35 trong common.py để fit 25-30 GiB VRAM

set -euo pipefail

PROJECT_ROOT="/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen"
REQUIRED_VRAM=25000

cleanup_mps() {
    local rc=$?
    echo "[INFO] cleanup rc=$rc at $(date)"
    if [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ]; then
        rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" 2>/dev/null || true
    fi
    if [ -n "${CUDA_MPS_LOG_DIRECTORY:-}" ]; then
        rm -rf "${CUDA_MPS_LOG_DIRECTORY}" 2>/dev/null || true
    fi
}
trap cleanup_mps EXIT

echo "[INFO] start at $(date)"
echo "[INFO] hostname=$(hostname)"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"

module clear -f
module load slurm/slurm/24.11
source /datastore/uittogether/tools/miniconda3/etc/profile.d/conda.sh
set +u
conda activate /datastore/uittogether/tools/miniconda3/envs/cs431mcq
set -u

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

# ── Acquire GPU ────────────────────────────────────────────────
unset CUDA_VISIBLE_DEVICES
set +e
CHECK_OUT=$("$PROJECT_ROOT/scripts/gpu_check.sh" "$REQUIRED_VRAM" "${SLURM_JOB_ID:-$$}" 2>&1)
EXIT_CODE=$?
set -e

if [ "$EXIT_CODE" -eq 10 ]; then echo "$CHECK_OUT"; exit 0; fi
if [ "$EXIT_CODE" -eq 11 ]; then echo "$CHECK_OUT"; exit 1; fi
if [ "$EXIT_CODE" -ne 0 ]; then echo "[ERROR] gpu_check.sh: $EXIT_CODE"; exit "$EXIT_CODE"; fi

BEST_GPU="$CHECK_OUT"
export CUDA_VISIBLE_DEVICES="$BEST_GPU"
echo "[INFO] BEST_GPU=$BEST_GPU"

# ── Step 01: Indexing (CPU) ────────────────────────────────────
echo "[INFO] Step 01: Indexing (CPU)"
python -u src/gen/indexing.py
echo "[INFO] Step 01 done"

# ── Step 02: Retrieval (CPU) ───────────────────────────────────
echo "[INFO] Step 02: Retrieval (CPU)"
python -u src/gen/retrieval.py
echo "[INFO] Step 02 done"

# ── Step 03: P1 Gen Stem ───────────────────────────────────────
echo "[INFO] Step 03: P1 Gen Stem (GPU $BEST_GPU)"
python -u src/gen/p1_gen_stem.py
echo "[INFO] Step 03 done"

# ── Step 04: P2+P3 Refine ──────────────────────────────────────
echo "[INFO] Step 04: P2+P3 Refine (GPU $BEST_GPU)"
python -u src/gen/p2_p3_refine.py
echo "[INFO] Step 04 done"

# ── Step 05: P4 Candidates ────────────────────────────────────
echo "[INFO] Step 05: P4 Candidates (GPU $BEST_GPU)"
python -u src/gen/p4_candidates.py
echo "[INFO] Step 05 done"

# ── Step 06: P5-P8 CoT ─────────────────────────────────────────
echo "[INFO] Step 06: P5-P8 CoT (GPU $BEST_GPU)"
python -u src/gen/p5_p8_cot.py
echo "[INFO] Step 06 done"

# ── Step 07: Eval Overall ────────────────────────────────────────
echo "[INFO] Step 07: Eval Overall (GPU $BEST_GPU)"
python -u src/eval/eval_overall.py
echo "[INFO] Step 07 done"

# ── Step 08: Eval IWF ───────────────────────────────────────────
echo "[INFO] Step 08: Eval IWF (GPU $BEST_GPU)"
python -u src/eval/eval_iwf.py
echo "[INFO] Step 08 done"

echo "[INFO] pipeline complete at $(date)"
