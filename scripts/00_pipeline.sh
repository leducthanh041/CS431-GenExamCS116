#!/bin/bash
#SBATCH --job-name=00_pipeline
#SBATCH --output=/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen/log/00_pipeline_%j.out
#SBATCH --error=/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen/log/00_pipeline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=mps:l40:2
#SBATCH --time=48:00:00

# Full MCQGen Pipeline — 9 bước chạy tuần tự
# Step 09: Explanation Generation với hybrid retrieval + web search
# gpu_memory_utilization=dynamic (detect VRAM free trong common.py)

REQUIRED_VRAM=40000

set -e

module clear -f
module load slurm/slurm/24.11
source /datastore/uittogether/tools/miniconda3/etc/profile.d/conda.sh

unset CUDA_VISIBLE_DEVICES
PROJECT_ROOT="/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen"
CHECK_OUT=$("$PROJECT_ROOT/scripts/gpu_check.sh" $REQUIRED_VRAM $SLURM_JOB_ID) || true
EXIT_CODE=$?

if [ $EXIT_CODE -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 1
elif [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1
fi

BEST_GPU=$CHECK_OUT
echo "🚀 Job $SLURM_JOB_ID — Full Pipeline Steps 01-09 | GPU $BEST_GPU"

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
export CUDA_VISIBLE_DEVICES=$BEST_GPU
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

# # ── Step 01: Indexing (CPU) ───────────────────────────────────────────────
# echo "[INFO] Step 01: Indexing (CPU)"
# python -u src/gen/indexing.py
echo "[INFO] Step 01 done"

# ── Step 02: Retrieval (CPU) ──────────────────────────────────────────────
echo "[INFO] Step 02: Retrieval (CPU)"
python -u src/gen/retrieval.py
echo "[INFO] Step 02 done"

# ── Step 03: P1 Gen Stem ────────────────────────────────────────────────
echo "[INFO] Step 03: P1 Gen Stem (GPU $BEST_GPU)"
python -u src/gen/p1_gen_stem.py
echo "[INFO] Step 03 done"

# ── Step 04: P2+P3 Refine ──────────────────────────────────────────────
echo "[INFO] Step 04: P2+P3 Refine (GPU $BEST_GPU)"
python -u src/gen/p2_p3_refine.py
echo "[INFO] Step 04 done"

# ── Step 05: P4 Candidates ──────────────────────────────────────────────
echo "[INFO] Step 05: P4 Candidates (GPU $BEST_GPU)"
python -u src/gen/p4_candidates.py
echo "[INFO] Step 05 done"

# ── Step 06: P5-P8 CoT ──────────────────────────────────────────────────
echo "[INFO] Step 06: P5-P8 CoT (GPU $BEST_GPU)"
python -u src/gen/p5_p8_cot.py
echo "[INFO] Step 06 done"

# ── Step 07: Eval Overall ────────────────────────────────────────────────
echo "[INFO] Step 07: Eval Overall (GPU $BEST_GPU)"
python -u src/eval/eval_overall.py
echo "[INFO] Step 07 done"

# ── Step 08: Eval IWF ───────────────────────────────────────────────────
echo "[INFO] Step 08: Eval IWF (GPU $BEST_GPU)"
python -u src/eval/eval_iwf.py
echo "[INFO] Step 08 done"

# ── Step 09: Explanation Generation ────────────────────────────────────
echo "[INFO] Step 09: Explanation Generation (GPU $BEST_GPU)"
python -u src/gen/explain_mcq.py
echo "[INFO] Step 09 done"

rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
echo "✅ Full pipeline complete at $(date)"
