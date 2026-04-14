#!/bin/bash
#SBATCH --job-name=09_explain
#SBATCH --output=log/09_explain_%j.out
#SBATCH --error=log/09_explain_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=mps:l40:2        # Use MPS (shared GPU) — avoids QOSMaxGRESPerUser limit
#SBATCH --time=12:00:00

# Step 09: Explanation Generation
# Model: Qwen2.5-14B-Instruct (vLLM)

# GPU requirement: wait until a GPU has >= 40000 MiB free VRAM.
# gpu_check.sh loops indefinitely (kill→wait→verify) until condition is met.
REQUIRED_VRAM=36000

# IMPORTANT: No "set -e" — gpu_check waits indefinitely and must NOT abort on errors.
# All error handling uses explicit exit codes below.

# ── Bootstrap: print immediately so log is never empty ─────────────────────────
echo "[$(date '+%H:%M:%S')] Job $SLURM_JOB_ID starting on $(hostname)..."
module clear -f 2>/dev/null || true
module load slurm/slurm/24.11 2>&1 | head -5 || true
source /datastore/uittogether/tools/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate cs431mcq 2>&1 | head -3 || true
echo "[$(date '+%H:%M:%S')] Environment ready"

unset CUDA_VISIBLE_DEVICES
PROJECT_ROOT="/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen"

# ── Wait for GPU ──────────────────────────────────────────────────────────────
# gpu_check.sh:
#   - Ranks GPUs by free VRAM (freest first)
#   - For each GPU: if already enough → use immediately
#   - If not: kill OUR processes → wait → VERIFY memory freed → retry
#   - If still insufficient after kill: try --gpu-reset → wait longer
#   - Loops INDEFINITELY — returns GPU index on success
#   - Exit 10 = unexpected (signal requeue), Exit 11 = system error
echo "[$(date '+%H:%M:%S')] Waiting for GPU with ${REQUIRED_VRAM} MiB free VRAM..."
echo "[$(date '+%H:%M:%S')] (Will wait indefinitely — looping kill→wait→verify until GPU available)"

CHECK_OUT=$("$PROJECT_ROOT/scripts/gpu_check.sh" "$REQUIRED_VRAM" 2>&1)
EXIT_CODE=$?

echo "$CHECK_OUT"

if [ $EXIT_CODE -ne 0 ]; then
    # Non-zero = system error or unexpected state → requeue
    echo "[$(date '+%H:%M:%S')] gpu_check exit $EXIT_CODE — requeueing job $SLURM_JOB_ID"
    scontrol requeue $SLURM_JOB_ID
    exit 0
fi

BEST_GPU=$(echo "$CHECK_OUT" | tail -1 | tr -d '[:space:]')
echo "[$(date '+%H:%M:%S')] Acquired GPU $BEST_GPU with >= ${REQUIRED_VRAM} MiB free"

# ── CUDA MPS setup ────────────────────────────────────────────────────────────
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
export CUDA_VISIBLE_DEVICES=$BEST_GPU
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ── Run explanation generation ────────────────────────────────────────────────
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

echo "[$(date '+%H:%M:%S')] Running Step 09: Explanation Generation..."
python -u src/gen/explain_mcq.py
PY_EXIT=$?

rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

if [ $PY_EXIT -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] ✅ Step 09 done successfully"
else
    echo "[$(date '+%H:%M:%S')] ❌ Step 09 failed (exit $PY_EXIT)"
    exit $PY_EXIT
fi
