#!/bin/bash
#SBATCH --job-name=05_gen_distr
#SBATCH --output=log/05_gen_distr_%j.out
#SBATCH --error=log/05_gen_distr_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=mps:1
#SBATCH --time=12:00:00

# Step 05: P4 — Generate Distractor Candidates
# Model: Qwen2.5-14B-Instruct (vLLM)

REQUIRED_VRAM=28000

set -e

module clear -f
module load slurm/slurm/24.11
source /datastore/uittogether/tools/miniconda3/etc/profile.d/conda.sh

unset CUDA_VISIBLE_DEVICES
CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?

if [ $EXIT_CODE -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0
elif [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1
fi

BEST_GPU=$CHECK_OUT
echo "🚀 Job $SLURM_JOB_ID — Step 05: P4 Distractor Candidates | GPU $BEST_GPU"

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
export CUDA_VISIBLE_DEVICES=$BEST_GPU

# ── Kill stale processes on this GPU before loading model ───────────────────
export PYTORCH_ALLOC_CONF=expandable_segments:True
echo "[cleanup] GPU $BEST_GPU: killing stale compute processes..."
nvidia-smi --id=$BEST_GPU --query-compute-apps=pid,used_memory --format=csv,noheader \
    | while IFS=, read -r pid mem; do
        pid=$(echo "$pid" | tr -d '[:space:]')
        echo "[cleanup] Killing stale PID $pid ($mem)"
        kill -9 "$pid" 2>/dev/null || true
    done
sleep 2
echo "[cleanup] GPU $BEST_GPU VRAM after cleanup:"
nvidia-smi --id=$BEST_GPU --query-gpu=memory.free --format=csv,noheader,nounits

PROJECT_ROOT="/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

echo "▶️  Running P4: Generate Distractor Candidates..."
python -u src/gen/p4_candidates.py

rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
echo "✅ Step 05 done"