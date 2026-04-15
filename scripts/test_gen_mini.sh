#!/bin/bash
#SBATCH --job-name=test_gen_mini
#SBATCH --output=log/test_gen_mini_%j.out
#SBATCH --error=log/test_gen_mini_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=mps:1
#SBATCH --time=08:00:00

# Test mini: gen 10 câu hỏi MCQ với YouTube citation
# Model: Qwen2.5-14B-Instruct (vLLM)

REQUIRED_VRAM=28000

set -e

module clear -f
module load slurm/slurm/24.11
source /datastore/uittogether/tools/miniconda3/etc/profile.d/conda.sh

unset CUDA_VISIBLE_DEVICES
PROJECT_ROOT="/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen"
CHECK_OUT=$("$PROJECT_ROOT/scripts/gpu_check.sh" $REQUIRED_VRAM $SLURM_JOB_ID) || true
EXIT_CODE=$?

if [ $EXIT_CODE -eq 10 ]; then
    echo "$CHECK_OUT"; exit 1
elif [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"; exit 1
fi

BEST_GPU=$CHECK_OUT
echo "🚀 Job $SLURM_JOB_ID — Test Mini: Gen MCQ | GPU $BEST_GPU"

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
export CUDA_VISIBLE_DEVICES=$BEST_GPU

export PYTORCH_ALLOC_CONF=expandable_segments:True
echo "[cleanup] GPU $BEST_GPU: killing stale processes..."
nvidia-smi --id=$BEST_GPU --query-compute-apps=pid,used_memory --format=csv,noheader \
    | while IFS=, read -r gpu_idx pid mem; do
        pid=$(echo "$pid" | tr -d '[:space:]')
        [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
    done
sleep 3
echo "[cleanup] Verifying GPU $BEST_GPU memory..."
nvidia-smi --id=$BEST_GPU --query-gpu=memory.free,memory.used --format=csv,noheader,nounits

PROJECT_ROOT="/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

echo "▶️  Running test_gen_mini: gen ~10 câu hỏi với YouTube citation..."
python -u src/gen/test_gen_mini.py

rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
echo "✅ Test mini done"