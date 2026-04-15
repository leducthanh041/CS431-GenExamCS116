#!/bin/bash
#SBATCH --job-name=06_gen_cot
#SBATCH --output=log/06_gen_cot_%j.out
#SBATCH --error=log/06_gen_cot_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=mps:1
#SBATCH --time=16:00:00

# Step 06: P5–P8 — CoT Distractor Selection + Assembly
# Model: Qwen2.5-14B-Instruct (vLLM)

REQUIRED_VRAM=36000

set -e

module clear -f
module load slurm/slurm/24.11
source /datastore/uittogether/tools/miniconda3/etc/profile.d/conda.sh

PROJECT_ROOT="/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen"

unset CUDA_VISIBLE_DEVICES
CHECK_OUT=$("$PROJECT_ROOT/scripts/gpu_check.sh" $REQUIRED_VRAM $SLURM_JOB_ID) || true
EXIT_CODE=$?

if [ $EXIT_CODE -eq 10 ] || [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1
fi

BEST_GPU=$CHECK_OUT
echo "🚀 Job $SLURM_JOB_ID — Step 06: P5-P8 CoT | GPU $BEST_GPU"

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
export CUDA_VISIBLE_DEVICES=$BEST_GPU
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

echo "▶️  Running P5-P8: CoT Distractor Selection + Assembly..."
python -u src/gen/p5_p8_cot.py

rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
echo "✅ Step 06 done"
