#!/bin/bash
#SBATCH --job-name=08_eval_iwf
#SBATCH --output=log/08_eval_iwf_%j.out
#SBATCH --error=log/08_eval_iwf_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=mps:1
#SBATCH --time=12:00:00

# Step 08: IWF Distractor Analysis + Final Output
# Model: Gemma-3-12b-it (vLLM)

REQUIRED_VRAM=24000

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
echo "🚀 Job $SLURM_JOB_ID — Step 08: Eval IWF | GPU $BEST_GPU"

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
export CUDA_VISIBLE_DEVICES=$BEST_GPU

PROJECT_ROOT="/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

echo "▶️  Running Step 08: IWF Distractor Analysis..."
python -u src/eval/eval_iwf.py

rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
echo "✅ Step 08 done"
