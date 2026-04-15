#!/bin/bash
#SBATCH --job-name=cuda_check
#SBATCH --output=log/cuda_check_%j.out
#SBATCH --error=log/cuda_check_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=mps:1
#SBATCH --time=00:05:00

REQUIRED_VRAM=1000

set -euo pipefail

module clear -f
module load slurm/slurm/24.11
source /datastore/uittogether/tools/miniconda3/etc/profile.d/conda.sh
set +u
conda activate /datastore/uittogether/tools/miniconda3/envs/cs431mcq
set -u

PROJECT_ROOT="/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen"

unset CUDA_VISIBLE_DEVICES
CHECK_OUT=$("$PROJECT_ROOT/scripts/gpu_check.sh" $REQUIRED_VRAM $SLURM_JOB_ID 2>&1)
EXIT_CODE=$?

if [ $EXIT_CODE -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 1
elif [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1
fi

BEST_GPU=$(echo "$CHECK_OUT" | grep -E '^[0-9]+$' | tail -1)
export CUDA_VISIBLE_DEVICES=$BEST_GPU

echo "=== CUDA Diagnostics (GPU $BEST_GPU) ==="
python3 -c "
import os, torch, whisper
print(f'CUDA_VISIBLE_DEVICES = {os.environ.get(\"CUDA_VISIBLE_DEVICES\",\"NOT SET\")}')
print(f'torch.__version__       = {torch.__version__}')
print(f'torch.version.cuda     = {torch.version.cuda}')
print(f'torch.cuda.is_available() = {torch.cuda.is_available()}')
print(f'torch.cuda.device_count() = {torch.cuda.device_count()}')
if torch.cuda.device_count() > 0:
    print(f'torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}')
print(f'whisper.__version__     = {whisper.__version__}')
import subprocess
r = subprocess.run([\"nvidia-smi\", \"--query-gpu=driver_version\", \"--format=csv,noheader\"], capture_output=True, text=True)
print(f'NVIDIA driver           = {r.stdout.strip()}')
"
echo "=== Done ==="