#!/bin/bash
#SBATCH --job-name=transcribe_new
#SBATCH --output=log/transcribe_new_%j.out
#SBATCH --error=log/transcribe_new_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=mps:l40:2
#SBATCH --time=12:00:00

# Transcribe only the 6 newly added videos using Whisper large-v3 (GPU inference)

REQUIRED_VRAM=14000

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
echo "🚀 Job $SLURM_JOB_ID — Transcription: Whisper large-v3 | GPU $BEST_GPU"

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
export CUDA_VISIBLE_DEVICES=$BEST_GPU

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

# ── Transcribe specific new videos (8.4–8.8 + 11.5) ─────────────────────────
echo "▶️  Transcribing 6 new videos..."

for video in 8.4 8.5 8.6 8.7 8.8 11.5; do
    video_path="$PROJECT_ROOT/input/video/${video}.mp4"
    if [ ! -f "$video_path" ]; then
        echo "⚠️  File not found: $video_path — skip"
        continue
    fi

    echo "  🎬 Transcribing: $video_path"
    python -u scripts/transcribe_videos.py \
        --video "$video_path" \
        --output "$PROJECT_ROOT/input/transcribe_data" \
        --model large-v3 \
        --language vi

    echo "  ✅ Done: ${video}.json"
done

rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
echo "✅ Transcription of new videos done"