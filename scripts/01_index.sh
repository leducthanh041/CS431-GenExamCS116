#!/bin/bash
#SBATCH --job-name=01_index
#SBATCH --output=log/01_index_%j.out
#SBATCH --error=log/01_index_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00

# Step 01: Indexing (CPU — BGE-m3 embedding + PyMuPDF)
# Part A: Chunk Whisper JSON transcripts with YouTube timestamps
# Part B: Merge slide + transcript chunks → ChromaDB (với citation metadata mới)

REQUIRED_VRAM=0

set -e

module clear -f
module load slurm/slurm/24.11
source /datastore/uittogether/tools/miniconda3/etc/profile.d/conda.sh

PROJECT_ROOT="/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen"
mkdir -p "$PROJECT_ROOT/log"

echo "🚀 Job $SLURM_JOB_ID — Step 01: Indexing"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

# ── Part A: Chunk Whisper JSON transcripts (Step 01b) ───────────────────────
echo "▶️  Part A: Chunking Whisper JSON transcripts with YouTube timestamps..."
python -u src/gen/chunk_transcript_with_timestamps.py

# ── Part B: Slide + Transcript → ChromaDB ───────────────────────────────────
echo "▶️  Part B: Slide + Transcript indexing → ChromaDB..."
python -u src/gen/indexing.py

echo "✅ Step 01 done"
