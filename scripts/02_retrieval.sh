#!/bin/bash
#SBATCH --job-name=02_retrieval
#SBATCH --output=log/02_retrieval_%j.out
#SBATCH --error=log/02_retrieval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00

# Step 02: RAG Retrieval (CPU — ChromaDB query)
# Khong can GPU

REQUIRED_VRAM=0

set -e

module clear -f
module load slurm/slurm/24.11
source /datastore/uittogether/tools/miniconda3/etc/profile.d/conda.sh

PROJECT_ROOT="/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen"
mkdir -p "$PROJECT_ROOT/log"

echo "🚀 Job $SLURM_JOB_ID — Step 02: Retrieval"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

echo "▶️  Running RAG retrieval..."
python -u src/gen/retrieval.py

echo "✅ Step 02 done"
