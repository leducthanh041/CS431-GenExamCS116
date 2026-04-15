#!/bin/bash
#SBATCH --job-name=10_eval_metrics
#SBATCH --output=log/10_eval_metrics_%j.out
#SBATCH --error=log/10_eval_metrics_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Step 10: Evaluate Pipeline Metrics
# No GPU needed — runs CPU-only metrics

set -e

module clear -f
module load slurm/slurm/24.11
source /datastore/uittogether/tools/miniconda3/etc/profile.d/conda.sh
conda activate cs431mcq

PROJECT_ROOT="/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

# Default experiment
EXP_NAME=""
OUTPUT_DIR=""
HUMAN_CSV="${2:-}"

echo "🚀 Job $SLURM_JOB_ID — Step 10: Eval Metrics | exp=$EXP_NAME"

if [ -n "$HUMAN_CSV" ]; then
    echo "▶️  Running metrics WITH human labels: $HUMAN_CSV"
    python -u src/eval/eval_metrics.py \
        --exp "$EXP_NAME" \
        --human-csv "$HUMAN_CSV" \
        --output "$OUTPUT_DIR"
else
    echo "▶️  Running metrics (no human labels)"
    python -u src/eval/eval_metrics.py \
        --exp "$EXP_NAME" \
        --output "$OUTPUT_DIR"
fi

echo "✅ Step 10 done"
