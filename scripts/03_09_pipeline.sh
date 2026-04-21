#!/bin/bash
#SBATCH --job-name=mcqgen_03_09
#SBATCH --output=log/03_09_%j.out
#SBATCH --error=log/03_09_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=mps:a100:1
#SBATCH --time=48:00:00

# ═══════════════════════════════════════════════════════════════════
# 03_09_pipeline.sh — Chạy lần lượt các bước 03 → 09
# Mỗi bước chạy xong mới chạy bước tiếp theo, trong cùng 1 job.
#
# EXP_NAME được lấy từ src/common.py → Config.EXP_NAME (mặc định)
# Override bằng biến môi trường nếu cần: EXP_NAME=my_exp sbatch ...
# ═══════════════════════════════════════════════════════════════════

set -e

PROJECT_ROOT="/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen"
REQUIRED_VRAM=39000

# EXP_NAME: lấy từ src/common.py → Config.EXP_NAME
if [ -z "$EXP_NAME" ]; then
    EXP_NAME=$(python -c "
import sys; sys.path.insert(0, 'src')
from common import Config
print(Config.EXP_NAME)
" 2>/dev/null) || EXP_NAME="demo_$(date +%Y%m%d_%H%M%S)"
fi

echo "════════════════════════════════════════"
echo "📝 MCQGen 03-09 Pipeline — Job $SLURM_JOB_ID"
echo "   Experiment: $EXP_NAME"
echo "   Node: $(hostname)"
echo "   Start: $(date)"
echo "════════════════════════════════════════"

# ── Environment setup ──────────────────────────────────────────────────────────
module clear -f 2>/dev/null || true
module load slurm/slurm/24.11 2>/dev/null || true
source /datastore/uittogether/tools/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true

if [ -d "/datastore/uittogether/tools/miniconda3/envs/cs431mcq" ]; then
    conda activate cs431mcq 2>/dev/null || true
fi

unset CUDA_VISIBLE_DEVICES
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
export EXP_NAME="$EXP_NAME"

# ── GPU allocation (dùng chung cho tất cả các step GPU 03-09) ─────────────────
echo "[$(date '+%H:%M:%S')] Waiting for GPU with ${REQUIRED_VRAM} MiB free VRAM..."

CHECK_OUT=$("$PROJECT_ROOT/scripts/gpu_check.sh" $REQUIRED_VRAM $SLURM_JOB_ID) || true
EXIT_CODE=$?

if [ $EXIT_CODE -eq 10 ] || [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    echo "[ERROR] GPU allocation failed — exiting"
    exit 1
fi

BEST_GPU=$CHECK_OUT
echo "[$(date '+%H:%M:%S')] 🟢 GPU allocated: $BEST_GPU (VRAM ≥ ${REQUIRED_VRAM}MB)"

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
export CUDA_VISIBLE_DEVICES=$BEST_GPU
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ── Helpers ────────────────────────────────────────────────────────────────────
log_step() {
    echo ""
    echo "════════════════════════════════"
    echo "STEP $1: $2"
    echo "════════════════════════════════"
}

run_step() {
    local step=$1
    local name=$2
    local python_file=$3

    log_step "$step" "$name"
    echo "[$(date '+%H:%M:%S')] ▶️  Running python -u $python_file"
    python -u "$python_file"
    PY_EXIT=$?
    if [ $PY_EXIT -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✅ Step $step done successfully"
        # Flush CUDA cache giữa các step
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        sleep 3
    else
        echo "[$(date '+%H:%M:%S')] ❌ Step $step failed (exit $PY_EXIT)"
        exit $PY_EXIT
    fi
}

# ════════════════════════════════════════════════════════════════════
# Step 03: P1 Generate Stem + Key
# ════════════════════════════════════════════════════════════════════
log_step "03" "P1: Generate Stem + Key (Qwen2.5-14B)"
run_step "03" "P1 Gen Stem" "$PROJECT_ROOT/src/gen/p1_gen_stem.py"

# ════════════════════════════════════════════════════════════════════
# Step 04: P2+P3 Self-Refine Stem
# ════════════════════════════════════════════════════════════════════
log_step "04" "P2+P3: Self-Refine Stem (Qwen2.5-14B)"
run_step "04" "P2+P3 Refine" "$PROJECT_ROOT/src/gen/p2_p3_refine.py"

# ════════════════════════════════════════════════════════════════════
# Step 05: P4 Distractor Candidates
# ════════════════════════════════════════════════════════════════════
log_step "05" "P4: Distractor Candidates (Qwen2.5-14B)"
run_step "05" "P4 Distractors" "$PROJECT_ROOT/src/gen/p4_candidates.py"

# ════════════════════════════════════════════════════════════════════
# Step 06: P5-P8 CoT Distractor Selection
# ════════════════════════════════════════════════════════════════════
log_step "06" "P5-P8: CoT Distractor Selection (Qwen2.5-14B)"
run_step "06" "P5-P8 CoT" "$PROJECT_ROOT/src/gen/p5_p8_cot.py"

# ════════════════════════════════════════════════════════════════════
# Step 07: Overall Evaluation (8 criteria)
# ════════════════════════════════════════════════════════════════════
log_step "07" "Eval: Overall (8 criteria) (Gemma-3-12b)"
run_step "07" "Eval Overall" "$PROJECT_ROOT/src/eval/eval_overall.py"

# ════════════════════════════════════════════════════════════════════
# Step 08: IWF Distractor Analysis
# ════════════════════════════════════════════════════════════════════
log_step "08" "Eval: IWF Distractor Analysis (Gemma-3-12b)"
run_step "08" "Eval IWF" "$PROJECT_ROOT/src/eval/eval_iwf.py"

# ════════════════════════════════════════════════════════════════════
# Step 09: Explanation Generation
# ════════════════════════════════════════════════════════════════════
log_step "09" "Explanation Generation"
run_step "09" "Explain MCQ" "$PROJECT_ROOT/src/gen/explain_mcq.py"

# ── Cleanup GPU ────────────────────────────────────────────────────────────────
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

# ── Final ─────────────────────────────────────────────────────────────────────────
FINAL_JSONL="$PROJECT_ROOT/output/$EXP_NAME/08_eval_iwf/final_accepted_questions.jsonl"
EXPLAIN_JSONL="$PROJECT_ROOT/output/$EXP_NAME/09_explain/explanations.jsonl"
if [ -f "$FINAL_JSONL" ]; then
    MCQ_COUNT=$(wc -l < "$FINAL_JSONL")
    echo ""
    echo "════════════════════════════════"
    echo "✅ Pipeline 03-09 complete — Job $SLURM_JOB_ID"
    echo "   Experiment: $EXP_NAME"
    echo "   MCQs accepted: $MCQ_COUNT"
    echo "   Output: $FINAL_JSONL"
    if [ -f "$EXPLAIN_JSONL" ]; then
        echo "   Explanations: $EXPLAIN_JSONL"
    fi
    echo "   End: $(date)"
    echo "════════════════════════════════"
else
    echo ""
    echo "⚠️  Pipeline finished but final output not found: $FINAL_JSONL"
fi