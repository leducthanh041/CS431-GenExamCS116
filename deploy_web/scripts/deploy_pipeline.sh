#!/bin/bash
#SBATCH --job-name=mcqgen_demo
#SBATCH --output=/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen/log/deploy_pipeline_%j.out
#SBATCH --error=/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen/log/deploy_pipeline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=mps:a100:2
#SBATCH --time=48:00:00
#SBATCH --open-mode=append

# ═══════════════════════════════════════════════════════════════════
# deploy_pipeline.sh — Chạy MCQGen Pipeline qua SLURM
# ═══════════════════════════════════════════════════════════════════
# Cách dùng (từ app.py hoặc command line):
#   sbatch deploy_pipeline.sh          # Demo auto name
#   EXP_NAME=my_exp sbatch deploy_pipeline.sh  # Tên custom
#
# Kết quả ra:
#   output/<EXP_NAME>/08_eval_iwf/final_accepted_questions.jsonl
# ═══════════════════════════════════════════════════════════════════

set -e

PROJECT_ROOT="/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen"
EXP_NAME="${EXP_NAME:-demo_$(date +%Y%m%d_%H%M%S)}"

REQUIRED_VRAM=36000

echo "════════════════════════════════════════"
echo "📝 MCQGen Pipeline — Job $SLURM_JOB_ID"
echo "   Experiment: $EXP_NAME"
echo "   Node: $(hostname)"
echo "   Start: $(date)"
echo "════════════════════════════════════════"

# ── Environment setup ──────────────────────────────────────────────────────────
module clear -f 2>/dev/null || true
module load slurm/slurm/24.11 2>/dev/null || true
source /datastore/uittogether/tools/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true

if [ -d "/datastore/uittogether/tools/miniconda3/envs/mcqgen" ]; then
    conda activate mcqgen 2>/dev/null || true
fi

unset CUDA_VISIBLE_DEVICES

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

# ── GPU allocation ─────────────────────────────────────────────────────────────
CHECK_OUT=$("$PROJECT_ROOT/scripts/gpu_check.sh" $REQUIRED_VRAM $SLURM_JOB_ID) || true
EXIT_CODE=$?

if [ $EXIT_CODE -eq 10 ]; then
    echo "$CHECK_OUT"
    echo "[ERROR] GPU allocation failed — system error"
    exit 1
elif [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    echo "[ERROR] GPU check failed — not enough VRAM"
    exit 1
fi

BEST_GPU=$CHECK_OUT
echo "🟢 GPU allocated: $BEST_GPU (VRAM ≥ ${REQUIRED_VRAM}MB)"

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
export CUDA_VISIBLE_DEVICES=$BEST_GPU
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ── Pipeline steps ─────────────────────────────────────────────────────────────
log_step() {
    echo ""
    echo "════════════════════════════════"
    echo "STEP $1: $2"
    echo "════════════════════════════════"
}

run_step() {
    local step=$1
    local name=$2
    local script=$3
    local python_file=$4

    log_step "$step" "$name"
    if [ -f "$script" ]; then
        bash "$script" 2>&1 | tee "$PROJECT_ROOT/log/${step}_${EXP_NAME}_${SLURM_JOB_ID}.log"
    else
        echo "⚠️  Script not found: $script — trying python directly"
        python -u "$python_file" 2>&1 | tee "$PROJECT_ROOT/log/${step}_${EXP_NAME}_${SLURM_JOB_ID}.log"
    fi
}

# ── Step 02: Hybrid Retrieval (CPU) ───────────────────────────────────────────
run_step "02" "Retrieval (BM25+Vector+RRF)" \
    "$PROJECT_ROOT/scripts/02_retrieval.sh" \
    "$PROJECT_ROOT/src/gen/retrieval.py"

# ── Step 03: P1 Generate Stem ─────────────────────────────────────────────────
run_step "03" "P1: Generate Stem + Key (GPU $BEST_GPU)" \
    "$PROJECT_ROOT/scripts/03_gen_stem.sh" \
    "$PROJECT_ROOT/src/gen/p1_gen_stem.py"

# ── Step 04: P2+P3 Self-Refine ────────────────────────────────────────────────
run_step "04" "P2+P3: Self-Refine Stem" \
    "$PROJECT_ROOT/scripts/04_gen_refine.sh" \
    "$PROJECT_ROOT/src/gen/p2_p3_refine.py"

# ── Step 05: P4 Distractor Candidates ────────────────────────────────────────
run_step "05" "P4: Distractor Candidates" \
    "$PROJECT_ROOT/scripts/05_gen_distractors.sh" \
    "$PROJECT_ROOT/src/gen/p4_candidates.py"

# ── Step 06: P5-P8 CoT Distractor Selection ───────────────────────────────────
run_step "06" "P5-P8: CoT Distractor Selection" \
    "$PROJECT_ROOT/scripts/06_gen_cot.sh" \
    "$PROJECT_ROOT/src/gen/p5_p8_cot.py"

# ── Step 07: Overall Evaluation ───────────────────────────────────────────────
run_step "07" "Eval: Overall (8 criteria)" \
    "$PROJECT_ROOT/scripts/07_eval.sh" \
    "$PROJECT_ROOT/src/eval/eval_overall.py"

# ── Step 08: IWF Evaluation ───────────────────────────────────────────────────
run_step "08" "Eval: IWF (Item Writing Flaws)" \
    "$PROJECT_ROOT/scripts/08_eval_iwf.sh" \
    "$PROJECT_ROOT/src/eval/eval_iwf.py"

# ── Cleanup ────────────────────────────────────────────────────────────────────
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

FINAL_JSONL="$PROJECT_ROOT/output/$EXP_NAME/08_eval_iwf/final_accepted_questions.jsonl"
if [ -f "$FINAL_JSONL" ]; then
    MCQ_COUNT=$(wc -l < "$FINAL_JSONL")
    echo ""
    echo "════════════════════════════════"
    echo "✅ Pipeline complete — Job $SLURM_JOB_ID"
    echo "   Experiment: $EXP_NAME"
    echo "   MCQs accepted: $MCQ_COUNT"
    echo "   Output: $FINAL_JSONL"
    echo "   End: $(date)"
    echo "════════════════════════════════"
else
    echo ""
    echo "⚠️  Pipeline finished but final output not found: $FINAL_JSONL"
    echo "   Check logs above for errors."
fi