#!/bin/bash
# ==============================================================================
# scripts/00_orchestrate.sh — Pipeline Orchestrator (Phases 1 + 2)
# ==============================================================================
# Chạy toàn bộ pipeline MCQGen (Steps 01b → 10):
#
#   Phase 1 (Python terminal): Steps 01b + 01 — indexing (chạy trực tiếp)
#   Phase 2 (SLURM scripts):   Steps 02 → 10 — retrieval, gen, eval, metrics
#
# Cách dùng:
#   bash scripts/00_orchestrate.sh           # Full pipeline (01b → 10)
#   bash scripts/00_orchestrate.sh 01b 01    # Chỉ indexing (01b + 01)
#   bash scripts/00_orchestrate.sh 02 10    # Chạy từ retrieval đến metrics
#   bash scripts/00_orchestrate.sh 03 08    # Chỉ gen + eval (bỏ qua indexing)
#
# Đặt EXP_NAME tại src/common.py (Config.EXP_NAME) trước khi chạy.
# ==============================================================================

PROJECT_ROOT="/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen"
LOG_DIR="$PROJECT_ROOT/log"
mkdir -p "$LOG_DIR"

# Step ordering: defines execution order and comparison order
# 01b = transcript chunking (before 01 indexing)
STEP_ORDER="01b 01 02 03 04 05 06 07 08 09 10"

# Parse step arguments (default: run all)
STEPS_ARG="${1:-01b-10}"
IFS='-' read -ra RANGE <<< "$STEPS_ARG"
LOWER="${RANGE[0]:-01b}"
UPPER="${RANGE[1]:-$LOWER}"

echo "=============================================="
echo "🚀 MCQGen Pipeline Orchestrator"
echo "   Range: Step $LOWER → Step $UPPER"
echo "   $(date)"
echo "=============================================="

# ── Helper: get the index of a step in STEP_ORDER ─────────────────────────────
_step_index() {
    local step="$1"
    local idx=0
    for s in $STEP_ORDER; do
        if [[ "$s" == "$step" ]]; then
            echo "$idx"
            return
        fi
        idx=$((idx + 1))
    done
    echo "-1"
}

# ── Helper: check if a step is within range (inclusive) ───────────────────────
_in_range() {
    local step_num="$1"
    local lower_idx=$(_step_index "$LOWER")
    local upper_idx=$(_step_index "$UPPER")
    local step_idx=$(_step_index "$step_num")

    # If step not found in order list, skip it
    [[ "$step_idx" -lt 0 ]] && return 1

    [[ "$step_idx" -ge "$lower_idx" && "$step_idx" -le "$upper_idx" ]]
}

# ── Helper: run a SLURM script step ────────────────────────────────────────────
_run_slurm() {
    local step_num="$1"
    local script="$2"
    local label="$3"

    if ! _in_range "$step_num"; then
        echo "  ⏭  [Step $step_num] Skip"
        return 0
    fi

    echo ""
    echo "  ▶  [Step $step_num] $label"
    echo "     Script: $script"

    if [[ -f "$script" ]]; then
        bash "$script"
        echo "  ✅  [Step $step_num] Done"
    else
        echo "  ⚠️   [Step $step_num] Script not found: $script"
    fi
}

# ── Helper: run a Python terminal step ─────────────────────────────────────────
_run_python() {
    local step_num="$1"
    local py_file="$2"
    local label="$3"

    if ! _in_range "$step_num"; then
        echo "  ⏭  [Step $step_num] Skip"
        return 0
    fi

    echo ""
    echo "  ▶  [Step $step_num] $label (Python terminal)"
    echo "     File: $py_file"

    if [[ -f "$PROJECT_ROOT/$py_file" ]]; then
        cd "$PROJECT_ROOT"
        export PYTHONPATH="$PROJECT_ROOT"
        export TOKENIZERS_PARALLELISM=false
        export OMP_NUM_THREADS=1
        python -u "$py_file"
        echo "  ✅  [Step $step_num] Done"
    else
        echo "  ⚠️   [Step $step_num] File not found: $py_file"
    fi
}

# ════════════════════════════════════════════════════════════════════════════════
# Phase 1: Python terminal steps
# ════════════════════════════════════════════════════════════════════════════════

# Step 01b: Chunk Whisper JSON transcripts with YouTube timestamps
_run_python "01b" "src/gen/chunk_transcript_with_timestamps.py" \
    "Chunk Transcripts (Whisper JSON → timestamped chunks)"

# Step 01: Indexing — Slide + Transcript → ChromaDB (BGE-m3)
_run_python "01" "src/gen/indexing.py" \
    "Indexing (Slide + Transcript → ChromaDB)"

# ════════════════════════════════════════════════════════════════════════════════
# Phase 2: SLURM script steps
# ════════════════════════════════════════════════════════════════════════════════

# Step 02: Hybrid RAG Retrieval (BM25 + ChromaDB + RRF + Rerank)
_run_slurm "02" "$PROJECT_ROOT/scripts/02_retrieval.sh" \
    "Hybrid RAG Retrieval (BM25 + ChromaDB + RRF + Rerank)"

# Step 03: P1 — Stem + Key Generation (Qwen2.5-14B)
_run_slurm "03" "$PROJECT_ROOT/scripts/03_gen_stem.sh" \
    "P1: Stem + Key Generation (Qwen2.5-14B)"

# Step 04: P2+P3 — Self-Refine Stems (Qwen2.5-14B)
_run_slurm "04" "$PROJECT_ROOT/scripts/04_gen_refine.sh" \
    "P2+P3: Self-Refine Stems"

# Step 05: P4 — Generate 6 Distractor Candidates (Qwen2.5-14B)
_run_slurm "05" "$PROJECT_ROOT/scripts/05_gen_distractors.sh" \
    "P4: Generate 6 Distractor Candidates"

# Step 06: P5-P8 — Chain-of-Thought Distractor Selection (Qwen2.5-14B)
_run_slurm "06" "$PROJECT_ROOT/scripts/06_gen_cot.sh" \
    "P5-P8: Chain-of-Thought Distractor Selection"

# Step 07: Overall Evaluation — 8-Criteria Check (Gemma-3-12b-it)
_run_slurm "07" "$PROJECT_ROOT/scripts/07_eval.sh" \
    "Eval Overall: 8-Criteria Check (Gemma-3-12b)"

# Step 08: IWF Evaluation — Item Writing Flaws (Gemma-3-12b-it)
_run_slurm "08" "$PROJECT_ROOT/scripts/08_eval_iwf.sh" \
    "Eval IWF: Item Writing Flaws (Gemma-3-12b)"

# Step 09: Explanation Generation (Qwen2.5-14B + optional web search)
_run_slurm "09" "$PROJECT_ROOT/scripts/09_explain.sh" \
    "Explanation: Gen explanations + citations (Qwen)"

# Step 10: Eval Metrics — BLEU/ROUGE/BERT + Bloom distribution
_run_slurm "10" "$PROJECT_ROOT/scripts/10_eval_metrics.sh" \
    "Eval Metrics: BLEU/ROUGE/BERT + Bloom distribution"

echo ""
echo "=============================================="
echo "✅ Orchestrator complete at $(date)"
echo "=============================================="