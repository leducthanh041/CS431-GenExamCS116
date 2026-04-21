#!/bin/bash
# ============================================================
# Post-Training: Merge LoRA + Run Socratic Behavior Evaluation
# ============================================================
#
# Chạy SAU khi 00_train_socratic_lora.sh hoàn thành.
#
# Usage:
#   bash scripts/01_merge_and_eval.sh
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADAPTER_DIR="${PROJECT_ROOT}/outputs/checkpoints/run_001"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/merged/run_001"
BASE_MODEL="/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen_copy_2/models/Qwen2.5-14B-Instruct"

echo "============================================================"
echo "  Step 1/2 — Merge LoRA Adapter"
echo "============================================================"

python3 "${PROJECT_ROOT}/src/merge_lora.py" \
    --adapter "${ADAPTER_DIR}" \
    --base "${BASE_MODEL}" \
    --output "${OUTPUT_DIR}" \
    --quantize

echo ""
echo "============================================================"
echo "  Step 2/2 — Socratic Behavior Evaluation"
echo "============================================================"

python3 "${PROJECT_ROOT}/src/merge_lora.py" \
    --adapter "${ADAPTER_DIR}" \
    --base "${BASE_MODEL}" \
    --output "${OUTPUT_DIR}" \
    --skip-merge \
    --eval

echo ""
echo "============================================================"
echo "  ✅ All steps complete!"
echo "  Merged model  : ${OUTPUT_DIR}"
echo "  Eval results  : ${PROJECT_ROOT}/outputs/merged/run_001/eval/behavior_eval.json"
echo "============================================================"
