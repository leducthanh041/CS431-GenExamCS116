#!/bin/bash
# ============================================================
# SLURM Training Script — QLoRA Fine-tune cho Socratic Tutor
# Model: Qwen2.5-14B-Instruct
# Method: QLoRA (4-bit NF4, LoRA rank=16, alpha=32)
# GPU: A100 80GB
#
# Usage:
#   cd socratic_tutor
#   sbatch scripts/00_train_socratic_lora.sh
#
# Theo dõi tiến độ:
#   squeue -u $USER
#   tail -f slurm-<JOBID>.out      (trong thư mục chạy sbatch)
#   grep -E "loss|epoch|step" slurm-<JOBID>.out
# ============================================================

# ── ABSOLUTE PATH — hardcoded để SLURM không bị confuse khi chạy từ spool ──
PROJECT_ROOT="/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen_copy_2/socratic_tutor"
LOG_DIR="${PROJECT_ROOT}/outputs/logs"

# ── SLURM directives ──────────────────────────────────────
# IMPORTANT: SBATCH %j = JOB_ID (log ra thư mục hiện tại)
#SBATCH --job-name=socratic_lora
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --output=%j.out          # log ghi vào thư mục chạy sbatch (vd: socratic_tutor/slurm-15143.out)
#SBATCH --error=%j.err
# ── Không gửi mail — theo dõi qua log file ────────────────

# ============================================================
# KHÔNG VIẾT GÌ SAU DÒNG NÀY TRỪ SBATCH TRÊN
# ============================================================

set -euo pipefail

# ── Verify PROJECT_ROOT ────────────────────────────────────
if [[ ! -d "${PROJECT_ROOT}" ]]; then
    echo "[FATAL] PROJECT_ROOT không tồn tại: ${PROJECT_ROOT}"
    exit 1
fi

# ── Paths ────────────────────────────────────────────────
MODEL_PATH="/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen_copy_2/models/Qwen2.5-14B-Instruct"
DATA_PATH="${PROJECT_ROOT}/data/filtered/train_filtered.jsonl"
CONFIG_PATH="${PROJECT_ROOT}/configs/qwen14b_socratic_qlora.yaml"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/checkpoints/run_001"

# ── Step 0: Banner + verify paths ────────────────────────
echo "============================================================"
echo "  Socratic Tutor — QLoRA Fine-tune"
echo "============================================================"
echo "  Project   : ${PROJECT_ROOT}"
echo "  Job ID    : ${SLURM_JOB_ID:-'local'}"
echo "  Model     : ${MODEL_PATH}"
echo "  Data      : ${DATA_PATH}"
echo "  Output    : ${OUTPUT_DIR}"
echo "============================================================"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "[ERROR] Model not found at ${MODEL_PATH}"
    exit 1
fi

if [[ ! -f "${DATA_PATH}" ]]; then
    echo "[ERROR] Training data not found at ${DATA_PATH}"
    exit 1
fi

# ── Step 1: Tạo output directories ───────────────────────
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/logs"

# ── Step 2: Load conda env ───────────────────────────────
# Thay "llm" bằng tên conda env đã cài axolotl + peft + trl + bitsandbytes
CONDA_ENV="llm"
CONDA_PATH="/datastore/uittogether/tools/miniconda3"

if [[ -d "${CONDA_PATH}/envs/${CONDA_ENV}" ]]; then
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    echo "[INFO] Activated conda env: ${CONDA_ENV}"
else
    echo "[WARN] Conda env '${CONDA_ENV}' not found — using current python"
fi

echo "[INFO] Python : $(which python 2>/dev/null || echo 'not found')"
echo "[INFO] PyTorch: $(python -c 'import torch; print(torch.__version__, "cuda=", torch.cuda.is_available())' 2>/dev/null || echo 'not available')"
echo "[INFO] GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not available')"

# ── Step 3: Chạy training ────────────────────────────────
AXOLOTL_LOG="${OUTPUT_DIR}/logs/axolotl_train.log"

echo ""
echo "[STEP 1/3] Training QLoRA adapter..."
echo "============================================================"
echo "  Config  : ${CONFIG_PATH}"
echo "  Data    : ${DATA_PATH}"
echo "  Log     : ${AXOLOTL_LOG}"
echo "============================================================"

python -m axolotl.entrypoint.train \
    "${CONFIG_PATH}" \
    --prepare-only=false \
    2>&1 | tee "${AXOLOTL_LOG}"

TRAIN_EXIT=${PIPESTATUS[0]}

if [[ ${TRAIN_EXIT} -ne 0 ]]; then
    echo ""
    echo "[ERROR] Training failed with exit code ${TRAIN_EXIT}"
    exit ${TRAIN_EXIT}
fi

# ── Step 4: Training summary ─────────────────────────────
echo ""
echo "[STEP 2/3] Training complete!"
echo "  Checkpoint : ${OUTPUT_DIR}/"

echo ""
echo "[STEP 3/3] Training metrics (last 30 lines)..."
if [[ -f "${AXOLOTL_LOG}" ]]; then
    grep -E "loss|epoch|step|learning_rate" "${AXOLOTL_LOG}" | tail -30 \
        || echo "  (Không tìm thấy dòng loss trong log)"
else
    echo "  [WARN] Log file not found: ${AXOLOTL_LOG}"
fi

echo ""
echo "============================================================"
echo "  ✅ Training COMPLETE"
echo "  Checkpoint: ${OUTPUT_DIR}/"
echo "  Next step: bash ${PROJECT_ROOT}/scripts/01_merge_and_eval.sh"
echo "============================================================"
