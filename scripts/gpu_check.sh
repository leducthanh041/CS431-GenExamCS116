#!/bin/bash
# gpu_check.sh — Chiếm GPU được SLURM assign và verify đủ VRAM
#
# Logic đơn giản: SLURM đã assign GPU → dùng GPU đó
# Nếu nvidia-smi check thấy không đủ VRAM → exit 10 (đợi)
#
# Exit codes:
#   0 + in GPU index  → OK, tiếp tục
#   10               → GPU đã bị chiếm, không đủ VRAM, đợi
#   11               → Lỗi hệ thống

set -euo pipefail

REQUIRED_VRAM_MB="${1:-25000}"   # default 25 GiB, fit gpu_memory_utilization=0.35
JOB_ID="${2:-$$}"

# ── Lấy GPU từ SLURM environment (ưu tiên cao nhất) ───────────
get_slurm_gpu() {
    # CUDA_VISIBLE_DEVICES do SLURM set khi allocate --gres
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "$CUDA_VISIBLE_DEVICES" | tr -d '[:space:]' | grep -oE '[0-9]+' | head -1
        return 0
    fi

    # SLURM_JOB_GPUS là GPU index được assign
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
        echo "$SLURM_JOB_GPUS" | grep -oE '[0-9]+' | head -1
        return 0
    fi

    # Thử scontrol
    if [ -n "${SLURM_JOB_ID:-}" ] && [ "${SLURM_JOB_ID}" != "$$" ]; then
        scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
            | grep -oP 'Gres=(gpu|mlmatrix):\K[^,\s]+' \
            | head -1 \
            | grep -oE '[0-9]+' || true
    fi
}

# ── Check VRAM free trên GPU đã chọn ──────────────────────────
check_free() {
    local gpu_idx="$1"
    nvidia-smi --query-gpu=index,memory.free \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' -v i="$gpu_idx" '$1 == i {print $2}' \
        | tr -d '[:space:]'
}

main() {
    local gpu_idx=""
    local free_mb=""

    # Step 1: Lấy GPU từ SLURM
    gpu_idx=$(get_slurm_gpu) || true

    if [ -z "$gpu_idx" ]; then
        echo "[gpu_check] Cannot determine GPU from SLURM env, using GPU 0 as fallback" >&2
        echo "0"
        exit 0
    fi

    echo "[gpu_check] SLURM assigned GPU $gpu_idx" >&2

    # Step 2: Verify đủ VRAM (nếu nvidia-smi available)
    if ! command -v nvidia-smi &>/dev/null; then
        echo "[gpu_check] nvidia-smi not available, trusting SLURM allocation" >&2
        echo "$gpu_idx"
        exit 0
    fi

    free_mb=$(check_free "$gpu_idx") || free_mb=""
    echo "[gpu_check] GPU $gpu_idx free: ${free_mb:-N/A} MiB (need ${REQUIRED_VRAM_MB} MiB)" >&2

    if [ -z "$free_mb" ] || [ "$free_mb" -lt 1000 ]; then
        echo "[gpu_check] GPU $gpu_idx: cannot read VRAM, trusting SLURM" >&2
        echo "$gpu_idx"
        exit 0
    fi

    if [ "$free_mb" -ge "$REQUIRED_VRAM_MB" ]; then
        echo "[gpu_check] GPU $gpu_idx: ${free_mb} MiB ≥ ${REQUIRED_VRAM_MB} MiB → OK" >&2
        echo "$gpu_idx"
        exit 0
    else
        echo "[gpu_check] GPU $gpu_idx: only ${free_mb} MiB free, need ${REQUIRED_VRAM_MB} MiB → waiting" >&2
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv 2>/dev/null >&2 || true
        exit 10
    fi
}

main "$@"
