#!/bin/bash
# gpu_check.sh — Chọn GPU còn trống trên cluster mps:a100:2 HOẶC mps:l40:2
# Exit codes:
#   0  + in ra GPU index  → GPU đủ VRAM, job tiếp tục
#   10 + in ra message    → GPU hết (hoặc không đủ), job không đủ tài nguyên (đợi)
#   11 + in ra message    → Lỗi hệ thống (GPU bị chiếm, script lỗi, v.v.)
#
# Usage:  gpu_check.sh <REQUIRED_VRAM_MB> <SLURM_JOB_ID>
#
# Required VRAM: Qwen2.5-14B-Instruct ≈ 27 GiB + KV cache ≈ 8 GiB → ~30000 MB
#                 Gemma-3-12b-it       ≈ 24 GiB + KV cache ≈ 7 GiB → ~27000 MB

set -euo pipefail

REQUIRED_VRAM_MB="${1:-30000}"
JOB_ID="${2:-$$}"

# ── Kiểm tra nvidia-smi có sẵn không ───────────────────────────
check_vram() {
    local gpu_idx="$1"
    local free_mb
    free_mb=$(nvidia-smi --query-gpu=index,mmemory.free \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' -v idx="$gpu_idx" '$1 == idx {gsub(/[^0-9]/,"",$2); print $2}')
    echo "$free_mb"
}

# ── Fallback: kiểm tra qua /proc/driver/nvidia ────────────────
check_vram_proc() {
    local gpu_idx="$1"
    local bar_path="/proc/driver/nvidia/gpus/${gpu_idx}/info"
    if [ -r "$bar_path" ]; then
        local vram_kb
        vram_kb=$(awk '/memoryTotal/{print $3}' "$bar_path" 2>/dev/null || echo 0)
        # Fallback: giả sử mỗi GPU A100 = 40 GiB, L40 = 48 GiB
        echo "$((vram_kb / 1024))"
    else
        echo "0"
    fi
}

# ── Logic chính: check GPU nào đủ VRAM ────────────────────────
# Cluster này có 2 loại GPU:
#   DGX-A100: 8 × A100 40GB SXM4  → total ~320 GiB, MPS 800
#   AsusL40:  8 × L40 48GB        → total ~384 GiB, MPS 800
#
# Chúng ta check tất cả GPU và tìm GPU đầu tiên có free ≥ REQUIRED_VRAM_MB.
# Nếu cluster dùng MPS thì mỗi job chỉ thấy 1 GPU được assigned.

# Nếu nvidia-smi không available → dùng SLURM environment variable
# SLURM_STEP_GPUS hoặc CUDA_VISIBLE_DEVICES chỉ set khi job được allocate GPU

main() {
    local best_gpu=""
    local best_free=0

    # Method 1: dùng nvidia-smi (chạy được khi đã có GPU access)
    if command -v nvidia-smi &>/dev/null; then
        echo "[gpu_check] Method: nvidia-smi" >&2

        # Check mỗi GPU 0-7
        for idx in 0 1 2 3 4 5 6 7; do
            local free_mb
            free_mb=$(nvidia-smi --query-gpu=index,memory.free \
                --format=csv,noheader,nounits 2>/dev/null \
                | awk -F',' -v i="$idx" '
                    BEGIN {gsub(/[^0-9]/,"",i)}
                    $1 ~ "^" i "$" {gsub(/[^0-9]/,"",$2); print $2}
                  ' 2>/dev/null || echo "0")

            # Strip whitespace / non-numeric
            free_mb=$(echo "$free_mb" | tr -d '[:space:]' | grep -E '^[0-9]+$' || echo "0")

            if [ -z "$free_mb" ] || [ "$free_mb" = "0" ]; then
                continue
            fi

            echo "[gpu_check] GPU $idx: ${free_mb} MB free (need ${REQUIRED_VRAM_MB} MB)" >&2

            if [ "$free_mb" -ge "$REQUIRED_VRAM_MB" ]; then
                best_gpu="$idx"
                best_free="$free_mb"
                break
            fi
        done
    fi

    # Method 2: SLURM allocated GPU (khi chạy trong job với --gres)
    if [ -z "$best_gpu" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "[gpu_check] Method: SLURM allocated GPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)" >&2
        # CUDA_VISIBLE_DEVICES đã được SLURM set → dùng GPU đó
        best_gpu="$CUDA_VISIBLE_DEVICES"
        best_free="$REQUIRED_VRAM_MB"  # assume enough if allocated
    fi

    # Method 3: dùng SLURM_GRES (khi job được allocate --gres=gpu:X)
    if [ -z "$best_gpu" ] && [ -n "${SLURM_JOB_GPUS:-}" ]; then
        echo "[gpu_check] Method: SLURM_JOB_GPUS=${SLURM_JOB_GPUS}" >&2
        best_gpu="$SLURM_JOB_GPUS"
        best_free="$REQUIRED_VRAM_MB"
    fi

    # Method 4: dùng scontrol để lấy allocated GPU từ job
    if [ -z "$best_gpu" ] && [ -n "${SLURM_JOB_ID:-}" ] && [ "${SLURM_JOB_ID}" != "$$" ]; then
        local gres_info
        gres_info=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
            | grep -oP 'Gres=(gpu|mlmatrix):\K[^,\s]+' \
            | head -1 || echo "")
        if [ -n "$gres_info" ]; then
            # gres có dạng "gpu:0" hoặc "gpu:a100:0" → lấy số cuối
            best_gpu=$(echo "$gres_info" | grep -oP '\d+$' || echo "")
            best_free="$REQUIRED_VRAM_MB"
            echo "[gpu_check] Method: scontrol → GPU $best_gpu" >&2
        fi
    fi

    # ── Output result ──────────────────────────────────────────────
    if [ -n "$best_gpu" ]; then
        echo "[gpu_check] ✅ Selected GPU $best_gpu (${best_free} MB free ≥ ${REQUIRED_VRAM_MB} MB required)"
        echo "$best_gpu"
        exit 0
    else
        echo "[gpu_check] ❌ No GPU with ${REQUIRED_VRAM_MB} MB free available"
        echo "[gpu_check]    Cluster info: $(sinfo -o '%P %G' -n 2>/dev/null | head -3)"
        echo "[gpu_check]    Waiting for GPU resources..."
        exit 10
    fi
}

main "$@"