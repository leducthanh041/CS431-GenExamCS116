#!/bin/bash
# gpu_check.sh — Chiếm GPU được SLURM assign và verify đủ VRAM
#
# Logic:
#   1. Xác định GPU index (từ SLURM hoặc manual)
#   2. Kill các compute process đang chiếm GPU đó
#   3. Nếu chưa đủ VRAM → đợi + retry (loop) cho đến khi đủ
#   4. Trả về GPU index khi đủ VRAM
#
# Exit codes:
#   0 + GPU index → OK, tiếp tục
#   11            → Lỗi hệ thống / timeout

set -eo pipefail

REQUIRED_VRAM_MB="${1:-25000}"
MAX_WAIT_MIN="${3:-30}"       # tối đa đợi bao lâu (phút)
SLEEP_SEC=60                   # mỗi lần đợi bao lâu

# ── Lấy GPU từ SLURM environment ────────────────────────────────────────────
get_slurm_gpu() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "$CUDA_VISIBLE_DEVICES" | tr -d '[:space:]' | grep -oE '[0-9]+' | head -1
        return 0
    fi
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
        echo "$SLURM_JOB_GPUS" | grep -oE '[0-9]+' | head -1
        return 0
    fi
    if [ -n "${SLURM_JOB_ID:-}" ] && [ "${SLURM_JOB_ID}" != "$$" ]; then
        scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
            | grep -oP 'Gres=(gpu|mlmatrix):\K[^,\s]+' \
            | head -1 \
            | grep -oE '[0-9]+' || true
    fi
}

# ── Kill compute processes on target GPU ──────────────────────────────────────
kill_stale_processes() {
    local gpu_idx="$1"
    local cleanup_attempt="${2:-1}"

    echo "[gpu_check] ─── Cleanup pass $cleanup_attempt on GPU $gpu_idx ───" >&2

    # Check current state first
    local before_free before_used
    before_free=$(nvidia-smi --id="$gpu_idx" --query-gpu=memory.free \
        --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || echo "N/A")
    before_used=$(nvidia-smi --id="$gpu_idx" --query-gpu=memory.used \
        --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || echo "N/A")
    echo "[gpu_check]   Before: ${before_free} MiB free / ${before_used} MiB used" >&2

    # Show ALL processes on this GPU (including MPS daemons)
    local killed=0
    local tmpfile
    tmpfile=$(mktemp)
    nvidia-smi --id="$gpu_idx" --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader 2>/dev/null > "$tmpfile" || true

    if [ -s "$tmpfile" ]; then
        echo "[gpu_check]   Processes on GPU $gpu_idx:" >&2
        while IFS=, read -r pid pname mem; do
            pid=$(echo "$pid" | tr -d '[:space:]')
            pname=$(echo "$pname" | tr -d '[:space:]')
            mem=$(echo "$mem" | tr -d '[:space:]')
            mem=$(echo "$mem" | sed 's/MiB$//' | tr -d '[:space:]')
            if [ -n "$pid" ] && [ "$pid" != "0" ]; then
                echo "[gpu_check]     → kill -9 $pid ($pname, ${mem}MiB)" >&2
                kill -9 "$pid" 2>/dev/null || true
                killed=$((killed + 1))
            fi
        done < "$tmpfile"
    fi
    rm -f "$tmpfile"

    # Also kill CUDA MPS daemons for this user on this GPU
    echo "[gpu_check]   Checking CUDA MPS daemons on GPU $gpu_idx..." >&2
    local mps_pids
    mps_pids=$(ps aux | grep -E "[n]vidia-cuda-mps|[n]vidia-mps" \
        | grep -v "^root" | awk '{print $2}' | tr '\n' ' ')
    if [ -n "$mps_pids" ]; then
        echo "[gpu_check]     MPS PIDs: $mps_pids" >&2
        for mp in $mps_pids; do
            kill -9 "$mp" 2>/dev/null || true
        done
    fi

    # Wait for driver to reclaim GPU memory — this is the critical part.
    # kill -9 frees GPU memory asynchronously via CUDA driver; give it 30s.
    if [ $killed -gt 0 ] || [ -n "$mps_pids" ]; then
        echo "[gpu_check]   Waiting 30s for CUDA driver to reclaim memory..." >&2
        sleep 30
    else
        echo "[gpu_check]   No processes to kill — skipping wait." >&2
    fi

    # Check state after cleanup
    local after_free after_used
    after_free=$(nvidia-smi --id="$gpu_idx" --query-gpu=memory.free \
        --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || echo "N/A")
    after_used=$(nvidia-smi --id="$gpu_idx" --query-gpu=memory.used \
        --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || echo "N/A")
    local reclaimed
    reclaimed=$(echo "$before_free $after_free" | awk '{print $2 - $1}')
    echo "[gpu_check]   After:  ${after_free} MiB free / ${after_used} MiB used (reclaimed ~${reclaimed} MiB)" >&2

    # Show all GPU statuses for full visibility
    echo "[gpu_check]   All GPUs:" >&2
    nvidia-smi --query-gpu=index,name,memory.used,memory.total \
        --format=csv 2>/dev/null >&2 || true
}

# ── Check VRAM free trên GPU đã chọn ─────────────────────────────────────────
check_free() {
    local gpu_idx="$1"
    nvidia-smi --id="$gpu_idx" --query-gpu=index,memory.free \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -v i="$gpu_idx" '$1 == i {print $2}' \
        | tr -d '[:space:]'
}

# ── Scan tất cả GPU, trả về GPU có nhiều free VRAM nhất ─────────────────────
find_best_gpu() {
    # Trả về GPU index có nhiều VRAM free nhất
    nvidia-smi --query-gpu=index,memory.free \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk '{gsub(/ /,"",$1); gsub(/ /,"",$2); if($2>max){max=$2; gpu=$1}} END{print gpu}'
}

# ── Scan tất cả GPU, hiển thị bảng trạng thái ────────────────────────────────
scan_all_gpus() {
    echo "[gpu_check] All GPU status:" >&2
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total \
        --format=csv 2>/dev/null | while IFS=, read -r idx name used free total; do
        echo "[gpu_check]   GPU $idx: $(echo "$used" | tr -d ' ') used / $(echo "$free" | tr -d ' ') free / $(echo "$total" | tr -d ' ') total  [$name]" >&2
    done
}

main() {
    local gpu_idx=""
    local free_mb=""
    local MY_USER="$(whoami)"

    # ── Step 1: Scan all GPUs, rank by free VRAM ─────────────────────────────
    scan_all_gpus

    if ! command -v nvidia-smi &>/dev/null; then
        echo "[gpu_check] nvidia-smi not available — using SLURM-assigned GPU" >&2
        gpu_idx=$(get_slurm_gpu) || gpu_idx="0"
        echo "[gpu_check] Falling back to GPU $gpu_idx" >&2
        echo "$gpu_idx"
        exit 0
    fi

    # ── Step 2: Build ranked list of GPUs (freest first) ───────────────────
    # Returns: "gpu0:12345\ngpu1:20000\n..."
    local gpu_rank_file
    gpu_rank_file=$(mktemp)
    nvidia-smi --query-gpu=index,memory.free \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk '{gsub(/ /,"",$1); gsub(/ /,"",$2); print "gpu"$1":"$2}' \
        | sort -t: -k2 -rn \
        > "$gpu_rank_file"

    local gpu_candidates=()
    while IFS=: read -r g f; do
        gpu_candidates+=("${g#gpu}:${f}")
    done < "$gpu_rank_file"
    rm -f "$gpu_rank_file"

    if [ ${#gpu_candidates[@]} -eq 0 ]; then
        echo "[gpu_check] No GPUs found" >&2
        exit 11
    fi

    echo "[gpu_check] GPU candidates (freest first):" >&2
    for c in "${gpu_candidates[@]}"; do
        echo "[gpu_check]   $c" >&2
    done

    # ── Step 3: Try GPUs in order — pick the first one we can successfully
    #            clean, or the first one that already has enough free memory ─
    local max_attempts=$((MAX_WAIT_MIN * 60 / SLEEP_SEC))
    local attempt=0
    local final_gpu_idx=""
    local final_free_mb=""

    for candidate in "${gpu_candidates[@]}"; do
        gpu_idx="${candidate%%:*}"
        free_mb="${candidate##*:}"

        echo "[gpu_check] ─── Evaluating GPU $gpu_idx (${free_mb} MiB free, need ${REQUIRED_VRAM_MB} MiB) ───" >&2

        # If already has enough free memory — perfect, use it directly
        if [ "$free_mb" -ge "$REQUIRED_VRAM_MB" ] 2>/dev/null; then
            echo "[gpu_check] GPU $gpu_idx already has ${free_mb} MiB ≥ ${REQUIRED_VRAM_MB} MiB → using it ✅" >&2
            final_gpu_idx="$gpu_idx"
            final_free_mb="$free_mb"
            break
        fi

        # Not enough free — try to kill stale processes on THIS GPU only
        echo "[gpu_check] GPU $gpu_idx needs cleanup (${free_mb} < ${REQUIRED_VRAM_MB} MiB)" >&2

        local attempt_gpu=0
        local cleaned=0
        while [ $attempt_gpu -lt $max_attempts ]; do
            attempt_gpu=$((attempt_gpu + 1))

            # Kill stale processes (only OUR processes — filter by user)
            echo "[gpu_check] ─── Cleanup attempt $attempt_gpu on GPU $gpu_idx ───" >&2

            local tmpfile
            tmpfile=$(mktemp)
            nvidia-smi --id="$gpu_idx" --query-compute-apps=pid,process_name,used_memory,owner \
                --format=csv,noheader 2>/dev/null > "$tmpfile" || true

            local killed=0
            if [ -s "$tmpfile" ]; then
                echo "[gpu_check]   Processes on GPU $gpu_idx:" >&2
                while IFS=, read -r pid pname mem owner; do
                    pid=$(echo "$pid" | tr -d '[:space:]')
                    pname=$(echo "$pname" | tr -d '[:space:]')
                    mem=$(echo "$mem" | sed 's/MiB$//' | tr -d '[:space:]')
                    owner=$(echo "$owner" | tr -d '[:space:]')
                    if [ -n "$pid" ] && [ "$pid" != "0" ]; then
                        if [ "$owner" = "$MY_USER" ] || [ "$owner" = "$(id -un)" ]; then
                            echo "[gpu_check]     → kill -9 $pid ($pname, ${mem}MiB, owner=$owner)" >&2
                            kill -9 "$pid" 2>/dev/null || true
                            killed=$((killed + 1))
                        else
                            echo "[gpu_check]     ⏭  skip $pid ($pname, ${mem}MiB, owner=$owner) — not our process" >&2
                        fi
                    fi
                done < "$tmpfile"
            fi
            rm -f "$tmpfile"

            # Kill our own MPS daemons
            local mps_pids
            mps_pids=$(ps aux | grep -E "[n]vidia-cuda-mps|[n]vidia-mps" \
                | grep "$MY_USER" | awk '{print $2}' | tr '\n' ' ')
            if [ -n "$mps_pids" ]; then
                echo "[gpu_check]   Killing our MPS daemons: $mps_pids" >&2
                for mp in $mps_pids; do kill -9 "$mp" 2>/dev/null || true; done
            fi

            if [ $killed -gt 0 ] || [ -n "$mps_pids" ]; then
                echo "[gpu_check]   Waiting 30s for CUDA driver to reclaim memory..." >&2
                sleep 30
            fi

            # Check result
            free_mb=$(check_free "$gpu_idx") || free_mb="0"
            echo "[gpu_check]   After cleanup: GPU $gpu_idx has ${free_mb} MiB free (need ${REQUIRED_VRAM_MB} MiB)" >&2
            scan_all_gpus

            if [ "$free_mb" -ge "$REQUIRED_VRAM_MB" ] 2>/dev/null; then
                echo "[gpu_check] GPU $gpu_idx: ${free_mb} MiB ≥ ${REQUIRED_VRAM_MB} MiB → OK ✅" >&2
                final_gpu_idx="$gpu_idx"
                final_free_mb="$free_mb"
                cleaned=1
                break
            fi

            # Not enough yet — wait and retry
            if [ $attempt_gpu -lt $max_attempts ]; then
                echo "[gpu_check] GPU $gpu_idx still only ${free_mb} MiB — sleeping ${SLEEP_SEC}s before retry..." >&2
                sleep "$SLEEP_SEC"
            fi
        done

        if [ "$cleaned" = "1" ]; then
            break
        fi

        # GPU couldn't be cleaned — try the next GPU in the ranking
        echo "[gpu_check] GPU $gpu_idx could not be cleaned — moving to next candidate..." >&2
    done

    # ── Step 4: Result ─────────────────────────────────────────────────────
    if [ -n "$final_gpu_idx" ]; then
        echo "[gpu_check] Selected GPU $final_gpu_idx with ${final_free_mb} MiB free" >&2
        echo "$final_gpu_idx"
        exit 0
    fi

    # Exhausted all GPUs
    echo "[gpu_check] TIMEOUT: no GPU could reach ${REQUIRED_VRAM_MB} MiB free after ${MAX_WAIT_MIN} min" >&2
    echo "[gpu_check] Hint: increase MAX_WAIT_MIN or wait for other jobs to finish" >&2
    exit 11
}

main "$@"
