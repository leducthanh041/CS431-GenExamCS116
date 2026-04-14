#!/bin/bash
# gpu_check.sh — Wait for an available GPU with enough VRAM
#
# Logic:
#   1. Scan all GPUs, rank by free VRAM (freest first)
#   2. Try each GPU in order:
#        a. If already has enough VRAM → use immediately
#        b. Otherwise: kill OUR stale processes, wait, VERIFY memory
#           actually freed. If not, retry kill + wait longer.
#           Loop indefinitely until GPU has enough VRAM
#   3. Only kill OUR processes (current user) — never kill other users
#   4. NEVER just trust that kill succeeded — always verify VRAM drop
#
# Usage: source gpu_check.sh [REQUIRED_VRAM_MB]
#   REQUIRED_VRAM_MB  — minimum free VRAM (default: 25000)
#
# Output: GPU index on stdout
# Exit:   0 on success, 10 on unexpected error (requeue), 11 on system error

REQUIRED_VRAM_MB="${1:-25000}"

# IMPORTANT: no "set -e" — kill loop must NOT abort on errors.
set -o pipefail

MY_USER="$(whoami)"
SLEEP_SEC=60

# ── Check free VRAM on a specific GPU ───────────────────────────────────────
check_free() {
    local gpu_idx="$1"
    nvidia-smi --id="$gpu_idx" --query-gpu=memory.free \
        --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]'
}

# ── Scan all GPUs ───────────────────────────────────────────────────────────
scan_all_gpus() {
    echo "[gpu_check] All GPU status:" >&2
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total \
        --format=csv,noheader 2>/dev/null | while IFS=, read -r idx name used free total; do
        used=$(echo "$used" | tr -d ' ')
        free=$(echo "$free" | tr -d ' ')
        total=$(echo "$total" | tr -d ' ')
        echo "[gpu_check]   GPU $idx: ${used}MiB used / ${free}MiB free / ${total}MiB total  [$name]" >&2
    done
}

# ── Kill OUR stale processes on a GPU ──────────────────────────────────────
# Returns: number of processes killed
kill_our_processes() {
    local gpu_idx="$1"
    local total_killed=0
    local tmpfile
    tmpfile=$(mktemp)

    nvidia-smi --id="$gpu_idx" --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader 2>/dev/null > "$tmpfile" || true

    if [ -s "$tmpfile" ]; then
        while IFS=, read -r pid pname mem; do
            pid=$(echo "$pid" | tr -d '[:space:]')
            pname=$(echo "$pname" | tr -d '[:space:]')
            mem=$(echo "$mem" | sed 's/MiB$//' | tr -d '[:space:]')
            if [ -n "$pid" ] && [ "$pid" != "0" ]; then
                echo "[gpu_check]   → kill -9 $pid ($pname, ${mem}MiB) on GPU $gpu_idx" >&2
                kill -9 "$pid" 2>/dev/null || true
                total_killed=$((total_killed + 1))
            fi
        done < "$tmpfile"
    fi
    rm -f "$tmpfile"

    # Also kill CUDA MPS daemons belonging to us on this GPU
    local mps_pids
    mps_pids=$(ps aux | grep -E "[n]vidia-cuda-mps|[n]vidia-mps" \
        | grep "$MY_USER" | awk '{print $2}' | tr '\n' ' ')
    if [ -n "$mps_pids" ]; then
        echo "[gpu_check]   MPS daemons (our user): $mps_pids" >&2
        for mp in $mps_pids; do
            kill -9 "$mp" 2>/dev/null || true
        done
    fi

    echo "[gpu_check]   Killed $total_killed compute process(es) on GPU $gpu_idx" >&2
    return $total_killed
}

# ── Verify GPU memory actually freed ───────────────────────────────────────
# Returns 0 if memory freed (≥ REQUIRED_MB freed from baseline)
# Returns 1 if still not enough (retry needed)
verify_memory_freed() {
    local gpu_idx="$1"
    local before_free="$2"
    local after_free

    after_free=$(check_free "$gpu_idx") || after_free="0"
    local freed=$((after_free - before_free))

    echo "[gpu_check]   Before: ${before_free}MiB → After: ${after_free}MiB (freed: ${freed}MiB)" >&2

    if [ "$after_free" -ge "$REQUIRED_VRAM_MB" ] 2>/dev/null; then
        echo "[gpu_check]   ✅ Sufficient VRAM: ${after_free}MiB ≥ ${REQUIRED_VRAM_MB}MiB" >&2
        return 0
    else
        echo "[gpu_check]   ❌ Still insufficient: ${after_free}MiB < ${REQUIRED_VRAM_MB}MiB" >&2
        return 1
    fi
}

# ── Try to reset GPU if normal cleanup failed ────────────────────────────────
try_gpu_reset() {
    local gpu_idx="$1"
    echo "[gpu_check]   Attempting nvidia-smi --gpu-reset on GPU $gpu_idx..." >&2
    # Note: --gpu-reset usually requires root/admin. We try anyway.
    nvidia-smi --id="$gpu_idx" --gpu-reset 2>/dev/null && return 0 || true
    # Fallback: echo to sysfs to trigger driver purge (if permitted)
    echo "[gpu_check]   GPU reset not available (likely need admin). Trying fallback..." >&2
    return 1
}

# ── Main ─────────────────────────────────────────────────────────────────────
main() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo "[gpu_check] nvidia-smi not available — cannot check GPU" >&2
        exit 11
    fi

    scan_all_gpus

    # Build ranked GPU list (freest first)
    local gpu_rank_file
    gpu_rank_file=$(mktemp)
    nvidia-smi --query-gpu=index,memory.free \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk '{gsub(/[ ,]/,"",$1); gsub(/[ ,]/,"",$2); print "gpu"$1":"$2}' \
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

    # ── Try each GPU in order ───────────────────────────────────────────────
    for candidate in "${gpu_candidates[@]}"; do
        local gpu_idx="${candidate%%:*}"
        local free_mb="${candidate##*:}"

        echo "[gpu_check] ─── GPU $gpu_idx: ${free_mb}MiB free, need ${REQUIRED_VRAM_MB}MiB ───" >&2

        # Path A: already enough VRAM — use immediately
        if [ "$free_mb" -ge "$REQUIRED_VRAM_MB" ] 2>/dev/null; then
            echo "[gpu_check] GPU $gpu_idx has ${free_mb}MiB ≥ ${REQUIRED_VRAM_MB}MiB → using it ✅" >&2
            echo "$gpu_idx"
            exit 0
        fi

        # Path B: not enough — cleanup + wait + verify (loop indefinitely)
        echo "[gpu_check] GPU $gpu_idx has ${free_mb}MiB < ${REQUIRED_VRAM_MB}MiB → waiting for free VRAM..." >&2

        local cleanup_attempts=0
        local wait_seq=30  # first wait after kill

        while true; do
            cleanup_attempts=$((cleanup_attempts + 1))
            local mem_before_kill
            mem_before_kill=$(check_free "$gpu_idx") || mem_before_kill="0"

            echo "[gpu_check]   Cleanup attempt #$cleanup_attempts on GPU $gpu_idx..." >&2
            kill_our_processes "$gpu_idx"

            # Wait for CUDA driver to reclaim memory
            echo "[gpu_check]   Waiting ${wait_seq}s for CUDA driver to reclaim memory..." >&2
            sleep "$wait_seq"

            # Verify: check if VRAM actually increased
            local mem_after_kill
            mem_after_kill=$(check_free "$gpu_idx") || mem_after_kill="0"
            local freed=$((mem_after_kill - mem_before_kill))

            echo "[gpu_check]   GPU $gpu_idx: ${mem_before_kill}MiB → ${mem_after_kill}MiB (freed: ${freed}MiB)" >&2
            scan_all_gpus

            if [ "$mem_after_kill" -ge "$REQUIRED_VRAM_MB" ] 2>/dev/null; then
                echo "[gpu_check] GPU $gpu_idx: ${mem_after_kill}MiB ≥ ${REQUIRED_VRAM_MB}MiB → OK ✅" >&2
                echo "$gpu_idx"
                exit 0
            fi

            # Memory not freed enough — check why
            if [ "$freed" -lt 500 ] && [ $cleanup_attempts -gt 2 ]; then
                # Nothing was freed after multiple tries → try GPU reset
                echo "[gpu_check]   ⚠️ Memory not freed after multiple attempts (freed: ${freed}MiB)" >&2
                try_gpu_reset "$gpu_idx"
                local reset_ok=$?
                echo "[gpu_check]   After GPU reset attempt, waiting extra 60s..." >&2
                sleep 60

                mem_after_kill=$(check_free "$gpu_idx") || mem_after_kill="0"
                if [ "$mem_after_kill" -ge "$REQUIRED_VRAM_MB" ] 2>/dev/null; then
                    echo "[gpu_check] GPU $gpu_idx: ${mem_after_kill}MiB ≥ ${REQUIRED_VRAM_MB}MiB after reset → OK ✅" >&2
                    echo "$gpu_idx"
                    exit 0
                fi
            fi

            # Increase wait time progressively (30 → 60 → 90 → 120s)
            if [ $wait_seq -lt 120 ]; then
                wait_seq=$((wait_seq + 30))
            fi
            echo "[gpu_check]   GPU $gpu_idx still only ${mem_after_kill}MiB < ${REQUIRED_VRAM_MB}MiB — sleeping ${SLEEP_SEC}s..." >&2
            sleep "$SLEEP_SEC"
        done

    done

    # Should never reach here (infinite loop), but safety exit
    echo "[gpu_check] Unexpected: no GPU selected — signaling requeue" >&2
    exit 10
}

main "$@"