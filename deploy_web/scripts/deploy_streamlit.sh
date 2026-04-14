#!/bin/bash
#SBATCH --job-name=mcqgen_streamlit
#SBATCH --output=/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen/log/deploy_streamlit_%j.out
#SBATCH --error=/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen/log/deploy_streamlit_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --gres=mps:l40:1
#SBATCH --open-mode=append

# ═══════════════════════════════════════════════════════════════════
# deploy_streamlit.sh — Khởi động MCQGen Web UI trên SLURM
# ═══════════════════════════════════════════════════════════════════
# Cách dùng:
#   sbatch deploy_streamlit.sh              # Chạy cổng mặc định 8501
#   sbatch deploy_streamlit.sh 8502         # Chỉ định port khác
#
# Sau khi job chạy, Streamlit sẽ hiển thị URL (VD):
#   External: http://server1:8501
# ═══════════════════════════════════════════════════════════════════

set -e

STREAMLIT_PORT="${1:-8501}"
PROJECT_ROOT="/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen"
DEPLOY_WEB="$PROJECT_ROOT/deploy_web"
LOG_FILE="$PROJECT_ROOT/log/deploy_streamlit_${SLURM_JOB_ID}.out"

echo "════════════════════════════════════════"
echo "🚀 MCQGen Streamlit — Job $SLURM_JOB_ID"
echo "   Port: $STREAMLIT_PORT"
echo "   Node: $(hostname)"
echo "   Time: $(date)"
echo "════════════════════════════════════════"

# Load modules
module clear -f 2>/dev/null || true
module load slurm/slurm/24.11 2>/dev/null || true
source /datastore/uittogether/tools/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true

# Activate conda env (điều chỉnh tên env nếu cần)
if [ -d "/datastore/uittogether/tools/miniconda3/envs/mcqgen" ]; then
    conda activate mcqgen 2>/dev/null || true
fi

cd "$DEPLOY_WEB"

echo "[$(date '+%H:%M:%S')] Starting Streamlit on port $STREAMLIT_PORT..."
echo "[$(date '+%H:%M:%S')] Working dir: $(pwd)"
echo "[$(date '+%H:%M:%S')] Python: $(which python)"
echo "[$(date '+%H:%M:%S')] Streamlit version: $(python -c 'import streamlit; print(streamlit.__version__)') 2>/dev/null || echo 'N/A'"

# Chạy Streamlit — keep alive
# Dùng exec để process thay thế shell, nhận SIGTERM đúng cách
exec streamlit run app.py \
    --server.port "$STREAMLIT_PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.browser.gatherUsageStats false \
    --logger.level info \
    2>&1 | tee "$LOG_FILE"