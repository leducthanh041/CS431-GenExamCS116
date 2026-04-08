#!/bin/bash
# download_models.sh — Download Qwen2.5-14B-Instruct + Gemma-3-12b-it từ HuggingFace
# Chạy: bash download_models.sh

set -e

DEST="/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen/models"
mkdir -p "$DEST"

echo "📦 Downloading Qwen2.5-14B-Instruct..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-14B-Instruct',
    local_dir='$DEST/Qwen2.5-14B-Instruct',
    local_dir_use_symlinks=False,
)
print('✅ Qwen2.5-14B-Instruct downloaded')
" 2>&1 | tee "$DEST/download_qwen.log"

echo ""
echo "📦 Downloading gemma-3-12b-it..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='google/gemma-3-12b-it',
    local_dir='$DEST/gemma-3-12b-it',
    local_dir_use_symlinks=False,
)
print('✅ gemma-3-12b-it downloaded')
" 2>&1 | tee "$DEST/download_gemma.log"

echo ""
echo "✅ All models downloaded to: $DEST"
du -sh "$DEST"/*/
