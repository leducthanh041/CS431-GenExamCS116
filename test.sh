sbatch --job-name=test_import --output=log/test_import_%j.out --error=log/test_import_%j.err --mem=16G --cpus-per-task=2 --time=00:05:00 --wrap='
source /datastore/uittogether/tools/miniconda3/etc/profile.d/conda.sh
conda activate cs431mcq
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Step 1: import torch"
python -u -c "import torch; print(torch.__version__)"

echo "Step 2: import transformers"
python -u -c "import transformers; print(transformers.__version__)"

echo "Step 3: import sentence_transformers"
python -u -c "from sentence_transformers import SentenceTransformer; print(\"OK\")"

echo "All done"
'