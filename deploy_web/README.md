# 🎯 MCQGen Deploy Web — Hướng dẫn triển khai Demo Web UI

## Mục lục
1. [Cấu trúc thư mục](#1-cấu-trúc-thư-mục)
2. [Cách chạy](#2-cách-chạy)
3. [Kiến trúc](#3-kiến-trúc)
4. [Prerequisites](#4-prerequisites)
5. [Khắc phục lỗi](#5-khắc-phục-lỗi)
6. [Customize](#6-customize)

---

## 1. Cấu trúc thư mục

```
CS431MCQGen/
│
├── deploy_web/                    ← ✨ Thư mục mới — KHÔNG ảnh hưởng code gốc
│   ├── app.py                    ← Streamlit UI (giao diện chính)
│   ├── requirements.txt          ← Python dependencies cho deploy
│   ├── demo_config.py            ← Config proxy (override EXP_NAME)
│   ├── pipeline_wrappers/
│   │   └── pipeline_runner.py   ← Gọi script gốc CS431MCQGen/scripts/
│   ├── html_render/
│   │   └── mcq_renderer.py      ← Render MCQ → HTML đẹp
│   └── README.md                ← File này
│
└── (code gốc — không sửa)        ← src/, scripts/, configs/, web/
```

---

## 2. Cách chạy

### 2.1 Cài đặt dependencies

```bash
cd /datastore/uittogether/LuuTru/Thanhld/CS431MCQGen/deploy_web
pip install -r requirements.txt
```

> **Lưu ý:** `vllm`, `chromadb`, `sentence-transformers`, `rank-bm25` phải cài trong cùng
> Python environment với CS431MCQGen. Nếu đã cài rồi ở project gốc thì bỏ qua.

### 2.2 Chạy Streamlit

```bash
cd /datastore/uittogether/LuuTru/Thanhld/CS431MCQGen/deploy_web
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Sau đó mở trình duyệt: **http://<server>:8501**

### 2.3 Chạy trên SLURM (production)

```bash
# Tạo bash script chạy Streamlit làm service
#!/bin/bash
# run_streamlit.sh
cd /datastore/uittogether/LuuTru/Thanhld/CS431MCQGen/deploy_web
exec streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
```

```bash
# sbatch submission
sbatch --gres=mps:l40:2 --mem=16G --time=12:00:00 \
  --wrap="cd /datastore/uittogether/LuuTru/Thanhld/CS431MCQGen/deploy_web && streamlit run app.py --server.port 8501"
```

### 2.4 Docker (production)

```dockerfile
# deploy_web/Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy deploy_web code
COPY . .

# Copy project gốc (để gọi scripts/)
COPY ../CS431MCQGen /app/CS431MCQGen

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

```bash
docker build -t mcqgen-web -f deploy_web/Dockerfile deploy_web/
docker run -p 8501:8501 --gpus all mcqgen-web
```

---

## 3. Kiến trúc

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit App (deploy_web/app.py)            │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │ Sidebar      │  │ Run Pipeline Tab │  │ Results / History  │  │
│  │ - GPU check  │  │ - Config params  │  │ - MCQ HTML render  │  │
│  │ - Index check│  │ - Run button     │  │ - Stats summary    │  │
│  │ - Topics     │  │ - Live log       │  │ - Pagination      │  │
│  └──────────────┘  │ - Progress bar   │  └───────────────────┘  │
│                    └──────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ subprocess
┌─────────────────────────────────────────────────────────────────┐
│              PipelineRunner (pipeline_wrappers/)                 │
│  pipeline_runner.run_retrieval()     → bash scripts/02_retrieval.sh
│  pipeline_runner.run_gen_stem()      → bash scripts/03_gen_stem.sh
│  pipeline_runner.run_refine()        → bash scripts/04_gen_refine.sh
│  pipeline_runner.run_distractors()    → bash scripts/05_gen_distractors.sh
│  pipeline_runner.run_cot()           → bash scripts/06_gen_cot.sh
│  pipeline_runner.run_eval_overall()   → bash scripts/07_eval.sh
│  pipeline_runner.run_eval_iwf()      → bash scripts/08_eval_iwf.sh
│  pipeline_runner.run_explain()        → bash scripts/09_explain.sh (optional)
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ file I/O
┌─────────────────────────────────────────────────────────────────┐
│        CS431MCQGen (code gốc — KHÔNG sửa)                         │
│  scripts/0X_*.sh     ← Bash scripts gốc                          │
│  src/gen/            ← P1-P8 prompts (Qwen2.5-14B-Instruct)       │
│  src/eval/           ← eval_overall, eval_iwf (Gemma-3-12b-it)    │
│  configs/            ← generation_config.yaml                     │
│  data/intermediate/  ← Output từng step                           │
│  output/demo_*/      ← Final results (JSONL)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              MCQRenderer (html_render/)                          │
│  render_mcq_card() → 1 MCQ → Beautiful HTML card                  │
│  stats_summary()   → Tính toán pass rate, phân bổ độ khó...      │
│  render_stats_html() → Stats → HTML badge row                    │
└─────────────────────────────────────────────────────────────────┘
```

**Luồng dữ liệu:**
```
User config → PipelineRunner → bash scripts/X.sh → data/intermediate/X/
  → final_accepted_questions.jsonl → MCQRenderer → HTML → Streamlit
```

---

## 4. Prerequisites

### 4.1 GPU
```
nvidia-smi
# Cần ≥ 20GB VRAM free cho Qwen2.5-14B-Instruct
```

### 4.2 Index đã tạo
```bash
cd /datastore/uittogether/LuuTru/Thanhld/CS431MCQGen
bash scripts/01_index.sh   # Tạo ChromaDB index từ slide + transcript
```

### 4.3 Python packages
```bash
pip install vllm chromadb sentence-transformers rank-bm25 streamlit
```

### 4.4 Kiểm tra môi trường
Mở Streamlit app → Sidebar hiển thị tự động:
- ✅ GPU + VRAM
- ✅ ChromaDB Index
- ✅ vLLM, ChromaDB, sentence-transformers, rank-bm25
- ✅ Topic list

---

## 5. Khắc phục lỗi

| Lỗi | Nguyên nhân | Cách xử lý |
|-----|------------|-----------|
| `GPU out of memory` | VRAM đầy khi chạy model | Giảm `num_candidates` slider, chạy `torch.cuda.empty_cache()` |
| `Script not found` | Working directory sai | Luôn chạy `streamlit` từ `CS431MCQGen/deploy_web/` |
| `ChromaDB not found` | Index chưa tạo | Chạy `bash scripts/01_index.sh` |
| `vllm import error` | Chưa cài vLLM | `pip install vllm` |
| Empty results | Topic filter loại hết | Thử để trống chapter filter |
| Streamlit port 8501 busy | Port đã dùng | Đổi port: `--server.port 8502` |
| Permission denied | Không có quyền GPU | Liên hệ admin hoặc dùng `sbatch` |

### GPU Memory Full — giải pháp nhanh
```bash
# Giải phóng VRAM
python -c "import torch; torch.cuda.empty_cache(); print('VRAM cleared')"

# Kiểm tra
nvidia-smi
```

### Rebuild index (nếu cần cập nhật slide/transcript)
```bash
cd /datastore/uittogether/LuuTru/Thanhld/CS431MCQGen
bash scripts/01_index.sh
```

---

## 6. Customize

### 6.1 Đổi port mặc định
```bash
streamlit run app.py --server.port 8502
```

### 6.2 Thêm chapter filter mới
Sửa `app.py` → `topic_filter = st.multiselect(...)` và truyền xuống `pipeline_runner.py`.

### 6.3 Đổi màu sắc UI
Sửa `html_render/mcq_renderer.py` → thay đổi mã màu hex trong các hàm `render_mcq_card()` và `render_stats_html()`.

### 6.4 Thêm bước Pipeline mới
1. Thêm method mới vào `pipeline_wrappers/pipeline_runner.py`
2. Thêm vào `run_full_pipeline()` list
3. Gọi từ `app.py` → tab Run Pipeline

### 6.5 Đổi giao diện (theme)
```bash
# Dark mode
streamlit run app.py --theme.base dark

# Light mode (mặc định)
streamlit run app.py --theme.base light
```

---

## 📌 Tóm tắt nhanh

```bash
# 1 dòng để chạy
cd /datastore/uittogether/LuuTru/Thanhld/CS431MCQGen/deploy_web
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```
