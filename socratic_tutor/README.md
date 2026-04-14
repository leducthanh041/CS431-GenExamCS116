# Socratic Tutor — QLoRA Fine-tune Pipeline

Fine-tune **Qwen2.5-14B-Instruct** theo phong cách Socratic tutoring — dẫn dắt bằng câu hỏi, không đưa đáp án trực tiếp.

---

## Pipeline tổng quan

```
Raw dialogue JSON (12 files, 560 samples)
        │
        ▼ Step 1: convert_dialogue.py
train_formatted.jsonl  (560 samples, messages format)
        │
        ▼ Step 2: filter_quality.py
train_filtered.jsonl   (551 samples, quality-checked)
        │
        ▼ Step 3: 00_train_socratic_lora.sh  (SLURM / Axolotl)
LoRA checkpoints (outputs/checkpoints/run_001/)
        │
        ▼ Step 4: 01_merge_and_eval.sh
Merged model (outputs/merged/run_001/)
        │
        ▼ Step 5: inference.py
Socratic behavior test
```

---

## Quick Start

### 1. Convert dialogues

```bash
cd socratic_tutor
python src/convert_dialogue.py \
    --input dialogue \
    --output data/formatted/train_formatted.jsonl
```

**Output:** 560 samples converted, format:
```json
{"messages": [
  {"role": "system", "content": "Bạn là Gia sư Socratic..."},
  {"role": "user",   "content": "Quan niệm của học sinh: ..."},
  {"role": "assistant", "content": "Gia sư: ...\nHọc sinh: ..."}
]}
```

### 2. Quality filter

```bash
python src/filter_quality.py \
    --input data/formatted/train_formatted.jsonl
```

**Result:** 551/560 pass (98.4%) — 9 samples rejected (tutor reveals answer too early).

### 3. Train (SLURM — A100)

```bash
# Edit CONDA_ENV in script first, then:
sbatch scripts/00_train_socratic_lora.sh
```

**Config:** `configs/qwen14b_socratic_qlora.yaml`

**Key hyperparams (A100 80GB):**
| Param | Value |
|---|---|
| Method | QLoRA (4-bit NF4) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Learning rate | 2e-4 |
| Batch size | 2 × grad_accum=8 = 16 |
| Epochs | 3 |
| Seq length | 2048 |
| Time estimate | ~3-6h |

**A100 40GB:** giảm `batch_size: 1` trong YAML.

### 4. Merge + Evaluate

```bash
bash scripts/01_merge_and_eval.sh
```

Hoặc từng bước:
```bash
# Merge adapter
python src/merge_lora.py \
    --adapter outputs/checkpoints/run_001 \
    --output outputs/merged/run_001 \
    --quantize

# Run behavior eval
python src/merge_lora.py \
    --adapter outputs/checkpoints/run_001 \
    --output outputs/merged/run_001 \
    --skip-merge --eval
```

### 5. Inference / Test

```bash
# Interactive multi-turn
python src/inference.py \
    --model outputs/merged/run_001 \
    --interactive

# Single prompt test
python src/inference.py \
    --model outputs/merged/run_001 \
    --prompt "Học sinh tin rằng '5.5' là số thực vì có dấu chấm."

# Compare base vs fine-tuned
python src/inference.py \
    --model outputs/merged/run_001 \
    --compare
```

---

## Project structure

```
socratic_tutor/
├── dialogue/               ← symlink → input/dialogues (12 tuan*.json)
├── data/
│   ├── formatted/          # train_formatted.jsonl (560 samples)
│   └── filtered/           # train_filtered.jsonl (551 samples) + rejected log
├── configs/
│   └── qwen14b_socratic_qlora.yaml   # Axolotl config
├── src/
│   ├── convert_dialogue.py # Step 1: JSON → messages format
│   ├── filter_quality.py   # Step 2: Quality filter
│   ├── merge_lora.py       # Step 4: Merge + eval
│   └── inference.py        # Step 5: Interactive inference
├── scripts/
│   ├── 00_train_socratic_lora.sh  # SLURM train
│   └── 01_merge_and_eval.sh      # Merge + eval
└── outputs/
    ├── checkpoints/run_001/      # LoRA adapters
    └── merged/run_001/           # Merged full model
```

---

## Evaluation checklist sau fine-tune

| # | Check | Criteria |
|---|---|---|
| 1 | Không đáp án trực tiếp | Model không reveal answer trước 3+ câu hỏi |
| 2 | Đủ câu hỏi dẫn dắt | ≥ 2 câu hỏi/turn của tutor |
| 3 | Tutor-to-student ratio | ~1:1 hoặc tutor chiếm nhiều hơn |
| 4 | Hội thoại multi-turn | ≥ 3 lượt tutor |
| 5 | Student tự sửa sai | Quan điểm học sinh thay đổi ở cuối |
| 6 | Tự nhiên | Không gượng ép, không template |

---

## Tham chiếu

- **Spec:** `../../socratic_tutor_pipeline_finetune.md`
- **Data:** `../../input/dialogues/` (560 Socratic dialogues)
- **Base model:** `../../models/Qwen2.5-14B-Instruct/`
- **Tieplm prompts:** `../../tieplm/backend/app/core/qa/prompts.py`

---

## Lưu ý quan trọng

- **Không học kiến thức** — chỉ học hành vi Socratic
- **Không rút gọn dialogue** khi convert
- **Quality filter rất quan trọng** — 9/560 samples bị reject (tutor reveal answer sớm)
- Model sau fine-tune **sẵn sàng kết hợp RAG** (ví dụ: bám vào CS116 transcript)
