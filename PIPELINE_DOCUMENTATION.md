# CS431MCQGen — Pipeline Documentation
## Hệ thống sinh câu hỏi trắc nghiệm tự động cho CS116

> **Phiên bản:** 2026-04-15
> **Nguồn tham chiếu:** Bài báo MCQGen (IEEE ACCESS 2024) — RAG + Self-Refine + CoT Distractor Selection
> **Môn học:** CS116 — Lập trình Python cho Máy học, Trường ĐH Công nghệ Thông tin, ĐHQG-HCM

---

## Mục lục

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Input của pipeline](#2-input-của-pipeline)
3. [Kiến trúc 11 bước chi tiết](#3-kiến-trúc-11-bước-chi-tiết)
4. [Output của mỗi bước](#4-output-của-mỗi-bước)
5. [Schema JSON của các file trung gian](#5-schema-json-của-các-file-trung-gian)
6. [Mô hình AI sử dụng](#6-mô-hình-ai-sử-dụng)
7. [Định dạng câu hỏi MCQ](#7-định-dạng-câu-hỏi-mcq)
8. [Cấu trúc thư mục project](#8-cấu-trúc-thư-mục-project)
9. [Cách chạy pipeline](#9-cách-chạy-pipeline)
10. [Evaluation metrics](#10-evaluation-metrics)
   10a. [Metrics đã loại bỏ](#103-metrics-đã-loại-bỏ)
11. [Web deployment architecture](#11-web-deployment-architecture)

---

## 1. Tổng quan hệ thống

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    CS431MCQGen — 11-Step Pipeline                         ║
╚══════════════════════════════════════════════════════════════════════════╝

  INPUT LAYER (offline / one-time)
  ─────────────────────────────────
  [Slide PDFs]  ──→ Step 01 ──→ ChromaDB + concept_chunks.jsonl
  [Video MP4]   ──→ Step 01b ──→ transcript_chunks.jsonl (Whisper)
  [Đề thi PDF]  ──→ Step 01 ──→ assessment_items.jsonl (future)

  RAG LAYER
  ─────────
  [topic_list.json] ──→ Step 02 ──→ Hybrid Retrieval (5-tier RAG)
                                    • BM25 lexical
                                    • ChromaDB vector (BGE-m3)
                                    • RRF fusion (k=60)
                                    • Cross-encoder rerank
                                    • Metadata enrichment

  GENERATION LAYER (P1–P8)
  ────────────────────────
  Step 03: P1 — Generate Stem + Key answers          [Qwen2.5-14B]
  Step 04: P2+P3 — Self-Refine (suggest → apply)     [Qwen2.5-14B]
  Step 05: P4 — Generate 6 distractor candidates      [Qwen2.5-14B]
  Step 06: P5–P8 — CoT Distractor Selection          [Qwen2.5-14B]
            P5: Evaluate each candidate
            P6: Remove irrelevant/grammatically wrong
            P7: Select final distractors (3 best)
            P8: Assemble MCQ + randomize A/B/C/D

  EVALUATION LAYER
  ─────────────────
  Step 07: Overall MCQ evaluation (8 criteria)        [Gemma-3-12b]
  Step 08: Distractor IWF analysis                   [Gemma-3-12b]
             • plausible_distractor, vague_terms, grammar_clue
             • absolute_terms, distractor_length, k_type_combination

  POST-PROCESSING LAYER
  ─────────────────────
  Step 09: Explanation generation (correct rationale + citations) [Qwen2.5-14B]
  Step 10: Quantitative metrics computation

  OUTPUT LAYER
  ───────────
  final/accepted_questions.jsonl  → HTML render → giao cho sinh viên
```

### Data flow tổng quan

```
Whisper ASR
    ↓
transcript_chunks_with_timestamps.jsonl  (Step 01b)
    ↓
concept_chunks.jsonl  +  ChromaDB (Step 01)
    ↓
topic_list.json  +  HybridRetriever  (Step 02)
    ↓  ← context_blocks cho mỗi topic
P1 → P2+P3 → P4 → P5+P6+P7+P8  (Steps 03-06)
    ↓
all_final_mcqs.jsonl
    ↓
eval_overall (8 criteria)  → evaluated_questions.jsonl  (Step 07)
    ↓
eval_iwf (6 IWF types) → final_accepted/rejected  (Step 08)
    ↓
explain_mcq (rationale + citations)  → explanations.jsonl  (Step 09)
    ↓
Metrics  → eval_metrics.json  (Step 10)
    ↓
HTML render  → đề thi hoàn chỉnh
```

---

## 2. Input của pipeline

### 2.1 Dữ liệu thô (Raw Data)

```
input/
├── slide/                        # Slide bài giảng (PDF)
│   ├── CS116-Bai02-Popular Libs.pdf
│   ├── CS116-Bai03-Pipeline & EDA.pdf
│   ├── CS116-Bai04-Data preprocessing.pdf
│   ├── CS116-Bai05-Eval model.pdf
│   ├── CS116-Bai06-Unsupervised learning.pdf
│   ├── CS116-Bai07a-Supervised learning-Regression.pdf
│   ├── CS116-Bai07b-Supervised learning-Classification.pdf
│   ├── CS116-Bai08-Deep learning với CNN.pdf
│   ├── CS116-Bai09-Parameter tuning.pdf
│   ├── CS116-Bai10-Ensemble model.pdf
│   └── CS116-Bai11-Model Deployment.pdf
│
├── video/                        # Video bài giảng
│   ├── videos1.txt              # YouTube URL mapping (chapter → URL)
│   └── *.mp4                    # Video files (để Whisper transcribe)
│
├── transcribe_data/              # Whisper JSON output (ASR)
│   ├── 1.1.json, 1.2.json ...   # Mỗi file = 1 video segment
│   └── 4.1.json ... 11.x.json  # có word-level timestamps
│
├── video_transcript/            # Transcript text thuần (backup)
│   └── *.txt
│
├── question_bank/              # Câu hỏi sẵn có (để reference style)
│   ├── multiple_choice/
│   ├── essay_questions/
│   └── coding/
│
├── sample_exams/               # Đề thi mẫu PDF
│   └── *.pdf
│
├── topic_list.json            # ★ Danh sách topics cần sinh câu hỏi
│                              # Cấu trúc: chapters[] → topics[] (xem bên dưới)
│
├── vietnamese-stopwords.txt   # Stopwords cho BM25
│
└── render_exam_template.html  # Template HTML để render đề thi
```

### 2.2 File cấu hình (Configs)

```
configs/
├── mcqgen_config.yaml          # Cấu hình toàn pipeline (chapters, topics, MCQ format)
├── generation_config.yaml       # Generation params (target range, ratios, weights)
├── generation_config_active.yaml # Override từ web app / CLI (tạm thời)
└── bloom_taxonomy.yaml         # Bloom's Taxonomy levels + keywords
```

### 2.3 Topic List Schema

`input/topic_list.json` là **file đầu vào quan trọng nhất** — định nghĩa WHAT to generate:

```json
[
  {
    "chapter_id": "ch04",
    "chapter_name": "Tiền xử lý dữ liệu",
    "topics": [
      {"topic_id": "ch04_t01", "topic_name": "Missing Data", "difficulty": "G2"},
      {"topic_id": "ch04_t02", "topic_name": "Outlier Detection", "difficulty": "G2"},
      {"topic_id": "ch04_t03", "topic_name": "Feature Extraction", "difficulty": "G3"},
      {"topic_id": "ch04_t04", "topic_name": "Feature Transformation", "difficulty": "G2"},
      {"topic_id": "ch04_t05", "topic_name": "Feature Selection", "difficulty": "G3"}
    ]
  },
  {
    "chapter_id": "ch07b",
    "chapter_name": "Supervised Learning - Classification",
    "topics": [
      {"topic_id": "ch07b_t01", "topic_name": "Logistic Regression", "difficulty": "G2"},
      {"topic_id": "ch07b_t02", "topic_name": "Decision Trees", "difficulty": "G2"},
      {"topic_id": "ch07b_t03", "topic_name": "SVM", "difficulty": "G3"}
    ]
  }
]
```

### 2.4 Đặc điểm dữ liệu quan trọng

| Đặc điểm | Chi tiết |
|---|---|
| **Ngôn ngữ** | Tiếng Việt hoàn toàn |
| **Số đáp án đúng** | 1, 2, hoặc 3 đáp án đúng |
| **Định dạng đề thi** | `[Một đáp án đúng]` hoặc `[Nhiều đáp án đúng]` |
| **Độ khó** | G1 (nhớ), G2 (áp dụng), G3 (phân tích/đánh giá) |
| **Bloom's Taxonomy** | G1→L1+L2, G2→L3+L4, G3→L5+L6 |
| **Số câu hỏi mục tiêu** | 25–35 câu (configurable) |
| **Tỉ lệ single/multiple** | 80% single / 20% multiple |

---

## 3. Kiến trúc 11 bước chi tiết

---

### Step 00 — Whisper Transcription *(offline, một lần)*

```
Input:  video/*.mp4
Output: input/transcribe_data/*.json (Whisper JSON với word-level timestamps)
Script: transcribe_videos.py / run_transcribe.sh
```

**Xử lý:**
1. Mỗi video MP4 được transcribe bằng Whisper (mô hình `medium` hoặc `large`)
2. Output: JSON với `segments[]` → `words[]` có `word`, `start`, `end`
3. File naming: `X.Y.json` → chapter X, segment Y

**Tại sao cần word-level timestamps:**
- Để map transcript chunks → slide page → YouTube URL
- Phục vụ trích dẫn chính xác trong explanation (bấm link nhảy đúng giây)

---

### Step 01 — Indexing: Slide PDF + Transcript → ChromaDB

```
Input:  input/slide/*.pdf
        input/transcribe_data/*.json  (từ Step 00)
        input/video/videos1.txt        (YouTube URL mapping)
Output: data/processed/concept_chunks.jsonl
        data/indexes/ (ChromaDB persistent)
Script: src/gen/indexing.py
```

**Xử lý 2 nguồn:**

#### 1a. Slide PDF indexing
```
For mỗi PDF trong slide/:
  For mỗi trang:
    Trích xuất text (PyMuPDF / fitz)
    Tách section_title (dòng đầu tiên) + body text
    Gán metadata: chapter_id, page_number, topics[]
    → 1 chunk = 1 trang slide
    chunk_id format: cs116_{chapter_id}_slide_p{page:03d}
```

#### 1b. Transcript chunking *(Step 01b — chunk_transcript_with_timestamps.py)*
```
Input: input/transcribe_data/*.json (Whisper JSON)
       input/video/videos1.txt       (URL mapping)

Xử lý:
  1. Parse videos1.txt → build VIDEO_META_MAP
     Format: "chapter_num|YouTube_URL[, slide: filename.pdf, trang N]"
     → (chapter_id, sub_index) → {url, slide_file, slide_start_page}

  2. Parse Whisper JSON → flat list of {word, start, end}

  3. Deduplicate: loại bỏ cụm từ lặp do lỗi nối file ASR
     VD: "đây đây đây" → "đây"

  4. Chunk by words: target=200 words, min=80, overlap=30
     → Gom câu nguyên vẹn (sentence boundary detection)

  5. Map timestamp → YouTube URL + slide page từ videos1.txt

  Output: transcript_chunks_with_timestamps.jsonl
  chunk_id format: cs116_{chapter_id}_transcript_{video_sub}_s{seq:03d}
```

#### 1c. Embed & Store in ChromaDB
```
Mô hình embedding: BAAI/bge-m3  (1024-dim, multilingual Vietnamese)
Chiến lược chunking: 300-500 tokens/chunk
Lưu trữ: ChromaDB PersistentClient (data/indexes/)

Metadata lưu kèm trong ChromaDB:
  • chapter_id, chapter_title, topics[]
  • source_type: "slide_pdf" | "video_transcript"
  • youtube_url, youtube_ts_start, youtube_ts_end  (cho video)
  • slide_file, slide_start_page                  (cho slide)
  • timestamp_start, timestamp_end                  (cho video)
```

---

### Step 02 — Hybrid Retrieval: RAG 5-tier

```
Input:  input/topic_list.json
        data/indexes/ (ChromaDB)
        data/processed/concept_chunks.jsonl (BM25 source)
Output: output/{EXP_NAME}/02_retrieval/{topic_id}.jsonl
Script: src/gen/retrieval.py
```

**5-Tier Hybrid Retrieval Pipeline:**

```
Query expansion (TOPIC_KEYWORDS_MAP):
  "Missing Data" → "missing data dữ liệu thiếu imputation KNNImputer dropna..."

  Tier 1: BM25 Lexical Search
    └─→ Top-150 results from concept_chunks.jsonl text

  Tier 2: ChromaDB Vector Search (BGE-m3)
    └─→ Top-150 results, optional chapter filter

  Tier 3: RRF Fusion (k=60)
    └─→ Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank)
    └─→ Kết hợp cả BM25 và vector → Top-150 fused

  Tier 4: Cross-encoder Reranking
    └─→ Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    └─→ Re-score top-30 fused results vs query
    └─→ Sort by cross-encoder score

  Tier 5: Metadata Enrichment
    └─→ Attach youtube_url, slide_file, timestamps
    └─→ Fallback enrich từ chunk_map JSONL nếu ChromaDB thiếu
```

**Per-topic output:**
```json
{
  "topic_id": "ch04_t01",
  "topic_name": "Missing Data",
  "query": "missing data imputation KNNImputer...",
  "num_context_blocks": 5,
  "context_blocks": [...],      // Top-5 enriched blocks
  "context_blocks_str": "..."   // Concatenated string cho prompt
}
```

**Tại sao 5-tier:**
- BM25 bắt exact keyword matches
- Vector search bắt semantic similarity
- RRF kết hợp cả hai — robust hơn
- Cross-encoder tinh chỉnh ranking cuối cùng
- Metadata enrichment đảm bảo citations đầy đủ

---

### Step 03 — P1: Generate Stem + Key *(Generation Layer, bước 1)*

```
Input:  output/{EXP_NAME}/02_retrieval/{topic_id}.jsonl
        input/topic_list.json
Output: output/{EXP_NAME}/03_gen_stem/all_p1_results.jsonl
Script: src/gen/p1_gen_stem.py
Model:  Qwen2.5-14B-Instruct (vLLM)
```

**Xử lý:**
```
For mỗi topic trong topic_list.json:
  Đọc retrieval result → context_blocks_str

  Tính question mix cho topic đó:
    VD: num_questions=3, single_correct_ratio=0.8
    → 2 single_correct (đúng 1 đáp án)
    → 1 multiple_correct (đúng 2 đáp án)

  For mỗi question trong batch:
    Gọi LLM với prompt P1:

    P1 Prompt chứa:
    • HARD CONSTRAINT: question_type + correct_answer_count
    • EXAM STYLE BLOCK: định nghĩa phong cách đề thi
    • DIVERSITY RULE: tránh các cách mở đầu yếu
    • CONCEPT CONTEXT: context từ retrieval
    • STRICT RULE: KHÔNG ghi đáp án vào stem
    • BATCH CONTEXT: số câu single/multiple cần sinh

  Output: JSON với question_text, correct_answers_content[],
          question_type, correct_answer_count, difficulty_label
```

**Validation sau P1:**
- Force `question_type` và `correct_answer_count` khớp với yêu cầu
- Fix số lượng correct_answers_content nếu LLM sai count
- Skip nếu parse failed

**P1 Prompt Design Principles:**
- Stem không chứa đáp án (tránh vi phạm hard constraint)
- Nhiều cách mở đầu đa dạng
- Bám sát context từ slide + transcript
- Phù hợp ngữ cảnh đề thi sinh viên đại học

---

### Step 04 — P2+P3: Self-Refine Stem *(Generation Layer, bước 2)*

```
Input:  output/{EXP_NAME}/03_gen_stem/all_p1_results.jsonl
Output: output/{EXP_NAME}/04_gen_refine/all_refined_results.jsonl
Script: src/gen/p2_p3_refine.py
Model:  Qwen2.5-14B-Instruct (vLLM)
```

**Self-Refine Chain:**

```
P1 stem ──→ P2: Propose improvement suggestion
               (LLM phê bình bản thân)
                   │
                   ▼
               P3: Apply refinement
               (LLM viết lại stem dựa trên gợi ý)
                   │
                   ▼
           Refined stem (giữ nguyên correct_answers_content)
```

**P2 — Propose improvement:**
```
Input: draft_stem_key JSON + difficulty_target
Task:  Đề xuất 1 cách cải tiến cụ thể, hành động được
Output: {improvement_suggestion, why_it_is_exam_appropriate, difficulty_effect}
```

**P3 — Apply refinement:**
```
Input: P1 stem + P2 suggestion
Task:  Viết lại stem theo gợi ý, GIỮ NGUYÊN đáp án đúng
Output: {refined_question_text, correct_answers_content[], ...}
```

**Fallback:** Nếu P3 parse failed → dùng lại P1 stem

---

### Step 05 — P4: Generate Distractor Candidates *(Generation Layer, bước 3)*

```
Input:  output/{EXP_NAME}/04_gen_refine/all_refined_results.jsonl
Output: output/{EXP_NAME}/05_gen_distractors/all_candidates_results.jsonl
Script: src/gen/p4_candidates.py
Model:  Qwen2.5-14B-Instruct (vLLM)
```

**Xử lý:**
```
For mỗi refined stem:
  Đếm số distractor cần: 4 - correct_answer_count
  VD: correct_answer_count=2 → cần 2 distractors

  Gọi LLM với prompt P4:
    • Input: refined_stem (từ P3)
    • Output: 6 distractor candidates
    • Ràng buộc:
      - Sai NHƯNG HỢP LÝ (đánh trúng lỗi sinh viên hay mắc)
      - Không paraphrase quá gần với correct answers
      - Không dùng "Tất cả đáp án trên", "Không đáp án nào đúng"
      - Không grammar clue, absolute terms, độ dài bất thường

  Output: {candidate_distractors: [6 items]}
```

---

### Step 06 — P5–P8: CoT Distractor Selection *(Generation Layer, bước 4)*

```
Input:  output/{EXP_NAME}/05_gen_distractors/all_candidates_results.jsonl
Output: output/{EXP_NAME}/06_gen_cot/all_final_mcqs.jsonl
Script: src/gen/p5_p8_cot.py
Model:  Qwen2.5-14B-Instruct (vLLM)
```

**Chain-of-Thought Pipeline cho mỗi MCQ:**

```
6 distractor candidates
    │
    ▼
P5: Evaluate each candidate  (relevance, misleading_likelihood, grammar, logic)
    │
    ▼
P6: Remove bad distractors
    • Loại bỏ nếu: misleading_likelihood≤2, relevance≤3,
      grammar_ok=false, absolute_term=true
    • Giữ lại distractors còn lại (kept_options)
    │
    ▼
P7: Select final distractors
    • Chọn đúng num_distractors_needed (thường 3)
    • Ưu tiên: misleading_likelihood cao + relevance cao
    • Đảm bảo distinct error types (không chọn 2 distractors giống nhau)
    │
    ▼
P8: Assemble MCQ
    • Gán A/B/C/D cho 4 options (correct + distractors)
    • RANDOMIZE thứ tự: xáo trộn để tránh bias
    • Gắn nhãn [Một đáp án đúng] hoặc [Nhiều đáp án đúng]
    • Output: question_text, options{A/B/C/D}, correct_answers[], ...
```

**Fallback assembly:** Nếu P8 parse failed → deterministic shuffle + gán A/B/C/D

---

### Step 07 — Overall Evaluation *(Evaluation Layer, bước 1)*

```
Input:  output/{EXP_NAME}/06_gen_cot/all_final_mcqs.jsonl
Output: output/{EXP_NAME}/07_eval/evaluated_questions.jsonl
        output/{EXP_NAME}/07_eval/failed_questions.jsonl
Script: src/eval/eval_overall.py
Model:  Gemma-3-12b-it (vLLM)
```

**8 Evaluation Criteria:**

| # | Criterion | Mô tả | Threshold |
|---|---|---|---|
| 1 | `format_pass` | Đúng 4 options A/B/C/D, có nhãn loại | ✅/❌ |
| 2 | `language_pass` | Tiếng Việt tự nhiên, không lỗi chính tả | ✅/❌ |
| 3 | `grammar_pass` | Các options cùng ngữ pháp, độ dài tương đương | ✅/❌ |
| 4 | `relevance_pass` | Câu hỏi và tất cả options liên quan đến topic | ✅/❌ |
| 5 | `answerability_pass` | SV đủ thông tin trong stem để trả lời | ✅/❌ |
| 6 | `correct_set_pass` | Đáp án đúng hợp lý, không dư thừa, không mâu thuẫn | ✅/❌ |
| 7 | `no_four_correct_pass` | **HARD REJECT**: Không có 4 options nào đều đúng | ✅/❌ |
| 8 | `answer_not_in_stem_pass` | **HARD REJECT**: Đáp án không xuất hiện trong stem | ✅/❌ |

**Decision:**
- `overall_valid = true` → accepted → `evaluated_questions.jsonl`
- `overall_valid = false` → rejected → `failed_questions.jsonl`
- Parse error → fallback to pass (để tránh reject quá nhiều)

---

### Step 08 — Distractor IWF Analysis *(Evaluation Layer, bước 2)*

```
Input:  output/{EXP_NAME}/07_eval/evaluated_questions.jsonl
Output: output/{EXP_NAME}/08_eval_iwf/final_accepted_questions.jsonl
        output/{EXP_NAME}/08_eval_iwf/final_rejected_questions.jsonl
Script: src/eval/eval_iwf.py
Model:  Gemma-3-12b-it (vLLM)
```

**6 Item Writing Flaws (IWF) Types:**

| # | IWF Type | Mô tả | Lỗi nếu |
|---|---|---|---|
| 1 | `plausible_distractor` | Distractor có lỗi phổ biến ở sinh viên không? | Quá rõ ràng là sai |
| 2 | `vague_terms` | Có từ ngữ mơ hồ, không rõ ràng? | Mơ hồ, không xác định được |
| 3 | `grammar_clue` | Lỗi ngữ pháp gây clue cho đáp án đúng? | Grammar không khớp stem |
| 4 | `absolute_terms` | Có từ tuyệt đối gây clue? ("luôn", "tất cả", "không bao giờ") | Có → dễ loại |
| 5 | `distractor_length` | Độ dài khác biệt lớn so với options khác? | Quá dài/ngắn |
| 6 | `k_type_combination` | (multiple_correct) Đáp án đúng phù hợp kiểu K không? | Kiểu K không đúng |

**Decision:**
- `total_iwf_count ≤ 3` AND `overall_distractor_quality_pass = true` → accepted
- Ngược lại → rejected

---

### Step 09 — Explanation Generation *(Post-processing)*

```
Input:  output/{EXP_NAME}/08_eval_iwf/final_accepted_questions.jsonl
Output: output/{EXP_NAME}/09_explain/explanations.jsonl
Script: src/gen/explain_mcq.py
Model:  Qwen2.5-14B-Instruct (vLLM)
```

**Mỗi câu hỏi được bổ sung 4 phần explanation:**

```
1. question_motivation:
   → TẠI SAO ra câu hỏi này? Kiểm tra khái niệm gì?
     Vì sao SV cần nắm vững? Tại sao hỏi theo cách này?

2. correct_answer_rationale:
   → Tại sao đáp án đúng là đúng?
   → Kèm TRÍCH DẪN cụ thể: slide file + số trang, video YouTube URL

3. distractor_explanations:
   → Tại sao từng distractor là SAI?
   → Confusion point + trích dẫn

4. knowledge_context:
   → topic_scope, prerequisites, advanced_knowledge, learning_value
```

**Citation Building Logic:**

```
Sources cho mỗi câu hỏi:
  • Slides: tối đa 4 slides, ưu tiên slides cùng chapter với câu hỏi
  • Video: 1 video YouTube, được chọn từ slide đầu tiên trong context

YouTube URL mapping:
  videos1.txt: "chapter_num|yt_url|slide: filename.pdf, trang N"
  → Build map: (chapter_id, file, page) → YouTube URL

  Priority:
  1. Slide cùng chapter với câu hỏi → video tương ứng
  2. Slide cross-chapter → fallback
  3. Không có slides → dùng chapter video đầu tiên

  Timestamp từ video_transcript block (hybrid retrieval)
  Format: https://youtu.be/...&t=90  (nhảy đúng giây)
```

**Lưu ý:** KHÔNG dùng external web search — chỉ dùng course material nội bộ

---

### Step 10 — Quantitative Metrics *(Evaluation)*

```
Input:  output/{EXP_NAME}/08_eval_iwf/final_accepted_questions.jsonl
        output/{EXP_NAME}/07_eval/evaluated_questions.jsonl
        input/topic_list.json
        output/{EXP_NAME}/human_annotation.json     (optional)
Output: output/{EXP_NAME}/metrics_report.json
        output/{EXP_NAME}/metrics_report.md
Script: src/eval/eval_metrics.py
```

**4 Core Metrics (automatic, no reference set required):**

| # | Metric | Công thức | Target |
|---|---|---|---|
| 1 | **Topic Coverage** | topics covered / total topics in topic_list.json | ≥ 80% |
| 2 | **LLM Judge Pass Rate** | accepted / total evaluated | ≥ 70% |
| 3 | **Bloom KL Divergence** | KL(actual_bloom_dist ∥ target_bloom_dist) | ≤ 0.3 |
| 4 | **Human Judgment** (κ, F1) | human vs LLM agreement from JSON annotation | κ ≥ 0.6 |

**Nội dung chi tiết từng metric:**

#### Metric 1 — Topic Coverage
```
Input:  final_accepted_questions.jsonl  (câu hỏi đã chấp nhận)
        topic_list.json                   (danh sách topics tham chiếu)

Logic:
  1. Extract all topic names từ accepted questions → set A
  2. Extract all topic names từ topic_list.json      → set B
  3. coverage_ratio = |A ∩ B| / |B|
  4. Per-chapter breakdown

Output dict fields:
  coverage_ratio, topics_covered[], topics_missing[],
  num_covered, num_total, extra_topics[], chapter_coverage{}
```

#### Metric 2 — LLM Judge Pass Rate
```
Input:  evaluated_questions.jsonl   (Step 07 output — đã qua Gemma-3-12b)
        final_accepted_questions.jsonl  (Step 08 output — đã qua IWF filter)

Logic:
  1. overall: accepted / total_evaluated
  2. Per-criterion pass rate (8 criteria từ Step 07)
  3. IWF pass rate + per-type (6 IWF flaws từ Step 08)
  4. Quality score stats: mean, std, median, min, max
  5. Per-difficulty breakdown: G1/G2/G3 pass rates

Output dict fields:
  final_pass_rate, total_evaluated, total_accepted, total_rejected,
  criterion_pass_rates{}, iwf_pass_rate, iwf_type_pass_rates{},
  quality_score_stats{}, per_difficulty_rates{}
```

#### Metric 3 — Bloom KL Divergence
```
Input:  final_accepted_questions.jsonl  (câu hỏi đã chấp nhận)

Logic:
  1. Classify each question stem vào Bloom level 1–6 bằng keyword matching
     (BLOOM_KEYWORDS dict: L1→"nhớ/định nghĩa", L3→"áp dụng/tính toán", v.v.)
  2. Count bloom distribution: actual_dist[]
  3. Target distribution: G1→{L1,L2}=20% each, G2→{L3,L4}=20% each,
                         G3→{L5,L6}=10% each
  4. KL_div = KL(actual_dist ∥ target_dist) via scipy.stats.entropy

Output dict fields:
  bloom_counts{1..6}, actual_distribution[], target_distribution[],
  kl_divergence, per_difficulty_bloom{}, per_question[]
```

#### Metric 4 — Human Judgment (Human vs LLM Agreement)
```
Input:  human_annotation.json   (reviewer annotate per-question, JSON schema)
        evaluated_questions.jsonl  (LLM verdict từ Step 07)
        final_accepted_questions.jsonl  (LLM IWF verdict từ Step 08)

Output: {EXP}/human_annotation.with_llm_comparison.json
        (auto-saved, hoặc chỉ định qua output_path param)

Hỗ trợ **2 format** annotation JSON:

**Format A — HTML export** (từ `render_review_html.py` → nhấn "Export JSON"):

```json
{
  "annotator": "Nguyen Van A",
  "timestamp": "2026-04-15T10:30:00.000Z",
  "total_annotated": 10,
  "verdicts": {
    "<question_id>": {
      "format_pass": true,
      "language_pass": true,
      "grammar_pass": true,
      "relevance_pass": true,
      "answerability_pass": true,
      "correct_set_pass": true,
      "overall_valid": true
    }
  }
}
```

→ Auto-detected bởi key `"verdicts"`. So sánh trực tiếp từng criterion với LLM verdict.

**Format B — Standalone annotation JSON** (viết tay hoặc custom tool):

```json
{
  "annotator": "Nguyen Van A",
  "date": "2026-04-15",
  "questions": {
    "<question_id>": {
      "overall_judgment": "accept" | "reject",
      "criteria": {
        "format_pass":             true | false,
        "language_pass":            true | false,
        "grammar_pass":             true | false,
        "relevance_pass":           true | false,
        "answerability_pass":       true | false,
        "correct_set_pass":         true | false,
        "no_four_correct_pass":     true | false,
        "answer_not_in_stem_pass":   true | false
      },
      "distractor_quality": {
        "iwf_overall":              "accept" | "reject",
        "plausible_distractor":     true | false | null,
        "vague_terms":              true | false | null,
        "grammar_clue":             true | false | null,
        "absolute_terms":           true | false | null,
        "distractor_length":        true | false | null,
        "k_type_combination":       true | false | null
      },
      "notes": "..."
    }
  }
}
```

→ Auto-detected bởi key `"questions"`. Full breakdown: 8 criteria + 6 IWF flaws.

Logic (4 sub-sections):
  4a. Overall accept/reject:
        → Confusion Matrix (TP/TN/FP/FN) + Cohen's κ + P/R/F1
  4b. Per-criterion (8 criteria):
        → Cohen's κ + agreement_rate per criterion + per-question match table
  4c. IWF overall:
        → Confusion Matrix + Cohen's κ for distractor quality accept/reject
  4d. IWF per-type (6 IWF flaws):
        → Pass rate (% distractors with no flaw) per type

Output dict fields:
  meta{annotator, date, n_annotated, n_matched, n_unmatched},
  overall{CM + κ + P/R/F1},
  per_criterion{κ + agreement per criterion},
  iwf_overall{CM + κ},
  iwf_per_type{pass rates per IWF flaw},
  per_question[detail với human verdict, LLM verdict, mismatch],
  disagreement_analysis[mismatched questions list],
  disagreement_summary{criterion: count}

Cohen's κ interpretation:
  < 0.00: Poor
  0.00–0.20: Slight
  0.20–0.40: Fair
  0.40–0.60: Moderate
  0.60–0.80: Substantial
  ≥ 0.80: Almost Perfect

Run:
```bash
# Chạy core metrics + human judgment
python src/eval/eval_metrics.py \
    --exp "exp_04" \
    --human-json output/exp_04/review/eval_review_annotations.json

# Output tự động:
#   output/exp_04/metrics_report.json
#   output/exp_04/metrics_report.md
#   output/exp_04/review/eval_review_annotations_with_llm_comparison.json  (auto-saved)
```

Cách workflow đầy đủ:
```
1. python scripts/render_review_html.py → output/{EXP}/review/eval_review.html
2. Mở HTML trong browser → review câu hỏi → nhấn "Export JSON"
3. → eval_review_annotations.json
4. python src/eval/eval_metrics.py --human-json eval_review_annotations.json
```

**Các metrics ĐÃ LOẠI BỎ** (không còn dùng):
| Metric | Lý do loại bỏ |
|---|---|
| **BLEU + ROUGE-L** | Cần reference set — không phù hợp open-ended MCQ generation |
| **BERTScore** | Cần reference set + tốn thêm LLM call |
| **Answer Ratio** | Inline trong generation config, không cần metric riêng |
| **Diversity Openings** | Metric chủ quan, không đo lường chất lượng thực |
| **Old Cohen's κ (CSV-based)** | Schema rải rác 3 hàm riêng → gộp vào `compute_human_judgment()` unified |

---

### Step 11 — HTML Rendering *(Output)*

```
Input:  output/{EXP_NAME}/09_explain/explanations.jsonl
Output: output/{EXP_NAME}/final/exam_final.html
Script: render_explanations_html.py / render_final_html.py
```

**Render features:**
- Câu hỏi theo thứ tự chapter
- Đáp án đúng ẩn (toggle hiện/ẩn)
- Explanation hiện khi click
- YouTube links nhúng timestamp
- Slide citations

---

## 4. Output của mỗi bước

| Step | Script | Output File | Format |
|---|---|---|---|
| 01 | `indexing.py` | `data/processed/concept_chunks.jsonl` | JSONL |
| 01 | `indexing.py` | `data/indexes/` (ChromaDB) | SQLite + binary |
| 01b | `chunk_transcript_with_timestamps.py` | `data/processed/transcript_chunks_with_timestamps.jsonl` | JSONL |
| 02 | `retrieval.py` | `output/{EXP}/02_retrieval/{topic_id}.jsonl` | JSONL |
| 03 | `p1_gen_stem.py` | `output/{EXP}/03_gen_stem/all_p1_results.jsonl` | JSONL |
| 04 | `p2_p3_refine.py` | `output/{EXP}/04_gen_refine/all_refined_results.jsonl` | JSONL |
| 05 | `p4_candidates.py` | `output/{EXP}/05_gen_distractors/all_candidates_results.jsonl` | JSONL |
| 06 | `p5_p8_cot.py` | `output/{EXP}/06_gen_cot/all_final_mcqs.jsonl` | JSONL |
| 07 | `eval_overall.py` | `output/{EXP}/07_eval/evaluated_questions.jsonl` | JSONL |
| 07 | `eval_overall.py` | `output/{EXP}/07_eval/failed_questions.jsonl` | JSONL |
| 08 | `eval_iwf.py` | `output/{EXP}/08_eval_iwf/final_accepted_questions.jsonl` | JSONL |
| 08 | `eval_iwf.py` | `output/{EXP}/08_eval_iwf/final_rejected_questions.jsonl` | JSONL |
| 09 | `explain_mcq.py` | `output/{EXP}/09_explain/explanations.jsonl` | JSONL |
| 10 | `eval_metrics.py` | `output/{EXP}/metrics_report.json` | JSON |
| 10 | `eval_metrics.py` | `output/{EXP}/metrics_report.md` | Markdown |

---

## 5. Schema JSON của các file trung gian

### `concept_chunks.jsonl` (Step 01 output)

```json
{
  "chunk_id": "cs116_ch04_slide_p05_c01",
  "course_id": "CS116",
  "chapter_id": "ch04",
  "chapter_title": "Tiền xử lý dữ liệu",
  "topics": ["Missing Data", "Outlier Detection", ...],
  "source_type": "slide_pdf",
  "source_file": "CS116-Bai04-Data preprocessing.pdf",
  "page_number": 5,
  "section_title": "Xử lý dữ liệu bị thiếu",
  "text": "Có ba cách tiếp cận chính...",
  "embedding_ready": true
}
```

```json
{
  "chunk_id": "cs116_ch04_transcript_1_s003",
  "course_id": "CS116",
  "chapter_id": "ch04",
  "chapter_title": "Tiền xử lý dữ liệu",
  "topics": ["Missing Data", ...],
  "source_type": "video_transcript",
  "source_file": "4.1.json",
  "text": "Bây giờ chúng ta sẽ học về dữ liệu bị thiếu...",
  "timestamp_start": 90.5,
  "timestamp_end": 125.3,
  "youtube_url": "https://youtu.be/abc&t=90",
  "youtube_timestamp_start": "1:30",
  "youtube_timestamp_end": "2:05",
  "slide_file": "CS116-Bai04-Data preprocessing.pdf",
  "slide_start_page": 5,
  "word_count": 187,
  "embedding_ready": true
}
```

### `02_retrieval/{topic_id}.jsonl` (Step 02 output)

```json
{
  "topic_id": "ch04_t01",
  "topic_name": "Missing Data",
  "query": "missing data imputation KNNImputer SimpleImputer...",
  "num_context_blocks": 5,
  "context_blocks": [
    {
      "chunk_id": "cs116_ch04_slide_p05_c01",
      "chapter_id": "ch04",
      "chapter_title": "Tiền xử lý dữ liệu",
      "source_type": "slide_pdf",
      "source_file": "CS116-Bai04-Data preprocessing.pdf",
      "page_number": 5,
      "section_title": "Xử lý dữ liệu bị thiếu",
      "text": "Có ba cách tiếp cận chính...",
      "youtube_url": "",
      "slide_start_page": 5
    },
    {
      "chunk_id": "cs116_ch04_transcript_1_s003",
      "chapter_id": "ch04",
      "source_type": "video_transcript",
      "text": "Bây giờ chúng ta sẽ học về dữ liệu bị thiếu...",
      "timestamp_start": 90.5,
      "youtube_url": "https://youtu.be/abc&t=90",
      "youtube_ts_start": "1:30",
      "youtube_ts_end": "2:05"
    }
  ],
  "context_blocks_str": "..."
}
```

### `03_gen_stem/all_p1_results.jsonl` (Step 03 output)

```json
{
  "draft_question_id": "ch04_t01_q0",
  "question_text": "Câu 1. [Nhiều đáp án đúng] (G2) Phương pháp nào sau đây được sử dụng để xử lý dữ liệu bị thiếu trong sklearn?",
  "question_type": "multiple_correct",
  "correct_answers_content": [
    "SimpleImputer(strategy='mean')",
    "KNNImputer()",
    "IterativeImputer()"
  ],
  "correct_answer_count": 3,
  "topic": "Missing Data",
  "subtopic": "",
  "difficulty_label": "G2",
  "used_concept_chunk_ids": ["cs116_ch04_slide_p05_c01"],
  "sources": [],
  "style_alignment_note": "Câu hỏi kiểm tra kiến thức cụ thể về API sklearn...",
  "stem_has_answer": false,
  "answer_in_stem_warning": "none",
  "_meta": {
    "topic_id": "ch04_t01",
    "topic_name": "Missing Data",
    "difficulty": "G2",
    "seq": 0,
    "prompt_version": "v3"
  }
}
```

### `06_gen_cot/all_final_mcqs.jsonl` (Step 06 output)

```json
{
  "question_id": "cs116_ch04_t01_q_0001",
  "question_text": "Câu 1. [Nhiều đáp án đúng] (G2) Phương pháp nào sau đây được sử dụng để xử lý dữ liệu bị thiếu trong sklearn?",
  "question_type": "multiple_correct",
  "options": {
    "A": "SimpleImputer(strategy='mean')",
    "B": "KNNImputer()",
    "C": "PolynomialFeatures()",
    "D": "IterativeImputer()"
  },
  "correct_answers": ["A", "B", "D"],
  "correct_answer_count": 3,
  "topic": "Missing Data",
  "difficulty_label": "G2",
  "used_concept_chunk_ids": ["cs116_ch04_slide_p05_c01"],
  "used_assessment_item_ids": [],
  "style_alignment_note": "...",
  "_meta": {
    "topic_id": "ch04_t01",
    "p5_evaluations_count": 9,
    "p6_kept_distractors_count": 5,
    "p7_selected_count": 3,
    "p8_assembly_method": "llm"
  },
  "_cot_steps": {
    "p5": { "evaluations": [...] },
    "p6": { "kept_options": [...], "removed_options": [...] },
    "p7": { "selected_distractors": [...], "rejected_distractors": [...] }
  }
}
```

### `07_eval/evaluated_questions.jsonl` (Step 07 output)

```json
{
  "question_id": "cs116_ch04_t01_q_0001",
  "question_text": "...",
  "question_type": "multiple_correct",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct_answers": ["A", "B", "D"],
  "correct_answer_count": 3,
  "topic": "Missing Data",
  "difficulty_label": "G2",
  "evaluation": {
    "format_pass": true,
    "language_pass": true,
    "grammar_pass": true,
    "relevance_pass": true,
    "answerability_pass": true,
    "correct_set_pass": true,
    "no_four_correct_pass": true,
    "answer_not_in_stem_pass": true,
    "overall_valid": true,
    "fail_reasons": [],
    "quality_score": 0.92
  },
  "status": "accepted"
}
```

### `08_eval_iwf/final_accepted_questions.jsonl` (Step 08 output)

```json
{
  "question_id": "cs116_ch04_t01_q_0001",
  "question_text": "...",
  "question_type": "multiple_correct",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct_answers": ["A", "B", "D"],
  "correct_answer_count": 3,
  "topic": "Missing Data",
  "difficulty_label": "G2",
  "evaluation": {
    "overall_valid": true,
    "quality_score": 0.92
  },
  "distractor_evaluation": {
    "distractor_evaluations": [
      {
        "option_label": "C",
        "option_text": "PolynomialFeatures()",
        "plausible_distractor": true,
        "vague_terms": "none",
        "grammar_clue": false,
        "absolute_terms": "none",
        "distractor_length": "bình thường",
        "iwf_count": 0
      }
    ],
    "total_iwf_count": 0,
    "bad_options": [],
    "overall_distractor_quality_pass": true
  },
  "status": "accepted",
  "final_iwf_count": 0
}
```

### `09_explain/explanations.jsonl` (Step 09 output)

```json
{
  "question_id": "cs116_ch04_t01_q_0001",
  "question_text": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct_answers": ["A", "B", "D"],
  "explanation": {
    "question_motivation": "Câu hỏi kiểm tra kiến thức về các API xử lý missing data trong sklearn...",
    "correct_answer_rationale": "SimpleImputer, KNNImputer, IterativeImputer đều là các imputer trong sklearn...",
    "distractor_explanations": {
      "C": "PolynomialFeatures() dùng để tạo đặc trưng đa thức, không phải để xử lý missing data. Xem slide CS116-Bai04-Data preprocessing.pdf, trang 5."
    },
    "knowledge_context": {
      "topic_scope": "Tiền xử lý dữ liệu - Missing Data",
      "prerequisites": ["Pandas", "NumPy", "DataFrame"],
      "advanced_knowledge": "Multiple Imputation by Chained Equations (MICE)",
      "learning_value": "SV hiểu rõ các phương pháp xử lý missing data trong sklearn..."
    },
    "sources": [
      {
        "type": "slide",
        "file": "CS116-Bai04-Data preprocessing.pdf",
        "page": "5",
        "description": "📄 CS116-Bai04-Data preprocessing.pdf, Trang 5 — Xử lý dữ liệu bị thiếu"
      },
      {
        "type": "video",
        "url": "https://youtu.be/abc&t=90",
        "description": "▶️ Video bài giảng: Tiền xử lý dữ liệu [1:30 → 2:05]"
      }
    ]
  }
}
```

---

## 6. Mô hình AI sử dụng

### Generation Models (Steps 03–06, 09)

| Mô hình | Vai trò | Cấu hình vLLM |
|---|---|---|
| **Qwen2.5-14B-Instruct** | Sinh stems, distractors, explanations | `temperature=0.7`, `max_tokens=4096` |
| Fallback: GPT-4o | (web app, không dùng trong pipeline script) | API call |

**Lý do chọn Qwen2.5-14B:**
- Hiệu suất tốt trên benchmark tiếng Việt
- Chạy được trên 1 GPU (A100 40GB)
- Quản lý VRAM tốt với vLLM
- Context length đủ cho prompt dài

### Evaluation Model (Steps 07–08)

| Mô hình | Vai trò | Cấu hình vLLM |
|---|---|---|
| **Gemma-3-12b-it** | LLM-as-judge: overall eval + IWF analysis | `temperature=0.1`, `max_tokens=1024` |

**Lý do chọn Gemma-3-12b:**
- Điểm cao trên reasoning benchmarks
- VRAM thấp hơn Qwen-14B → chạy song song được
- Tốc độ nhanh cho batch evaluation

### Embedding Model

| Mô hình | Vai trò |
|---|---|
| **BAAI/bge-m3** | Vector embedding cho ChromaDB retrieval |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranking |

---

## 7. Định dạng câu hỏi MCQ

### 7.1 Single Correct (1 đáp án đúng)

```
Câu 1. [Một đáp án đúng] (G2) Phương pháp nào sau đây được sử dụng để xử lý
dữ liệu bị thiếu trong sklearn?

A. PolynomialFeatures()
B. SimpleImputer(strategy='mean')
C. StandardScaler()
D. LabelEncoder()
```

### 7.2 Multiple Correct (2–3 đáp án đúng)

```
Câu 2. [Nhiều đáp án đúng] (G3) Phương pháp nào sau đây có thể được sử dụng
để xử lý dữ liệu bị thiếu trong sklearn?

A. SimpleImputer(strategy='mean')
B. KNNImputer()
C. IterativeImputer()
D. OneHotEncoder()
```

### 7.3 HARD CONSTRAINTS

| Ràng buộc | Mô tả |
|---|---|
| **Đúng 4 options** | Luôn A/B/C/D, không thêm E |
| **1–3 đáp án đúng** | Không có 4 đáp án đúng (Step 07 check) |
| **Đáp án trong stem** | KHÔNG được ghi đáp án (hoặc một phần) vào stem (Step 07 check) |
| **Multi-correct label** | Stem phải ghi rõ `[Nhiều đáp án đúng]` |
| **Single-correct label** | Stem phải ghi rõ `[Một đáp án đúng]` |
| **Độ khó label** | Ghi sau nhãn loại: `(G1)`, `(G2)`, `(G3)` |

---

## 8. Cấu trúc thư mục project

```
CS431MCQGen/                          # ★ PROJECT ROOT
│
├── input/                            # Dữ liệu thô (commit lên git)
│   ├── topic_list.json               # ★ Danh sách topics cần sinh
│   ├── slide/                        # Slide PDF
│   ├── video/                        # Video MP4 + videos1.txt
│   ├── transcribe_data/              # Whisper JSON output
│   ├── video_transcript/             # Transcript text backup
│   ├── question_bank/                # Câu hỏi sẵn có
│   ├── sample_exams/                 # Đề thi PDF
│   ├── render_exam_template.html     # Template HTML render
│   └── vietnamese-stopwords.txt
│
├── data/                             # Dữ liệu đã xử lý
│   ├── processed/
│   │   ├── concept_chunks.jsonl              # Tất cả chunks (slide + transcript)
│   │   ├── transcript_chunks_with_timestamps.jsonl
│   │   └── assessment_items.jsonl            # (future) Câu hỏi sẵn có
│   ├── raw/
│   │   ├── slides/
│   │   ├── transcripts/
│   │   ├── exams/
│   │   └── generated_mcq_raw/
│   └── indexes/                      # ChromaDB vector indexes
│       ├── chroma.sqlite3
│       ├── header.bin / link_lists.bin / ...
│       └── data_level0.bin
│
├── configs/                          # Cấu hình
│   ├── mcqgen_config.yaml            # Schema + chapters + topics
│   ├── generation_config.yaml        # Generation params
│   ├── generation_config_active.yaml  # Override từ web app
│   └── bloom_taxonomy.yaml           # Bloom levels
│
├── src/                              # Source code
│   ├── common.py                      # ★ Config + P1-P8 prompts + utils
│   ├── eval/
│   │   ├── eval_overall.py           # Step 07: 8 criteria eval
│   │   ├── eval_iwf.py               # Step 08: IWF distractor analysis
│   │   ├── eval_metrics.py           # Step 10: 5 metrics computation
│   │   └── __init__.py
│   └── gen/
│       ├── indexing.py               # Step 01: slide PDF → ChromaDB
│       ├── chunk_transcript_with_timestamps.py  # Step 01b
│       ├── retrieval.py              # Step 02: 5-tier hybrid RAG
│       ├── retrieval_hybrid.py       # HybridRetriever class
│       ├── p1_gen_stem.py           # Step 03: P1 stem generation
│       ├── p2_p3_refine.py          # Step 04: P2+P3 self-refine
│       ├── p4_candidates.py         # Step 05: P4 distractor candidates
│       ├── p5_p8_cot.py             # Step 06: P5-P8 CoT selection
│       ├── explain_mcq.py           # Step 09: explanation + citations
│       ├── prompt_config.py         # Generation config loader
│       ├── prompt_parser.py         # Free-text → structured config
│       └── test_gen_mini.py         # Mini test (1 topic, 1 question)
│
├── output/                           # Kết quả mỗi experiment
│   └── {EXP_NAME}/                  # ★ Experiment-specific
│       ├── 01_indexing/
│       ├── 02_retrieval/
│       │   └── {topic_id}.jsonl     # Per-topic retrieval results
│       ├── 03_gen_stem/
│       │   └── all_p1_results.jsonl
│       ├── 04_gen_refine/
│       ├── 05_gen_distractors/
│       ├── 06_gen_cot/
│       │   └── all_final_mcqs.jsonl  # ★ Final MCQs (trước eval)
│       ├── 07_eval/
│       │   ├── evaluated_questions.jsonl  # Accepted
│       │   └── failed_questions.jsonl     # Rejected
│       ├── 08_eval_iwf/
│       │   ├── final_accepted_questions.jsonl
│       │   └── final_rejected_questions.jsonl
│       ├── 09_explain/
│       │   └── explanations.jsonl
│       ├── final/
│       │   └── exam_final.html       # Rendered exam
│       ├── metrics_report.json
│       └── metrics_report.md
│
├── scripts/                          # Shell scripts (SLURM / local)
│   ├── 00_pipeline.sh               # Full pipeline orchestrator
│   ├── 01_index.sh
│   ├── 02_retrieval.sh
│   ├── 03_gen_stem.sh
│   ├── 04_gen_refine.sh
│   ├── 05_gen_distractors.sh
│   ├── 06_gen_cot.sh
│   ├── 07_eval.sh
│   ├── 08_eval_iwf.sh
│   ├── 09_explain.sh
│   ├── 10_eval_metrics.sh
│   ├── transcribe_videos.py
│   ├── run_transcribe.sh
│   ├── download_models.sh
│   └── check_cuda.sh
│
├── log/                              # Logs
│
├── models/                           # Local model cache
│   ├── Qwen2.5-14B-Instruct/
│   └── gemma-3-12b-it/
│
├── web/                              # Streamlit web app
│   └── app.py
│
├── deploy_web/                       # FastAPI backend + Docker
│   ├── backend/
│   └── nginx/
│
├── EXPLAINABLE_QUESTION_GENERATION_PIPELINE.md
├── PIPELINE_DOCUMENTATION.md         # ★ File này
└── README.md
```

---

## 9. Cách chạy pipeline

### 9.1 Full Pipeline (tất cả 10 bước)

```bash
# Đặt tên experiment (THAY ĐỔI MỖI LẦN CHẠY)
export EXP_NAME="exp_04_ch07b_ch08_50q"

# Chạy full pipeline (tất cả steps)
cd /datastore/uittogether/LuuTru/Thanhld/CS431MCQGen
bash scripts/00_pipeline.sh
```

### 9.2 Chạy từng bước riêng lẻ

```bash
# Step 01 — Indexing (slide + transcript → ChromaDB)
export EXP_NAME="exp_04"
bash scripts/01_index.sh

# Step 01b — Chunk transcripts with timestamps
python -u src/gen/chunk_transcript_with_timestamps.py

# Step 02 — Hybrid retrieval
export EXP_NAME="exp_04"
bash scripts/02_retrieval.sh

# Step 03 — P1 stem generation
export EXP_NAME="exp_04"
bash scripts/03_gen_stem.sh

# Step 04 — P2+P3 self-refine
export EXP_NAME="exp_04"
bash scripts/04_gen_refine.sh

# Step 05 — P4 distractor candidates
export EXP_NAME="exp_04"
bash scripts/05_gen_distractors.sh

# Step 06 — P5-P8 CoT distractor selection
export EXP_NAME="exp_04"
bash scripts/06_gen_cot.sh

# Step 07 — Overall evaluation
export EXP_NAME="exp_04"
bash scripts/07_eval.sh

# Step 08 — IWF analysis
export EXP_NAME="exp_04"
bash scripts/08_eval_iwf.sh

# Step 09 — Explanation generation
export EXP_NAME="exp_04"
bash scripts/09_explain.sh

# Step 10 — Metrics
export EXP_NAME="exp_04"
bash scripts/10_eval_metrics.sh
```

### 9.3 Cấu hình experiment (generation_config_active.yaml)

```yaml
# Để generate tập trung vào một số chapters/topics
generation:
  target_range: [40, 55]           # Số câu hỏi mục tiêu
  single_correct_ratio: 0.80        # 80% single, 20% multiple
  focus_chapters: ["ch07b", "ch08"] # Chỉ sinh cho 2 chapters này
  topic_weights:                    # Trọng số per chapter
    "ch07b": 2.0
    "ch08": 2.0
  difficulty_distribution:
    G1: 0.20
    G2: 0.50
    G3: 0.30

diversity:
  prefer_openings:
    - "Điều gì khiến..."
    - "Đâu là điểm khác biệt giữa..."
    - "Trường hợp nào sau đây minh họa đúng nhất về..."
```

### 9.4 Mini test (1 topic, 1 câu)

```bash
cd /datastore/uittogether/LuuTru/Thanhld/CS431MCQGen
export EXP_NAME="mini_test"
python -u src/gen/test_gen_mini.py
```

### 9.5 Web deployment (Streamlit)

```bash
# Terminal 1: FastAPI backend
cd deploy_web/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Terminal 2: Streamlit frontend
cd web
streamlit run app.py --server.port 8501

# Browser: http://localhost:8501
```

---

## 10. Evaluation Metrics

### 10.1 Metrics computation

```bash
export EXP_NAME="exp_04"
cd /datastore/uittogether/LuuTru/Thanhld/CS431MCQGen
python -u src/eval/eval_metrics.py \
    --exp "$EXP_NAME" \
    --human-json input/human_review.json \
    --output output/"$EXP_NAME"/metrics_report.json
```

### 10.2 Metrics summary table

| Metric | Target | Công thức | Đánh giá |
|---|---|---|---|
| **Topic Coverage** | ≥ 80% | covered / total topics | Độ rộng chương trình học |
| **LLM Judge Pass Rate** | ≥ 70% | accepted / total evaluated | Chất lượng generation |
| **IWF Pass Rate** | ≥ 70% | iwf_passed / iwf_total | Chất lượng distractors |
| **Quality Score** | ≥ 0.80 | avg(quality_score) | Điểm chất lượng tổng hợp |
| **Bloom KL Divergence** | ≤ 0.30 | KL(actual ∥ target) | Phân bố độ khó |
| **Human Judgment κ** | ≥ 0.60 | Cohen's κ human vs LLM | Độ tin cậy LLM judge |

### 10.3 Metrics đã loại bỏ

| Metric | Lý do |
|---|---|
| **BLEU + ROUGE-L** | Cần reference set — không phù hợp open-ended MCQ generation |
| **BERTScore** | Cần reference set + tốn thêm LLM call, không cải thiện quyết định |
| **Answer Ratio** | Inline trong generation config, không cần metric riêng |
| **Diversity Openings** | Chủ quan, không đo lường chất lượng thực |
| **Old Cohen's κ (CSV-based)** | Schema lộn xộn, logic rải rác nhiều hàm → thay bằng module Human Judgment JSON |

### 10.4 Interpreting results

```
Pass Rate ≥ 70%:
  ✅ Pipeline chất lượng tốt — dùng được cho production

Pass Rate 50–70%:
  ⚠️  Trung bình — vấn đề thường gặp:
      • Distractors yếu (quá rõ ràng là sai)
      • Stem chứa đáp án (vi phạm hard constraint)
      • Poor alignment với course material

Pass Rate < 50%:
  ❌ Pipeline cần tuning — nguyên nhân cần điều tra:
      • Retrieval context không đủ/relevant
      • P1 prompt cần cải thiện
      • Temperature quá cao → format lỗi

Bloom KL > 0.5:
  ❌ Phân bố độ khó lệch nhiều
      • Quá nhiều câu G1 (nhớ) → cần tăng difficulty target
      • Quá nhiều câu G3 (phân tích) → cần giảm
```

---

## 11. Web Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Browser (Streamlit — port 8501)                        │
│  ┌───────────────────────────────────────────────────┐  │
│  │ User enters:                                       │  │
│  │ "Tôi muốn 40 câu hỏi về classification và CNN"     │  │
│  │                                                    │  │
│  │ → prompt_parser.py (LLM parse hoặc keyword match) │  │
│  │ → generation_config_active.yaml                    │  │
│  └───────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                      │ HTTP POST /api/generate
                      ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI Backend (port 8000)                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │ /api/generate                                     │   │
│  │   1. Write generation_config_active.yaml          │   │
│  │   2. Run 00_pipeline.sh (background job)          │   │
│  │   3. SSE stream progress back to frontend        │   │
│  │                                                    │   │
│  │ /api/results/{exp_name}                          │   │
│  │   → Return final_accepted_questions.jsonl        │   │
│  │   → Return metrics_report.json                    │   │
│  └─────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────┘
                      │ Background process (SLURM/local)
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Pipeline execution (Steps 01–10)                      │
│  Model: Qwen2.5-14B-Instruct + Gemma-3-12b-it         │
│  Output: output/{EXP_NAME}/                             │
└─────────────────────────────────────────────────────────┘
```

---

## Phụ lục A — Hard Constraints tổng hợp

| # | Constraint | Check step | Action on violation |
|---|---|---|---|
| 1 | Stem không chứa đáp án | P1, P2, P3 | Hard reject (Step 07) |
| 2 | Đúng 4 options A/B/C/D | P8 | Hard reject (Step 07) |
| 3 | 1–3 đáp án đúng | P1 | Force override |
| 4 | Nhãn loại đúng | P1, P3, P8 | Force override |
| 5 | Không 4 đáp án đúng | Step 07 | Hard reject |
| 6 | Tiếng Việt | P1, Step 07 | Hard reject nếu không |
| 7 | Không grammar clue trong distractors | Step 08 | Reject if IWF > 3 |
| 8 | Không absolute terms gây clue | Step 08 | Reject if IWF > 3 |
| 9 | Cách mở đầu đa dạng | P1 diversity rule | Warning (metric only) |

## Phụ lục B — Bloom's Taxonomy Mapping

```
Bloom's Taxonomy (6 levels):

L1: Remember — nhớ, liệt kê, định nghĩa, thuộc tính
L2: Understand — giải thích, so sánh, mô tả, tại sao
L3: Apply — áp dụng, sử dụng, tính toán, kết quả
L4: Analyze — phân tích, đánh giá, hiệu suất, cách cải thiện
L5: Evaluate — đánh giá, lựa chọn, ưu nhược điểm
L6: Create — thiết kế, xây dựng, đề xuất, lập trình

Pipeline mapping:
  G1 (dễ) → L1 + L2
  G2 (vừa) → L3 + L4
  G3 (khó) → L5 + L6

Target distribution:
  G1: 40% (L1: 20%, L2: 20%)
  G2: 40% (L3: 20%, L4: 20%)
  G3: 20% (L5: 10%, L6: 10%)
```

## Phụ lục C — Ràng buộc về sources

```
Slide citation format:
  <(Trích dẫn: CS116-Bai04-Data preprocessing.pdf, Trang 5)>

Video citation format:
  <(Trích dẫn: https://youtu.be/abc&t=90 | 1:30 - 2:05)>
  (Khi bấm link → YouTube nhảy đến đúng giây 90)

Priority:
  1. Slides cùng chapter với câu hỏi
  2. Slides cross-chapter (nếu cần bổ sung)
  3. Video YouTube được map từ slide đầu tiên trong context

Không dùng:
  ✗ External web search
  ✗ Wikipedia
  ✗ Stack Overflow
  ✗ GitHub repositories
```

---

*Document generated: 2026-04-15*
*Pipeline version: CS431MCQGen v1.2*
