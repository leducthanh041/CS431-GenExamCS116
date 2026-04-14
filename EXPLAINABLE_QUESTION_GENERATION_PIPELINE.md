# Explainable Question Generation Pipeline — Thiết kế chi tiết

> **Ngày:** 2026-04-09
> **Dự án:** CS431MCQGen — Pipeline sinh câu hỏi trắc nghiệm có giải thích & truy vết nguồn gốc
> **Trạng thái:** Thiết kế — chưa implement

---

## 1. Mục tiêu tổng quát

Mở rộng pipeline MCQGen hiện tại (8 bước: Indexing → Retrieval → P1-P8 Gen → Eval) thành **Explainable Question Generation Pipeline** — mỗi câu hỏi sinh ra phải:

1. **Truy vết được nguồn gốc** từ tài liệu chính thức (slide PDF, transcript video)
2. **Gắn citation** cho: transcript bài giảng, timestamp YouTube (nếu suy ra được), nội dung slide
3. **Dùng sample_exams để điều khiển style/cấu trúc** câu hỏi — không phải nguồn fact
4. **Tự động sinh explanation** giải thích tại sao đáp án đúng, tại sao các đáp án sai

---

## 2. tieplm giải quyết gì — Bài học áp dụng

### 2.1 Contextual Retrieval (Anthropic/tieplm)

tieplm áp dụng **Contextual Chunking** theo approach của Anthropic:
- Với mỗi chunk 60 giây ASR, một LLM (gpt-5-mini) sinh **contextual prefix** mô tả chunk đó trong ngữ cảnh toàn bộ video
- Prefix được prepended vào chunk text **trước khi embed** bằng OpenAI text-embedding-3-small
- Kết quả: retrieval accuracy tăng đáng kể vì embedding model "hiểu" chunk đó nói về cái gì trong toàn bộ bài giảng

### 2.2 Bidirectional Citation Flow

- Model generate `[1]`, `[2]` markers trong generated text
- Frontend render markers thành clickable YouTube links với timestamp
- Citation accuracy = ground truth chunk in top-10 reranked?

### 2.3 5-Tier Retrieval Pipeline

```
Vector search (top-150) → BM25 (top-150) → RRF Fusion (k=60)
→ PostgreSQL metadata join → Cross-encoder rerank (top-10)
```

### 2.4 Bài học cho CS431MCQGen

| tieplm | CS431MCQGen cần làm |
|---|---|
| Contextual prefix cho từng chunk | Thêm contextual prefix khi indexing transcript |
| `[1]` markers trong generated text | Thêm citation markers vào P1-P8 prompts |
| Frontend render citation → YouTube link | Backend trả về structured citation JSON |
| PostgreSQL join video metadata | Thêm video metadata (video_id, youtube_url) vào transcript chunks |
| Cross-encoder rerank (top-10) | Thêm rerank bước cuối trong retrieval |
| Ground truth chunk → Citation accuracy | Đánh giá `used_concept_chunk_ids` không rỗng |

---

## 3. Pipeline hiện tại thiếu gì

### 3.1 Retrieval — thiếu `source_file` và `page_number`

Trong `format_context_block()` (common.py:786-797):

```python
# HIỆN TẠI — thiếu source_file và page_number
parts = []
if chunk.get("chapter_id"):
    parts.append(f"[{chunk['chapter_id']}]")
if chunk.get("section_title"):
    parts.append(f"Tiêu đề: {chunk['section_title']}")
if chunk.get("topic"):
    parts.append(f"Chủ đề: {chunk['topic']}")
if chunk.get("text"):
    parts.append(chunk["text"])
```

→ Chunk context truyền vào prompt **không có thông tin nguồn**, model không thể gắn citation chính xác.

### 3.2 Retrieval — `used_concept_chunk_ids` luôn rỗng

Trong `retrieval.py`, context blocks được lưu nhưng **không có cơ chế để model trả về** `used_concept_chunk_ids`. Model sinh ra câu hỏi nhưng không gắn nguồn.

### 3.3 Transcript — không parse ASR timestamp

File transcript 4.1.txt có format `<00:00:14.360><c>` — **timestamp tag** có thể dùng để suy ra YouTube timestamp. Nhưng hiện tại:
- `extract_transcript()` trong indexing.py chỉ split text thành chunks, **không parse timestamp**
- Không có `video_id`, `youtube_url` trong metadata

### 3.4 Sample exams — chưa được dùng

3 file PDF trong `input/sample_exams/` (de1.pdf, de2.pdf, HK1 24-25.pdf) **chưa được index**. Chúng chỉ nên dùng để:
- Học style/cấu trúc câu hỏi (format, độ khó, cách diễn đạt)
- **Không phải nguồn fact** — không thể generate fact từ đây

### 3.5 Generation — không có explanation

Pipeline hiện tại sinh stem + key + distractors nhưng **không sinh explanation**. Mỗi câu hỏi cần:
- Giải thích tại sao đáp án đúng
- Giải thích tại sao mỗi distractor sai

---

## 4. Thiết kế Pipeline Mới

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXPLAINABLE QUESTION PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 0: ENHANCED INDEXING (mở rộng từ Step 01)                            │
│  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────────┐  │
│  │  Slide PDF   │   │ Video Transcript │   │    Sample Exams (PDF)      │  │
│  │  (PyMuPDF)   │   │   (ASR + TS)     │   │    (style reference)      │  │
│  └──────┬───────┘   └────────┬─────────┘   └──────────┬────────────────┘  │
│         │                    │                           │                   │
│         │  page_number      │  timestamp tags            │                   │
│         │  source_file      │  video_id, youtube_url     │  format/style    │
│         └────────┬───────────┘                           │                   │
│                  │                                        │                   │
│  ┌───────────────┴────────────────────────────────────────┘                │
│  │  NEW: Contextual Prefix Generation (LLM gpt-5-mini hoặc local)         │
│  │  → Mỗi chunk có prefix mô tả ngữ cảnh trong bài giảng                  │
│  └──────────────────────────────────────┬──────────────────────────────────┘
│                                         │                                    
│  ┌──────────────────────────────────────┴──────────────────────────────────┐
│  │  BGE-m3 Embedding + ChromaDB Storage                                     │
│  │  Metadata: chunk_id, chapter_id, source_type, source_file,              │
│  │            page_number, timestamp_start, video_id, youtube_url,         │
│  │            contextual_prefix                                            │
│  └──────────────────────────────────────┬──────────────────────────────────┘
│                                         │                                    
│  STEP 2: ENHANCED RETRIEVAL (mở rộng từ Step 02)                          │
│  ┌──────────────────────────────────────┴──────────────────────────────────┐
│  │  Query → Vector (top-150) → BM25 (top-150) → RRF Fusion (k=60)          │
│  │  → Metadata join (video_id, youtube_url) → Cross-encoder (top-10)       │
│  │                                                                            │
│  │  Context blocks được format với ĐẦY ĐỦ metadata:                        │
│  │  [ch04] Slide: CS116-Bai04-Data preprocessing.pdf, Trang 3              │
│  │  Video: Bài 4.1 — Tiền xử lý dữ liệu | YouTube: https://youtu.be/...   │
│  │  Thời điểm: 00:00:14 - 00:01:00 | [1] Transcript text here...          │
│  └──────────────────────────────────────┬──────────────────────────────────┘
│                                         │                                    
│  STEPS 3-6: GENERATION (P1-P8 + NEW CITATION + EXPLANATION)                │
│  ┌──────────────────────────────────────┴──────────────────────────────────┐
│  │  P1 (gen_stem_key):   Sinh stem + key + markers [1], [2]              │
│  │  P2/P3 (self_refine): Refine đồng thời refine citation markers         │
│  │  P4 (distractors):     Sinh distractor candidates                      │
│  │  P5-P8 (CoT):          Chọn distractors + giải thích tại sao sai      │
│  │                                                                            │
│  │  NEW: P9 (gen_explanation) — sinh giải thích cho mỗi option             │
│  │  [1] Giải thích tại sao đáp án đúng (trích nguyên văn text từ nguồn)  │
│  │  [2-4] Giải thích tại sao mỗi distractor sai nhưng hấp dẫn             │
│  └──────────────────────────────────────┬──────────────────────────────────┘
│                                         │                                    
│  STEP 7-8: EVALUATION (mở rộng)                                         │
│  ┌──────────────────────────────────────┴──────────────────────────────────┐
│  │  eval_overall: 6 checklist + Citation Accuracy + Explanation Quality  │
│  │  eval_iwf: 6 IWF types + Distractor Plausibility                       │
│  └──────────────────────────────────────────────────────────────────────────┘
│                                                                             │
│  OUTPUT: exam.jsonl                                                         │
│  { question_text, options, correct_answers, difficulty,                     │
│    citations: [{chunk_id, source_type, source_file, page_number,            │
│                 timestamp_start, youtube_url, quoted_text}],                 │
│    explanations: [{option, is_correct, explanation, cited_chunk_id}] }      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Vai trò từng nguồn dữ liệu

### 5.1 Slide PDF — Nguồn fact chính ✅

- **Role:** Nguồn kiến thức chính thức, có page number chính xác
- **Citation format:** `[ch04] CS116-Bai04-Data preprocessing.pdf, Trang 5`
- **Embedding:** Mỗi page = 1 chunk, metadata lưu page_number
- **Không dùng cho style** — slide format khác exam format

### 5.2 Video Transcript — Nguồn bổ sung + timestamp ✅

- **Role:** Giải thích bằng lời, có timestamp cho video reference
- **Citation format:** `Video "Tiền xử lý dữ liệu" — YouTube [00:14-01:00]`
- **ASR timestamp extraction:** Parse `<00:00:14.360><c>` → `timestamp_start = 14.360`
- **Video metadata:** Cần thêm `video_id`, `youtube_url` (lấy từ `input/video_transcript/` hoặc external mapping)
- **⚠️ Cảnh báo:** Transcript có Whisper artifact — text bị duplicate 2-3 lần. Cần deduplicate trước khi chunk.

### 5.3 Assessment Items (quiz bank) — Không dùng cho generation

- **Role:** Chỉ dùng để validate quality (so sánh generated vs existing)
- **Không dùng làm retrieval source** — tránh bias và plagiarism
- **Lưu trong ChromaDB riêng (`assessment_kb`)** — không trộn với concept chunks

### 5.4 Sample Exams (de1.pdf, de2.pdf, HK1 24-25.pdf) — Style reference ONLY

- **Role:** Học format, structure, độ khó của câu hỏi
  - Format: Số thứ tự, cách viết đáp án (A., B., C., D.)
  - Cấu trúc: Câu hỏi → Options → Đáp án đúng
  - Độ khó: Tỉ lệ G1/G2/G3
  - Cách diễn đạt: Từ vựng, cú pháp tiếng Việt học thuật
- **Không phải nguồn fact** — extract bằng PyMuPDF, **không embed vào ChromaDB**
- **Cách dùng:** Đọc khi thiết kế prompt P1 (style instruction)

---

## 6. Citation Mechanism — Gắn nguồn cho từng câu hỏi

### 6.1 Citation markers trong generation

**P1 prompt bổ sung:**
```
Với mỗi câu hỏi, gắn citation marker [N] cho mỗi fact quan trọng.
Format: [1], [2], [3]... tương ứng với các context blocks đã cung cấp.
Nếu fact đến từ slide: [N] (Slide: <source_file>, Trang <page_number>)
Nếu fact đến từ video: [N] (Video: <chapter_title>, YouTube <youtube_url?t=<timestamp>)
```

**Model output example:**
```
Câu hỏi: Phương pháp nào được khuyến nghị khi tỉ lệ dữ liệu thiếu trên 50% ở một cột?
[1] Theo slide CS116-Bai04-Data preprocessing.pdf, Trang 12:
    "Nếu tỉ lệ dữ liệu bị thiếu trên 50%, nên cân nhắc loại bỏ cột đó"
[2] Giảng viên giải thích thêm: "drop column với threshold=0.5"
```

### 6.2 Citation trong explanation

**P9 prompt (gen_explanation):**
```
Với mỗi option (đúng và sai), sinh giải thích ngắn gọn (1-2 câu).
Đáp án đúng: Giải thích tại sao đúng, TRÍCH NGUYÊN VĂN từ nguồn nếu có.
Distractor: Giải thích tại sao sai nhưng vẫn hấp dẫn (plausible).
Gắn citation tương tự format trên.
```

### 6.3 Citation structure trong JSON output

```json
{
  "question_id": "ch04_t01_q1",
  "question_text": "...",
  "options": [
    {"id": "A", "text": "...", "is_correct": true},
    {"id": "B", "text": "...", "is_correct": false},
    {"id": "C", "text": "...", "is_correct": false},
    {"id": "D", "text": "...", "is_correct": false}
  ],
  "citations": [
    {
      "marker": "[1]",
      "chunk_id": "cs116_ch04_slide_p12",
      "source_type": "slide_pdf",
      "source_file": "CS116-Bai04-Data preprocessing.pdf",
      "page_number": 12,
      "youtube_url": null,
      "timestamp_start": null,
      "quoted_text": "Nếu tỉ lệ dữ liệu bị thiếu trên 50%, nên cân nhắc loại bỏ cột đó"
    },
    {
      "marker": "[2]",
      "chunk_id": "cs116_ch04_transcript_s003",
      "source_type": "video_transcript",
      "source_file": "4.1.txt",
      "page_number": null,
      "youtube_url": "https://youtu.be/VIDEO_ID",
      "timestamp_start": 14.360,
      "quoted_text": "drop column với threshold=0.5"
    }
  ],
  "explanations": [
    {
      "option_id": "A",
      "is_correct": true,
      "explanation": "Đúng. Theo [1], khi tỉ lệ missing > 50%, drop column là phương pháp được khuyến nghị vì imputation không đáng tin cậy.",
      "cited_chunk_ids": ["cs116_ch04_slide_p12"]
    },
    {
      "option_id": "B",
      "is_correct": false,
      "explanation": "Sai. [2] giải thích rằng mean imputation chỉ phù hợp khi missing < 10%, không phải > 50%.",
      "cited_chunk_ids": ["cs116_ch04_transcript_s003"]
    }
  ],
  "used_concept_chunk_ids": [
    "cs116_ch04_slide_p12",
    "cs116_ch04_transcript_s003"
  ]
}
```

---

## 7. Transcript → YouTube Timestamp Mapping

### 7.1 Phát hiện từ 4.1.txt

Transcript format:
```
Kind: captions Language: vi [âm nhạc]
Hôm<00:00:14.360><c> nay</c><00:00:14.519><c> chúng</c><00:00:14.679><c> ta</c>...
```

**Quy tắc:**
- `<MM:SS.mmm><c>` = word-level timestamp, `<c>` = confidence tag
- Text sau mỗi timestamp = từ được nhận diện tại thời điểm đó
- **YouTube timestamp** = trực tiếp (Whisper dùng video audio → timestamp = video time)

### 7.2 Extraction algorithm

```python
import re

def parse_transcript_with_timestamps(txt_path: str) -> list[dict]:
    """Parse transcript, extract word-level timestamps."""
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Pattern: timestamp tag followed by word
    # <00:00:14.360><c>word<00:00:14.519><c>next_word
    pattern = r'<(\d{2}:\d{2}:\d{2}\.\d{3})><c>([^<]+)</c>'

    words_with_ts = []
    for match in re.finditer(pattern, content):
        ts_str = match.group(1)  # "00:00:14.360"
        word = match.group(2).strip()

        # Parse to seconds
        parts = ts_str.split(':')
        seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        words_with_ts.append({'timestamp': seconds, 'word': word})

    # Deduplicate repeated words (Whisper artifact)
    # If same word appears consecutively with similar timestamps, keep one
    cleaned = []
    prev_word, prev_ts = None, None
    for item in words_with_ts:
        if item['word'] != prev_word or item['timestamp'] - prev_ts > 0.5:
            cleaned.append(item)
            prev_word, prev_ts = item['word'], item['timestamp']

    return cleaned


def chunk_with_timestamps(words: list[dict], chunk_size: int = 200) -> list[dict]:
    """Group words into chunks of ~200 words, preserve timestamp range."""
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i+chunk_size]
        chunk_text = " ".join(w['word'] for w in chunk_words)
        ts_start = chunk_words[0]['timestamp']
        ts_end = chunk_words[-1]['timestamp']

        chunks.append({
            'text': chunk_text,
            'timestamp_start': ts_start,
            'timestamp_end': ts_end,
            'youtube_url': f"https://youtu.be/{VIDEO_ID}?t={int(ts_start)}"
        })
    return chunks
```

### 7.3 Video metadata mapping

Cần tạo mapping file `input/video_transcript/video_metadata.json`:

```json
{
  "4.1.txt": {
    "video_id": "UCxxxxxxx",
    "youtube_url": "https://www.youtube.com/watch?v=UCxxxxxxx",
    "chapter_title": "Tiền xử lý dữ liệu",
    "video_start_time": 0
  },
  "4.2.txt": { ... }
}
```

---

## 8. Sample Exams cho Style — Không làm méo Factual Grounding

### 8.1 Nguyên tắc thiết kế

```
MẪU SAMPLE EXAM (style reference)
↓
Extract format/structure → STYLE PROMPT TEMPLATE
↓
Inject vào P1 prompt: "Dựa theo format đề thi mẫu, viết câu hỏi..."
↓
Model sinh câu hỏi theo style, SỬ DỤNG FACTS TỪ RETRIEVAL CONTEXT
↓
KHÔNG BAO GIỜ lấy fact từ sample exam
```

### 8.2 Style template extraction

```python
def extract_style_template(pdf_path: str) -> dict:
    """Extract style/format patterns từ sample exam PDF."""
    import fitz
    doc = fitz.open(pdf_path)
    style = {
        'numbering': 'Câu X.',      # vs "X." hoặc "Câu hỏi X:"
        'option_format': 'A. ...',  # vs "A)" hoặc "a)"
        'has_bracket_correct': True, # "[Đúng]" / "[Sai]" indicators
        'difficulty_markers': ['Dễ', 'Trung bình', 'Khó'] if G1/G2/G3
    }
    return style
```

### 8.3 P1 prompt bổ sung

```python
CITATION_STYLE = """
Style tham khảo từ đề thi mẫu:
- Đánh số: "Câu {i}."
- Options: "A. ...", "B. ...", "C. ...", "D. ..."
- Đáp án đúng: đánh dấu ✓ ở cuối option đúng
- Độ khó: {difficulty} ({difficulty_label})
- Ngôn ngữ: Tiếng Việt học thuật, ngắn gọn, rõ ràng

Tuy nhiên, NỘI DUNG câu hỏi phải dựa HOÀN TOÀN vào các context blocks [1], [2]...
được cung cấp bên dưới. KHÔNG sử dụng kiến thức bên ngoài context blocks.
"""
```

---

## 9. Bộ tiêu chí đánh giá mới

### 9.1 Tiêu chí hiện có (giữ nguyên)

| Tiêu chí | Mô tả |
|---|---|
| Factual Accuracy | Câu hỏi đúng về mặt kiến thức |
| Question Clarity | Câu hỏi rõ ràng, không mơ hồ |
| Distractor Quality | Distractors hấp dẫn nhưng sai |
| Answer Uniqueness | Chỉ 1 đáp án đúng |
| Topic Alignment | Câu hỏi đúng topic |
| Difficulty Match | Độ khó phù hợp |

### 9.2 Tiêu chí mới cho Explainable Questions

| Tiêu chí | Mô tả | Đo lường |
|---|---|---|
| **Citation Presence** | Câu hỏi có gắn citation markers [N] không? | % câu hỏi có ≥1 citation |
| **Citation Accuracy** | Cited chunk có chứa fact trong câu hỏi không? | LLM judge: cited text ↔ fact overlap |
| **Citation Traceability** | `used_concept_chunk_ids` có khớp với citations không? | Set equality |
| **Explanation Presence** | Mỗi option có explanation không? | % options có explanation |
| **Explanation Correctness** | Explanation đúng về mặt khoa học? | LLM judge |
| **Quoted Text Accuracy** | `quoted_text` trong citation có trùng với source chunk? | String similarity |
| **Source Diversity** | Câu hỏi dùng citation từ cả slide và transcript? | % multi-source |
| **YouTube Link Validity** | `youtube_url` có đúng format và timestamp? | Regex + URL format |

### 9.3 Eval prompt mới

**Citation Accuracy Evaluation:**
```
Task: Evaluate whether the cited chunk [N] actually supports the fact in the question.
Question: {question_text}
Cited chunk text: {cited_chunk_text}
Expected fact: {fact_derived_from_question}

Score: 1 (fully supported) | 0.5 (partially supported) | 0 (not supported / hallucinated)
Reason: {brief justification}
```

---

## 10. Rủi ro và Trade-offs

### 10.1 Rủi ro cao

| Rủi ro | Mức | Giải pháp |
|---|---|---|
| Model không gắn citation markers | Cao | Thêm instruction cứng trong prompt, validate output schema |
| Citation chứa hallucination | Cao | Cross-validate quoted_text với source chunk |
| Transcript artifact (duplicate text) | Cao | Pre-processing deduplication + ASR confidence filter |
| Sample exam bias | Trung | Không embed sample exam, chỉ dùng làm style prompt |

### 10.2 Trade-offs thiết kế

| Quyết định | Ưu điểm | Nhược điểm |
|---|---|---|
| Thêm P9 (gen_explanation) | Tăng explainability | Tăng thời gian generation, token cost |
| Dùng GPT-5-mini cho contextual prefix | Tăng retrieval accuracy | Chi phí API, phụ thuộc OpenAI |
| 5-tier retrieval (RRF + rerank) | Tăng recall, precision | Phức tạp, cần maintain BM25 index |
| Thêm YouTube timestamp | Tăng traceability | Cần video metadata mapping, có thể không có YouTube URL |

### 10.3 Giải pháp thay thế

1. **Thay vì GPT-5-mini contextual prefix:** Dùng local LLM (Qwen2.5-14B) cho contextual prefix generation — giảm chi phí, tăng privacy
2. **Thay vì 5-tier retrieval:** Chỉ dùng vector + rerank (đơn giản hơn, đủ tốt cho dataset nhỏ)
3. **Thay vì YouTube timestamp:** Chỉ dùng transcript timestamp — đơn giản hơn nếu không có YouTube URLs

---

## 11. Lộ trình triển khai theo giai đoạn

### Phase 1: Foundation (Tuần 1-2) — Không thay đổi generation

```
1.1 Sửa format_context_block() — thêm source_file, page_number
    File: src/common.py (dòng ~786)

1.2 Thêm video metadata mapping
    File: input/video_transcript/video_metadata.json (mới)

1.3 Cập nhật extract_transcript() — parse ASR timestamps
    File: src/gen/indexing.py (dòng ~113-156)

1.4 Cập nhật ChromaDB metadata schema — thêm timestamp_start, youtube_url
    File: src/gen/indexing.py (dòng ~190-200)

1.5 Cập nhật retrieval.py — đưa đầy đủ metadata vào context blocks
    File: src/gen/retrieval.py (dòng ~196-207)
```

### Phase 2: Enhanced Retrieval (Tuần 2-3)

```
2.1 Thêm BM25 retrieval (dùng rank_bm25 library)
    File: src/gen/retrieval.py (mới)

2.2 Thêm RRF Fusion
    File: src/gen/retrieval.py

2.3 Thêm Cross-encoder rerank (dùng cross-encoder/ms-marco model)
    File: src/gen/retrieval.py

2.4 Tạo pipeline retrieval 5-tier hoàn chỉnh
```

### Phase 3: Generation với Citations (Tuần 3-4)

```
3.1 Cập nhật P1 prompt — thêm citation instruction + style template
    File: src/common.py

3.2 Cập nhật generation schema — thêm citations field
    File: src/common.py

3.3 Parse và validate citation markers từ model output
    File: src/gen/p1_gen_stem_key.py (mới)

3.4 Cập nhật P2/P3 refine prompts — refine citations đồng thời

3.5 Thêm P9 (gen_explanation) — sinh explanation cho từng option
    File: src/gen/p9_explanation.py (mới)
```

### Phase 4: Evaluation (Tuần 4-5)

```
4.1 Thêm evaluation criteria cho citations và explanations
    File: src/eval/eval_overall.py

4.2 Cập nhật eval_overall prompts — LLM judge cho citation accuracy

4.3 Thêm eval citation accuracy metrics

4.4 Integration test: full pipeline từ indexing → generation → eval
```

### Phase 5: Sample Exams Integration (Tuần 5-6)

```
5.1 Index sample exam PDFs — extract style patterns
    File: src/gen/style_extractor.py (mới)

5.2 Tạo style prompt template từ sample exams

5.3 Tích hợp style template vào P1 prompt
```

---

## 12. Xác nhận

File thiết kế này được viết và lưu tại:
`/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen/EXPLAINABLE_QUESTION_GENERATION_PIPELINE.md`

---

## 13. Ba quyết định thiết kế quan trọng nhất

### Quyết định 1: Contextual Retrieval — Local LLM hay External API?

**Phương án A (Chọn):** Dùng **local Qwen2.5-14B-Instruct** cho contextual prefix generation
- Ưu: Không phụ thuộc OpenAI API, chi phí = 0, dữ liệu không rời khỏi server
- Nhược: Chậm hơn GPT-5-mini, quality có thể thấp hơn
- Phù hợp với: Dataset nhỏ (32 topics), không cần real-time

**Phương án B:** Dùng **OpenAI GPT-5-mini** (như tieplm)
- Ưu: Quality cao, nhanh
- Nhược: Chi phí API, phụ thuộc external service, data privacy concerns
- Phù hợp với: Production scale, budget available

**Quyết định: Phương án A** — Dùng local Qwen2.5-14B-Instruct cho Phase 1-2, có thể nâng cấp lên GPT-5-mini nếu quality không đạt.

---

### Quyết định 2: Citation Format — Inline markers [N] hay Structured JSON?

**Phương án A (Chọn):** **Hybrid — Inline markers [N] trong text + Structured JSON trong output**
- Inline: `[1]`, `[2]` trong `question_text` và `explanation` — dễ đọc, tương thích với frontend tieplm
- JSON: `citations` array trong output — dễ validate, render, và trace
- Giữ cả hai: Model generate inline, backend parse ra structured JSON

**Phương án B:** Chỉ structured JSON
- Ưu: Rõ ràng, dễ validate
- Nhược: Không tương thích với tieplm frontend (cần render inline text)

**Quyết định: Phương án A** — Tương thích ngược với hệ thống tieplm đã có, đồng thời có structured validation.

---

### Quyết định 3: Sample Exams — Index vào ChromaDB hay Chỉ Parse Style?

**Phương án A (Chọn):** **Không index sample exam vào ChromaDB, chỉ parse style patterns**
- Không embed → không thể retrieve fact từ sample exam
- Parse format, structure, độ khó → inject vào P1 prompt
- Đảm bảo factual grounding từ slide/transcript only

**Phương án B:** Index sample exam vào ChromaDB riêng
- Ưu: Retrieval có thể lấy style examples
- Nhược: Nguy cơ model lấy fact từ đề thi cũ → hallucination, không traceable

**Quyết định: Phương án A** — Bảo vệ factual integrity. Sample exam chỉ là style guide, không phải knowledge source. Nếu cần verify alignment với existing exams, dùng LLM judge ở evaluation step.
