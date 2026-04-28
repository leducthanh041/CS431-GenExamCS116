"""
common.py — MCQGen Pipeline Configuration & Prompts
====================================================
Chạy: 1 exp = 1 lần full pipeline (index → retrieval → gen → eval)
Đổi EXP_NAME trong class Config trước khi chạy.

Models: Qwen2.5-14B-Instruct (gen) + Gemma-3-12b-it (eval)
Technology: vLLM (vllm.LLM class), BGE-m3, ChromaDB
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Any

# ==============================================================================
# 1. EXPERIMENT CONFIG — ĐỔI TÊN EXP TRƯỚC KHI CHẠY
# ==============================================================================

class Config:
    # ⚠️  Đặt tên experiment tại đây — mỗi lần chạy pipeline nên đổi tên
    EXP_NAME = "full_pipeline"  # Mini test: 15 câu, focus ch04/ch07b/ch08

    # ─── Paths (tự động tính từ vị trí file này) ───────────────────
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Model paths (tải từ HuggingFace vào thư mục này)
    MODEL_ROOT = PROJECT_ROOT / "models"
    MODEL_GEN  = MODEL_ROOT / "Qwen2.5-14B-Instruct"
    MODEL_EVAL = MODEL_ROOT / "gemma-3-12b-it"

    # Input / Output
    INPUT_DIR         = PROJECT_ROOT / "input"
    TOPIC_LIST_FILE   = INPUT_DIR / "topic_list.json"

    # Data (đã copy từ cs431mcq)
    DATA_DIR          = PROJECT_ROOT / "data"
    PROCESSED_DIR     = DATA_DIR / "processed"
    INDEX_DIR         = DATA_DIR / "indexes"
    RAW_DIR           = DATA_DIR / "raw"
    INTERMEDIATE_DIR  = DATA_DIR / "intermediate"

    # Processed files
    CONCEPT_CHUNKS_FILE = PROCESSED_DIR / "concept_chunks.jsonl"
    ASSESSMENT_ITEMS_FILE = PROCESSED_DIR / "assessment_items.jsonl"

    # ChromaDB indexes — cả concept_kb và assessment_kb cùng nằm trong INDEX_DIR
    # (indexing.py lưu vào INDEX_DIR, không phải subfolder)
    CONCEPT_KB_DIR    = INDEX_DIR
    ASSESSMENT_KB_DIR = INDEX_DIR

    # Output cho từng step — gom vào 1 thư mục exp
    OUTPUT_DIR        = PROJECT_ROOT / "output" / EXP_NAME
    INDEX_OUTPUT      = OUTPUT_DIR / "01_indexing"
    RETRIEVE_OUTPUT   = OUTPUT_DIR / "02_retrieval"
    GEN_STEM_OUTPUT   = OUTPUT_DIR / "03_gen_stem"
    GEN_REFINE_OUTPUT = OUTPUT_DIR / "04_gen_refine"
    GEN_DISTR_OUTPUT  = OUTPUT_DIR / "05_gen_distractors"
    GEN_COT_OUTPUT    = OUTPUT_DIR / "06_gen_cot"
    EVAL_OUTPUT       = OUTPUT_DIR / "07_eval"
    EVAL_IWF_OUTPUT   = OUTPUT_DIR / "08_eval_iwf"
    EXPLAIN_OUTPUT    = OUTPUT_DIR / "09_explain"
    EXPLAIN_OUTPUT    = OUTPUT_DIR / "09_explain"
    FINAL_OUTPUT      = OUTPUT_DIR / "final"

    # Logs
    LOG_DIR           = PROJECT_ROOT / "log"

    # ─── Model names (HuggingFace) ───────────────────────────────────
    MODEL_GEN_NAME  = "Qwen/Qwen2.5-14B-Instruct"
    MODEL_EVAL_NAME = "google/gemma-3-12b-it"

    # ─── vLLM defaults ───────────────────────────────────────────────
    GEN_TP          = 1
    EVAL_TP         = 1
    GEN_MAX_TOKENS  = 4096
    EVAL_MAX_TOKENS = 1024
    GEN_TEMPERATURE = 0.7
    EVAL_TEMPERATURE = 0.1

    # ─── Retrieval settings ───────────────────────────────────────────
    RETRIEVAL_CONCEPT_TOP_K    = 5
    RETRIEVAL_ASSESSMENT_TOP_K = 3
    RETRIEVAL_MIN_SIM_CONCEPT  = 0.3
    RETRIEVAL_MIN_SIM_ASSESS   = 0.4

    # ─── Generation settings ──────────────────────────────────────────
    NUM_CANDIDATE_DISTRACTORS  = 6
    SINGLE_CORRECT_RATIO       = 0.6   # 80% single, 20% multiple (target ~5-7/30)
    RETRY_ON_FAILURE           = 2

    # ─── Evaluation thresholds ────────────────────────────────────────
    IWF_MAX_ERRORS = 3   # Auto-reject nếu > 3 lỗi IWF

    # ─── MCQ format ───────────────────────────────────────────────────
    MCQ_FIXED_OPTIONS        = 4
    MCQ_ALLOWED_CORRECT_COUNTS = [1, 2, 3]
    MCQ_LABEL_SINGLE         = "[Một đáp án đúng]"
    MCQ_LABEL_MULTIPLE       = "[Nhiều đáp án đúng]"
    MCQ_DIFFICULTIES         = ["G1", "G2", "G3"]

    @staticmethod
    def makedirs():
        """Tạo tất cả thư mục output/log cần thiết cho experiment."""
        dirs = [
            Config.OUTPUT_DIR,
            Config.INDEX_OUTPUT,
            Config.RETRIEVE_OUTPUT,
            Config.GEN_STEM_OUTPUT,
            Config.GEN_REFINE_OUTPUT,
            Config.GEN_DISTR_OUTPUT,
            Config.GEN_COT_OUTPUT,
            Config.EVAL_OUTPUT,
            Config.EVAL_IWF_OUTPUT,
            Config.EXPLAIN_OUTPUT,
            Config.EXPLAIN_OUTPUT,
            Config.FINAL_OUTPUT,
            Config.LOG_DIR,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        print(f"✅ Directories initialized for experiment: {Config.EXP_NAME}")


config = Config()


# ==============================================================================
# 2. PROMPTS P1–P8
# ==============================================================================

EXAM_STYLE_BLOCK = """[EXAM STYLE PROFILE]
Đây là câu hỏi dùng cho sinh viên đại học trong bối cảnh ôn tập hoặc thi cuối kỳ.
Style mong muốn:
- Bám sát ngôn ngữ học thuật của môn học, slide, transcript và đề thi sẵn có.
- Ưu tiên kiểm tra hiểu khái niệm, phân biệt phương pháp, điều kiện áp dụng,
  diễn giải kết quả, và so sánh kỹ thuật.
- Giữ câu hỏi ngắn gọn, rõ ràng, đúng kiểu đề thi cho sinh viên.
- Không sa đà vào tình huống quá thực tế, quá đời thường, hoặc quá thiên về
  bối cảnh doanh nghiệp nếu học liệu không yêu cầu.
- Chỉ dùng ví dụ ứng dụng khi ví dụ đó vẫn giữ đúng ngữ cảnh đề thi
  và giúp kiểm tra kiến thức cốt lõi.
- Không biến câu hỏi thành case study dài.
- Không thêm chi tiết ngoài context chỉ để làm câu hỏi trông thực tế hơn.
- Độ khó đến từ kiến thức, không đến từ văn phong rối hoặc bối cảnh
  ngoài chương trình."""

COURSE_KNOWLEDGE_SCOPE = (
    "Python (lập trình cơ bản và nâng cao), "
    "Machine Learning (hồi quy, phân loại, clustering, đánh giá mô hình), "
    "Deep Learning (neural networks, CNN, RNN, tối ưu hóa), "
    "và kiến thức về quy trình xây dựng một hệ thống máy học hoàn chỉnh "
    "(thu thập dữ liệu, tiền xử lý, feature engineering, huấn luyện, đánh giá, triển khai)."
)


# ─── P1: Generate Stem + Key ────────────────────────────────────────────────

def build_p1_gen_stem_key(
    topic: str,
    difficulty_target: str,
    concept_context_blocks: str,
    question_type_target: str | None = None,
    correct_answer_count_target: int | None = None,
    assessment_reference: str = "",
    num_questions_total: int = 1,
    num_single_correct: int = 1,
    num_multiple_correct: int = 0,
    num_two_correct: int = 0,
    num_three_correct: int = 0,
    course_name: str = "CS116 – Lập trình Python cho Máy học",
    chapter_or_week: str = "",
    subtopic: str = "",
) -> str:
    type_hint = (
        f"- Loại câu hỏi: {question_type_target}\n"
        if question_type_target else ""
    )
    count_hint = (
        f"- Số đáp án đúng: {correct_answer_count_target}\n"
        if correct_answer_count_target else ""
    )
    ref_block = (
        f"\n\n[STYLE REFERENCE — Câu hỏi mẫu từ đề thi sẵn có]\n"
        f"{assessment_reference}\n"
        if assessment_reference else ""
    )
    return f"""[ROLE]
Bạn là giảng viên trường đại học dạy môn {course_name}.
Bạn có kiến thức về {COURSE_KNOWLEDGE_SCOPE}.
Bạn đang biên soạn câu hỏi cho sinh viên đại học để dùng trong đề thi hoặc bộ câu ôn tập cuối kỳ.

[HARD CONSTRAINTS — BẮT BUỘC TUÂN THỦ tuyệt đối]
[HARD CONSTRAINTS — BẮT BUỘC TUÂN THỦ tuyệt đối]
- Câu hỏi phải bằng tiếng Việt.
- Câu hỏi phải phù hợp với ngữ cảnh ra đề cho sinh viên đại học.
- Câu hỏi phải bám sát context và style đề thi đã cung cấp.
- Không tự ý thêm bối cảnh thực tế dài dòng nếu context không yêu cầu.
- Câu hỏi cuối cùng của hệ thống sẽ có đúng 4 options.
- Số đáp án đúng của câu này phải thuộc {{1, 2, 3}}.
- **TUYỆT ĐỐI: Loại câu hỏi phải là "{question_type_target}" cho câu này.**
  Nếu question_type_target = "single_correct" → câu này phải có ĐÚNG 1 đáp án đúng.
  Nếu question_type_target = "multiple_correct" → câu này phải có ĐÚNG {correct_answer_count_target} đáp án đúng.
  **KHÔNG ĐƯỢC tự ý đổi loại câu hỏi sang loại khác.**
- Nếu là "single_correct" → stem phải có nhãn [Một đáp án đúng].
- Nếu là "multiple_correct" → stem phải có nhãn [Nhiều đáp án đúng].
- **TUYỆT ĐỐI: Loại câu hỏi phải là "{question_type_target}" cho câu này.**
  Nếu question_type_target = "single_correct" → câu này phải có ĐÚNG 1 đáp án đúng.
  Nếu question_type_target = "multiple_correct" → câu này phải có ĐÚNG {correct_answer_count_target} đáp án đúng.
  **KHÔNG ĐƯỢC tự ý đổi loại câu hỏi sang loại khác.**
- Nếu là "single_correct" → stem phải có nhãn [Một đáp án đúng].
- Nếu là "multiple_correct" → stem phải có nhãn [Nhiều đáp án đúng].
- Nếu context không đủ thông tin thì phải từ chối sinh câu hỏi.
- Không sinh distractor ở bước này.

[STRICT RULE — CÂU HỎI KHÔNG ĐƯỢC CHỨA ĐÁP ÁN
[STRICT RULE — CÂU HỎI KHÔNG ĐƯỢC CHỨA ĐÁP ÁN
- **TUYỆT ĐỐI KHÔNG** được ghi đáp án đúng (hoặc một phần nội dung của đáp án đúng) vào trong question_text / stem.
- Ví dụ SAI: "Phương pháp nào sau đây KHÔNG thuộc kỹ thuật ensemble: Boosting, Bagging, Random Forest, hoặc SVM?" ← đã ghi đáp án đúng (Boosting, Bagging, Random Forest) vào stem!
- Ví dụ SAI: "Trong các phương pháp sau: (i) Khi dữ liệu bị thiếu phân bố ngẫu nhiên...; (ii) Khi dữ liệu bị thiếu tập trung ở một số hàng... — câu hỏi nào đúng?" ← đã ghi đáp án vào stem!
- **CHỈ MÔ TẢ CHỦ ĐỀ / VẤN ĐỀ** trong stem, ví dụ: "Phương pháp nào sau đây không thuộc nhóm kỹ thuật ensemble learning?" ← KHÔNG liệt kê đáp án trong stem.
- **Lưu ý quan trọng**: Nếu câu hỏi có dạng liệt kê (i)(ii)(iii)(iv) trong stem, bạn phải di chuyển toàn bộ các mục (i)(ii)(iii)(iv) vào phần OPTIONS (A/B/C/D), KHÔNG giữ chúng trong stem.

[DIVERSITY RULE — TRÁNH CÁC CÁCH MỞ ĐẦU YẾU, TẠO SỰ TÒ MÒ]
- **Mỗi câu hỏi phải có cách mô tả ĐA DẠNG, không bắt đầu bằng cùng một từ/cụm từ.**
- **CÁC TỪ/CỤM TỪ SAU ĐÂY TUYỆT ĐỐI KHÔNG được dùng làm mở đầu câu hỏi:**
  - ❌ "Hãy xác định..." (quá chung chung, gợi nhớ bài kiểm tra)
  - ❌ "khi" (thường dẫn đến câu hỏi dài, không rõ ràng)
  - ❌ "đâu" (câu hỏi mơ hồ, không đủ thông tin)
  - ❌ "Trong quá trình..." (quá dài, không cần thiết)
  - ❌ "Khi xây dựng..." / "Khi huấn luyện..." (lặp, không tạo tò mò)
  - ❌ "Trong các phương pháp..." (lặp, quá generic)
  - ❌ "Trong các kỹ thuật..." (tương tự)
  - ❌ "Cho biết..." (lặp, không tạo hứng thú)
- **ƯU TIÊN các cách mở đầu TẠO SỰ TÒ MÒ, học thuật, đúng kiểu đề thi:**
  - ✅ "Điều gì khiến..." (gợi sự tò mò)
  - ✅ "Đâu là điểm khác biệt giữa... và...?" (so sánh, phổ biến trong đề thi)
  - ✅ "Nếu phải chọn giữa... và..., bạn sẽ ưu tiên điều gì?" (tình huống thực tế ngắn)
  - ✅ "Một mô hình có đặc điểm... sẽ hoạt động ra sao khi...?" (áp dụng kiến thức)
  - ✅ "Trường hợp nào sau đây minh họa đúng nhất về...?" (nhận định đúng sai)
  - ✅ "Điều kiện tiên quyết để... hoạt động hiệu quả là gì?" (điều kiện)
  - ✅ "Sau khi áp dụng..., kết quả mong đợi là gì?" (dự đoán)
  - ✅ "Quan sát đoạn code sau, output nào phù hợp nhất?" (code-based)
  - ✅ "Nếu thay đổi tham số... thì điều gì sẽ xảy ra với...?" (phân tích)
  - ✅ "Vai trò chính của... trong kiến trúc này là gì?" (vai trò/thành phần)
  - ✅ "Nhận định nào sau đây là chính xác nhất về...?" (đánh giá)
  - ✅ "Mục đích chính của... là gì?" (mục đích)
  - ✅ "Tính chất nào giúp phân biệt... với...?" (so sánh tính chất)
  - ✅ "Phương pháp nào..." (cổ điển, dùng được nhưng tránh lặp)
  - ✅ "Câu lệnh nào..." / "Hàm nào..." (code-based)
- **Nếu batch có nhiều câu hỏi, mỗi câu phải dùng cách mở đầu KHÁC NHAU từ danh sách ưu tiên.**
- **KHÔNG bắt đầu 2 câu liên tiếp bằng cùng một cụm từ.**
[DIVERSITY RULE — TRÁNH CÁC CÁCH MỞ ĐẦU YẾU, TẠO SỰ TÒ MÒ]
- **Mỗi câu hỏi phải có cách mô tả ĐA DẠNG, không bắt đầu bằng cùng một từ/cụm từ.**
- **CÁC TỪ/CỤM TỪ SAU ĐÂY TUYỆT ĐỐI KHÔNG được dùng làm mở đầu câu hỏi:**
  - ❌ "Hãy xác định..." (quá chung chung, gợi nhớ bài kiểm tra)
  - ❌ "khi" (thường dẫn đến câu hỏi dài, không rõ ràng)
  - ❌ "đâu" (câu hỏi mơ hồ, không đủ thông tin)
  - ❌ "Trong quá trình..." (quá dài, không cần thiết)
  - ❌ "Khi xây dựng..." / "Khi huấn luyện..." (lặp, không tạo tò mò)
  - ❌ "Trong các phương pháp..." (lặp, quá generic)
  - ❌ "Trong các kỹ thuật..." (tương tự)
  - ❌ "Cho biết..." (lặp, không tạo hứng thú)
- **ƯU TIÊN các cách mở đầu TẠO SỰ TÒ MÒ, học thuật, đúng kiểu đề thi:**
  - ✅ "Điều gì khiến..." (gợi sự tò mò)
  - ✅ "Đâu là điểm khác biệt giữa... và...?" (so sánh, phổ biến trong đề thi)
  - ✅ "Nếu phải chọn giữa... và..., bạn sẽ ưu tiên điều gì?" (tình huống thực tế ngắn)
  - ✅ "Một mô hình có đặc điểm... sẽ hoạt động ra sao khi...?" (áp dụng kiến thức)
  - ✅ "Trường hợp nào sau đây minh họa đúng nhất về...?" (nhận định đúng sai)
  - ✅ "Điều kiện tiên quyết để... hoạt động hiệu quả là gì?" (điều kiện)
  - ✅ "Sau khi áp dụng..., kết quả mong đợi là gì?" (dự đoán)
  - ✅ "Quan sát đoạn code sau, output nào phù hợp nhất?" (code-based)
  - ✅ "Nếu thay đổi tham số... thì điều gì sẽ xảy ra với...?" (phân tích)
  - ✅ "Vai trò chính của... trong kiến trúc này là gì?" (vai trò/thành phần)
  - ✅ "Nhận định nào sau đây là chính xác nhất về...?" (đánh giá)
  - ✅ "Mục đích chính của... là gì?" (mục đích)
  - ✅ "Tính chất nào giúp phân biệt... với...?" (so sánh tính chất)
  - ✅ "Phương pháp nào..." (cổ điển, dùng được nhưng tránh lặp)
  - ✅ "Câu lệnh nào..." / "Hàm nào..." (code-based)
- **Nếu batch có nhiều câu hỏi, mỗi câu phải dùng cách mở đầu KHÁC NHAU từ danh sách ưu tiên.**
- **KHÔNG bắt đầu 2 câu liên tiếp bằng cùng một cụm từ.**

{EXAM_STYLE_BLOCK}

[COURSE / EXAM CONTEXT]
- Môn học: {course_name}
- Chủ đề lớn: {topic}
- Chủ đề con: {subtopic}
- Chapter / tuần học: {chapter_or_week}
- Mức độ khó mục tiêu: {difficulty_target}
- Kiểu câu hỏi mục tiêu cho câu này: {question_type_target or 'tự chọn theo context'}
- Số đáp án đúng mục tiêu cho câu này: {correct_answer_count_target or 'tự chọn theo context'}

[BATCH CONTEXT]
- Tổng số câu cần sinh cho batch: {num_questions_total}
- Số câu một đáp án đúng cần sinh: {num_single_correct}
- Số câu nhiều đáp án đúng cần sinh: {num_multiple_correct}
- Trong các câu nhiều đáp án đúng:
  - Số câu có đúng 2 đáp án đúng: {num_two_correct}
  - Số câu có đúng 3 đáp án đúng: {num_three_correct}

[CONCEPT CONTEXT]
{concept_context_blocks}

[STYLE REFERENCE FROM EXISTING EXAMS / QUIZZES]
{assessment_reference if assessment_reference else '(không có — dùng style học thuật chuẩn môn)'}

[TASK]
Hãy sinh trước:
1. Stem hoặc question_text
2. correct answer set ở mức nội dung (chưa cần gán A/B/C/D)
3. question_type và correct_answer_count

Chỉ sinh câu hỏi phù hợp với ngữ cảnh đề thi cho sinh viên đại học.
Không sinh distractor ở bước này.

[OUTPUT FORMAT - JSON ONLY]
HƯỚNG DẪN NGHIÊM NGẶT: Trường "question_type" trong JSON output phải KHỚP CHÍNH XÁC với "Loại câu hỏi mục tiêu cho câu này" ở trên:
  - Nếu mục tiêu là "single_correct" → question_type phải là "single_correct"
  - Nếu mục tiêu là "multiple_correct" → question_type phải là "multiple_correct"
  Sai: question_type ghi "single_correct" nhưng thực tế câu có 3 đáp án đúng.
HƯỚNG DẪN NGHIÊM NGẶT: Trường "question_type" trong JSON output phải KHỚP CHÍNH XÁC với "Loại câu hỏi mục tiêu cho câu này" ở trên:
  - Nếu mục tiêu là "single_correct" → question_type phải là "single_correct"
  - Nếu mục tiêu là "multiple_correct" → question_type phải là "multiple_correct"
  Sai: question_type ghi "single_correct" nhưng thực tế câu có 3 đáp án đúng.
{{
  "draft_question_id": "<string>",
  "question_text": "<string — BẮT BUỘC phải có nhãn phù hợp: [Một đáp án đúng] hoặc [Nhiều đáp án đúng]>",
  "question_type": "{question_type_target}",
  "question_text": "<string — BẮT BUỘC phải có nhãn phù hợp: [Một đáp án đúng] hoặc [Nhiều đáp án đúng]>",
  "question_type": "{question_type_target}",
  "correct_answers_content": [
    "<đáp án đúng 1>",
    "<đáp án đúng 2 nếu có>",
    "<đáp án đúng 3 nếu có>"
  ],
  "correct_answer_count": {correct_answer_count_target},
  "correct_answer_count": {correct_answer_count_target},
  "topic": "<string>",
  "subtopic": "<string>",
  "difficulty_label": "<string>",
  "used_concept_chunk_ids": ["<chunk_id_1>", "<chunk_id_2>"],
  "sources": [],
  "sources": [],
  "style_alignment_note": "<ngắn gọn, nêu vì sao câu này phù hợp với ngữ cảnh đề thi>",
  "stem_has_answer": false,
  "answer_in_stem_warning": "<mô tả nếu phát hiện đáp án nằm trong stem, hoặc 'none'>"
}}
"""


# ─── P2: Self-Refine Suggestion ───────────────────────────────────────────────

def build_p2_refine_suggest(draft_stem_key_json: dict[str, Any], difficulty_target: str) -> str:
    draft_str = json.dumps(draft_stem_key_json, ensure_ascii=False, indent=2)
    return f"""[ROLE]
Bạn là giảng viên trường đại học có kinh nghiệm thiết kế câu hỏi trắc nghiệm
cho đề thi và ôn tập cuối kỳ.
Nhiệm vụ của bạn là phê bình và cải tiến câu hỏi draft để phù hợp hơn
với ngữ cảnh ra đề cho sinh viên đại học.

[INPUT — Câu hỏi draft]
{draft_str}

[GOAL]
- Cải tiến câu hỏi để phù hợp hơn với ngữ cảnh đề thi cho sinh viên đại học.
- Tăng độ khó từ mức hiện tại → {difficulty_target} (nếu phù hợp).
- Không thay đổi bản chất kiến thức kiểm tra.
- Không thay đổi tập đáp án đúng (correct_answers_content).
- Câu hỏi sau cải tiến phải vẫn kiểm tra cùng concept.

[STRICT RULE — question_text KHÔNG được chứa đáp án]
- **TUYỆT ĐỐI KHÔNG** được ghi đáp án đúng (hoặc một phần nội dung đáp án đúng) vào question_text.
- stem chỉ mô tả CHỦ ĐỀ / VẤN ĐỀ, ví dụ: "Phương pháp nào sau đây không thuộc nhóm kỹ thuật ensemble learning?" ← KHÔNG liệt kê Boosting, Bagging, Random Forest trong stem.
- Nếu câu hỏi dạng liệt kê (i)(ii)(iii)(iv) — di chuyển các mục này vào OPTIONS (A/B/C/D), không giữ trong stem.

[DIVERSITY — TRÁNH LẶP CÁCH MỞ ĐẦU]
- Cách mô tả của câu hỏi phải khác biệt so với các câu trước.
- Tránh: "Trong quá trình...", "Khi...", "Trong các phư thuật..."
- Ưu tiên: "Cho biết...", "Hãy xác định...", "Đâu là...", "Phương pháp nào...", "Câu lệnh nào...", "Hàm nào...", "Kết quả nào...", "Output là gì...", "Thuật toán nào...", "Công thức nào...", "Điều kiện nào...", "Tình huống nào...", "Trường hợp nào...", "Đặc điểm nào...", "Nguyên nhân nào..."

{EXAM_STYLE_BLOCK}

[HARD CONSTRAINTS]
- Chỉ đề xuất MỘT cách cải tiến cụ thể, hành động được.
- Ưu tiên cải tiến theo hướng:
    rõ ý hơn, chuẩn thuật ngữ hơn, đúng trọng tâm hơn,
    và nếu cần thì khó hơn một cách hợp lý.
- KHÔNG đề xuất cải tiến theo hướng:
    biến câu hỏi thành tình huống thực tế dài dòng,
    thêm bối cảnh doanh nghiệp / đời thường nếu học liệu không yêu cầu.
- Không đề xuất thay đổi đáp án đúng.
- Cải tiến phải phù hợp với Bloom's Taxonomy của {difficulty_target}:
    G1 = Nhớ/Hiểu (nhận biết, định nghĩa)
    G2 = Áp dụng/Phân tích (tính toán, so sánh, phân tích tình huống)
    G3 = Đánh giá/Thiết kế (phê phán, đề xuất, tổng hợp)

[IMPORTANT]
Mô tả cải tiến bằng các từ như: "phù hợp với đề thi", "đúng ngữ cảnh ra đề",
"chặt hơn", "sát kiến thức", "đúng trọng tâm".
KHÔNG dùng các từ như: "thực tế", "sinh động", "ứng dụng cao",
"tình huống doanh nghiệp" khi học liệu không yêu cầu.

[OUTPUT FORMAT - JSON ONLY]
{{
  "improvement_suggestion": "<một gợi ý cụ thể, hành động được>",
  "why_it_is_exam_appropriate": "<vì sao gợi ý này làm câu hỏi phù hợp hơn với ngữ cảnh đề thi>",
  "difficulty_effect": "<không đổi / tăng nhẹ / tăng vừa>"
}}
"""


# ─── P3: Apply Refinement ─────────────────────────────────────────────────────

def build_p3_refined_stem(
    draft_stem_key_json: dict[str, Any],
    improvement_suggestion_json: dict[str, Any],
) -> str:
    draft_str = json.dumps(draft_stem_key_json, ensure_ascii=False, indent=2)
    suggest_str = json.dumps(improvement_suggestion_json, ensure_ascii=False, indent=2)
    return f"""[ROLE]
Bạn là giảng viên trường đại học chuyên viết câu hỏi trắc nghiệm cho đề thi.
Nhiệm vụ của bạn là viết lại câu hỏi dựa trên gợi ý cải tiến.

[INPUT — Câu hỏi gốc]
{draft_str}

[SUGGESTION — Gợi ý cải tiến từ bước trước]
{suggest_str}

{EXAM_STYLE_BLOCK}

[TASK]
1. Áp dụng gợi ý để viết lại câu hỏi (stem).
2. GIỮ NGUYÊN tập đáp án đúng (correct_answers_content).
3. GIỮ NGUYÊN question_type và correct_answer_count.
4. Stem mới phải bằng tiếng Việt, rõ ràng, phù hợp với ngữ cảnh đề thi.

[HARD CONSTRAINTS]
- Không thay đổi correct_answers_content.
- Không thay đổi correct_answer_count.
- Nếu cải tiến làm tăng độ khó → sự tăng độ khó PHẢI đến từ kiến thức,
  không phải từ việc thêm bối cảnh thực tế, tình huống doanh nghiệp,
  hay ví dụ ngoài chương trình học.
- Nếu câu mới khó hơn câu cũ, câu mới vẫn phải đúng ngữ cảnh đề thi.
  Không đánh đố bằng ngôn ngữ mơ hồ hoặc tình huống giả lập phức tạp
  không xuất hiện trong học liệu.
- Nếu là multiple_correct → stem phải ghi rõ [Nhiều đáp án đúng].
- Nếu là single_correct → stem phải ghi rõ [Một đáp án đúng].

[OUTPUT FORMAT - JSON ONLY]
{{
  "refined_question_text": "<câu hỏi mới bằng tiếng Việt, có nhãn loại>",
  "question_type": "single_correct hoặc multiple_correct",
  "correct_answers_content": ["<giữ nguyên từ input>"],
  "correct_answer_count": <giữ nguyên từ input>,
  "topic": "<giữ nguyên từ input>",
  "subtopic": "<giữ nguyên từ input nếu có>",
  "difficulty_label": "<giữ nguyên từ input>",
  "style_alignment_note": "<ngắn gọn, nêu vì sao câu mới vẫn đúng ngữ cảnh đề thi>"
}}
"""


# ─── P4: Generate Distractor Candidates ────────────────────────────────────────

def build_p4_option_candidates(
    refined_stem_key_json: dict[str, Any],
    similar_mcqs_reference: str = "",
    num_candidates: int = 6,
    assessment_style_examples: str = "",
) -> str:
    stem_key_str = json.dumps(refined_stem_key_json, ensure_ascii=False, indent=2)
    correct_answer_count = refined_stem_key_json.get("correct_answer_count", 1)
    num_distractors_needed = 4 - correct_answer_count
    ref_block = (
        f"\n\n[STYLE REFERENCE FROM EXISTING EXAMS / QUIZZES]\n"
        f"{assessment_style_examples}\n"
        if assessment_style_examples else ""
    )
    rag_block = (
        f"\n\n[MCQ TƯƠNG TỰ TỪ DATABASE — RAG retrieval]\n{similar_mcqs_reference}\n"
        if similar_mcqs_reference else ""
    )
    return f"""[ROLE]
Bạn là giảng viên trường đại học chuyên thiết kế phương án nhiễu (distractors)
cho câu hỏi trắc nghiệm trong đề thi.
Nhiệm vụ của bạn là sinh các distractor candidates dựa trên câu hỏi đã được
tinh chỉnh ở bước trước.

[INPUT — Câu hỏi đã tinh chỉnh (TỪ P3)]
{stem_key_str}

{EXAM_STYLE_BLOCK}

[IMPORTANT]
Stem và đáp án đúng đã có ở trên. Bước này CHỈ sinh distractor candidates.
KHÔNG được sinh lại đáp án đúng.
KHÔNG được thay đổi stem.
KHÔNG được thay đổi tập đáp án đúng.

[CONSTRAINTS]
- Tổng số distractor cần sinh: {num_candidates}
- Số distractor cần thiết cho câu hỏi này: {num_distractors_needed}
  (vì correct_answer_count = {correct_answer_count})
- Các distractor phải SAI NHƯNG HỢP LÝ — sai trong ngữ cảnh cụ thể của câu hỏi.
- Các distractor phải đánh trúng các lỗi sai phổ biến của sinh viên.
- KHÔNG dùng: "Tất cả đáp án trên", "Không đáp án nào đúng",
  "Đáp án A và B đều đúng".
- Không được paraphrase quá gần với correct answers.
- Không lộ mẹo làm bài bằng grammar clue, absolute terms, hoặc độ dài quá khác biệt.
- Tất cả distractors bằng tiếng Việt, phù hợp ngữ cảnh đề thi.{ref_block}{rag_block}

[TASK]
1. Đọc kỹ stem và correct answers từ input.
2. Sinh {num_candidates} distractor candidates theo các tiêu chí trên.
3. Mỗi distractor phải có lý do sai cụ thể (ghi trong reasoning).

[OUTPUT FORMAT - JSON ONLY]
{{
  "candidate_distractors": [
    "<distractor 1>",
    "<distractor 2>",
    "<distractor 3>",
    "<distractor 4>",
    "<distractor 5>",
    "<distractor 6>"
  ],
  "style_alignment_note": "<ngắn gọn, nêu vì sao distractors phù hợp style đề thi>"
}}
"""


# ─── P5: CoT Evaluate Candidates ──────────────────────────────────────────────

def build_p5_cot_evaluate(
    refined_stem_key_json: dict[str, Any],
    all_candidates: list[str],
    correct_options: list[str],
) -> str:
    stem_key_str = json.dumps(refined_stem_key_json, ensure_ascii=False, indent=2)
    options_str = "\n".join(
        f"  Option {i+1}: {opt}" for i, opt in enumerate(all_candidates)
    )
    return f"""[ROLE]
Bạn là giảng viên trường đại học, chuyên gia đánh giá câu hỏi trắc nghiệm
(MCQ Item Writing Expert) cho đề thi và ôn tập cuối kỳ.
Nhiệm vụ của bạn là đánh giá từng candidate option theo các tiêu chí nghiêm ngặt.

[INPUT — Câu hỏi đã tinh chỉnh]
{stem_key_str}

[CANDIDATE OPTIONS]
{options_str}

{EXAM_STYLE_BLOCK}

[EVALUATION CRITERIA — đánh giá từng option]
1. is_correct: Option này có phải là đáp án đúng không?
2. relevance: Có liên quan trực tiếp đến topic câu hỏi không?
3. misleading_likelihood: Có khả năng gây nhầm lẫn cao (vì gần đúng) không?
4. grammar_ok: Có lỗi ngữ pháp nghiêm trọng không?
5. logical_error: Có lỗi logic rõ ràng không?
6. absolute_term: Có chứa từ tuyệt đối gây clue không?
   (e.g. "luôn luôn", "tất cả", "không bao giờ")

[OUTPUT FORMAT - JSON ONLY]
{{
  "evaluations": [
    {{
      "option_index": <0>,
      "option_text": "<text>",
      "is_correct": true|false,
      "relevance": <0-10>,
      "misleading_likelihood": <0-10>,
      "grammar_ok": true|false,
      "logical_error": "<mô tả lỗi nếu có, hoặc 'none'>",
      "absolute_term": true|false,
      "notes": "<ngắn gọn>"
    }}
  ]
}}
"""


# ─── P6: Remove Bad Distractors ──────────────────────────────────────────────

def build_p6_remove_bad(
    refined_stem_key_json: dict[str, Any],
    all_candidates: list[str],
    p5_evaluations: list[dict[str, Any]],
) -> str:
    stem_key_str = json.dumps(refined_stem_key_json, ensure_ascii=False, indent=2)
    eval_str = json.dumps(p5_evaluations, ensure_ascii=False, indent=2)
    return f"""[ROLE]
Bạn là giảng viên trường đại học, chuyên gia lọc distractors cho đề thi trắc nghiệm.
Nhiệm vụ của bạn là loại bỏ các phương án nhiễu không phù hợp với ngữ cảnh
ra đề cho sinh viên đại học.

[INPUT — Câu hỏi]
{stem_key_str}

[P5 EVALUATIONS — Kết quả đánh giá từ bước trước]
{eval_str}

{EXAM_STYLE_BLOCK}

[RULES — Loại bỏ nếu CÓ ÍT NHẤT 1 trong các điều kiện:]
a) Quá sai rõ ràng: misleading_likelihood ≤ 2
b) Không liên quan: relevance ≤ 3
c) Lỗi ngữ pháp nghiêm trọng: grammar_ok = false
d) Từ tuyệt đối gây clue rõ ràng: absolute_term = true
e) Lỗi logic rõ ràng (diễn giải được)
f) Không phù hợp ngữ cảnh đề thi học thuật

[CONSTRAINTS]
- GIỮ NGUYÊN đáp án đúng.
- Chỉ loại distractors (phương án SAI).
- Ghi rõ LÝ DO loại bỏ cho từng option.
- Nếu tất cả distractors đều bị loại → ghi "NONE_PASSED"

[OUTPUT FORMAT - JSON ONLY]
{{
  "removed_options": [
    {{
      "option_index": <int>,
      "option_text": "<text>",
      "reason": "<lý do loại bỏ>"
    }}
  ],
  "kept_options": [
    {{
      "option_index": <int>,
      "option_text": "<text>",
      "reason": "<tại sao giữ lại>"
    }}
  ]
}}
"""


# ─── P7: Select Final Distractors ───────────────────────────────────────────

def build_p7_select_final(
    refined_stem_key_json: dict[str, Any],
    kept_options: list[dict[str, Any]],
    correct_answer_count: int,
) -> str:
    stem_key_str = json.dumps(refined_stem_key_json, ensure_ascii=False, indent=2)
    kept_str = json.dumps(kept_options, ensure_ascii=False, indent=2)
    num_distractors_needed = 4 - correct_answer_count
    return f"""[ROLE]
Bạn là giảng viên trường đại học, chuyên gia thiết kế MCQ cuối cùng
cho đề thi và ôn tập cuối kỳ.
Nhiệm vụ của bạn là chọn {num_distractors_needed} distractor cuối cùng
từ tập kept candidates.

[INPUT — Câu hỏi đã tinh chỉnh]
{stem_key_str}

[KEPT CANDIDATE OPTIONS — sau khi lọc từ P6]
{kept_str}

{EXAM_STYLE_BLOCK}

[CONSTRAINTS]
- Cần chọn đúng {num_distractors_needed} distractors.
- Các distractor phải ĐẠI DIỆN CHO CÁC LỖI SAI KHÁC NHAU (distinct error types).
- Ưu tiên distractors có misleading_likelihood cao và relevance cao.
- Không chọn 2 distractors quá giống nhau (cùng sai theo cùng 1 cách).
- Nếu có ít hơn {num_distractors_needed} kept options → dùng tất cả và ghi rõ lý do.

[OUTPUT FORMAT - JSON ONLY]
{{
  "selected_distractors": [
    {{
      "option_text": "<text>",
      "error_type": "<loại lỗi sai (e.g. nhầm khái niệm, thiếu điều kiện, đảo ngược quan hệ)>",
      "misleading_score": <0-10>
    }}
  ],
  "rejected_distractors": [
    {{
      "option_text": "<text>",
      "reason": "<tại sao không chọn>"
    }}
  ],
  "selection_rationale": "<ngắn gọn giải thích cách chọn>"
}}
"""


# ─── P8: Assemble Final MCQ ───────────────────────────────────────────────────

def build_p8_assemble(
    refined_stem_key_json: dict[str, Any],
    selected_distractors: list[dict[str, Any]],
    correct_options: list[str],
    used_concept_chunk_ids: list[str] | None = None,
    used_assessment_item_ids: list[str] | None = None,
) -> str:
    stem_key_str = json.dumps(refined_stem_key_json, ensure_ascii=False, indent=2)
    distractor_str = json.dumps(selected_distractors, ensure_ascii=False, indent=2)
    chunk_ids_str = json.dumps(used_concept_chunk_ids or [], ensure_ascii=False)
    assess_ids_str = json.dumps(used_assessment_item_ids or [], ensure_ascii=False)
    return f"""[ROLE]
Bạn là giảng viên trường đại học hoàn thiện câu hỏi trắc nghiệm cho đề thi và ôn tập cuối kỳ.
Nhiệm vụ của bạn là lắp ráp câu hỏi hoàn chỉnh với 4 options A/B/C/D.

[INPUT — Câu hỏi tinh chỉnh]
{stem_key_str}

[CORRECT OPTIONS]
{chr(10).join(f"  - {opt}" for opt in correct_options)}

[SELECTED DISTRACTORS]
{chr(10).join(f"  - {d['option_text']}" for d in selected_distractors)}

{EXAM_STYLE_BLOCK}

[CONSTRAINTS]
- Tổng số options = 4.
- Gán nhãn A, B, C, D cho 4 options.
- RANDOMIZE thứ tự: xáo trộn vị trí giữa correct options và distractors.
- Nếu là multiple_correct → stem phải ghi rõ: [Nhiều đáp án đúng]
- Nếu là single_correct → stem phải ghi rõ: [Một đáp án đúng]
- Stem phải bám sát refined_question_text từ input.
- Giữ giọng văn ngắn gọn, học thuật, đúng kiểu đề thi.
- Chỉ trả về JSON hợp lệ, không có text khác.

[OUTPUT FORMAT — JSON ONLY]
{{
  "question_id": "<string — format: cs116_<chapter>_<seq>",
  "question_text": "<string — câu hỏi bằng tiếng Việt, có nhãn loại và độ khó>",
  "question_type": "single_correct hoặc multiple_correct",
  "options": {{
    "A": "<text>",
    "B": "<text>",
    "C": "<text>",
    "D": "<text>"
  }},
  "correct_answers": ["<A>", "<B>", "<C>"],
  "correct_answer_count": <1|2|3>,
  "topic": "<string>",
  "difficulty_label": "<string>",
  "used_concept_chunk_ids": {chunk_ids_str},
  "used_assessment_item_ids": {assess_ids_str},
  "style_alignment_note": "<ngắn gọn>"
}}
"""


# ─── Evaluation Prompts ──────────────────────────────────────────────────────

EVAL_OVERALL_SYSTEM = """Bạn là chuyên gia đánh giá câu hỏi trắc nghiệm (MCQ Item Writing Expert).
Nhiệm vụ: kiểm tra câu hỏi MCQ hoàn chỉnh theo 6 tiêu chí nghiêm ngặt.
Chỉ trả về JSON hợp lệ."""


def build_eval_overall_prompt(mcq_json: dict[str, Any]) -> str:
    mcq_str = json.dumps(mcq_json, ensure_ascii=False, indent=2)
    return f"""[ROLE]
Bạn là chuyên gia đánh giá câu hỏi trắc nghiệm (MCQ Item Writing Expert).
Đánh giá câu hỏi MCQ hoàn chỉnh theo 8 tiêu chí:

1. format_pass: Câu hỏi có đúng 4 options A/B/C/D, có nhãn loại, có nhãn độ khó không?
2. language_pass: Câu hỏi bằng tiếng Việt, rõ ràng, không lỗi chính tả không?
3. grammar_pass: Các options có cùng ngữ pháp, cùng độ dài tương đối không?
4. relevance_pass: Câu hỏi và tất cả options đều liên quan đến topic không?
5. answerability_pass: Sinh viên có đủ thông tin trong stem để trả lời mà không cần đoán không?
6. correct_set_pass: Tập đáp án đúng có hợp lý, không dư thừa, không mâu thuẫn không?
7. NO_FOUR_CORRECT_PASS: Không có câu hỏi nào có tất cả 4 options đều là đáp án đúng.
   Luôn kiểm tra: nếu correct_answer_count = 4 hoặc tất cả A/B/C/D đều được đánh dấu đúng → FAIL.
   Câu hỏi trắc nghiệm phải có đáp án sai (distractors) → không thể có 4 đáp án đúng.
8. ANSWER_NOT_IN_STEM_PASS: Đáp án đúng không được xuất hiện (dù là 1 phần hay toàn bộ) trong câu hỏi (stem).
   Kiểm tra: nếu nội dung correct_answers xuất hiện trong question_text → FAIL.

[INPUT — MCQ cần đánh giá]
{mcq_str}

[HARD CONSTRAINTS]
- Nếu bất kỳ tiêu chí nào thất bại → overall_valid = false
- **Đặc biệt chú ý**: tiêu chí 7 và 8 là HARD REJECT — câu hỏi vi phạm sẽ bị loại tuyệt đối.
- Chỉ trả về JSON, không có text giải thích
- Dùng tiếng Việt cho mô tả lỗi

[OUTPUT FORMAT — JSON ONLY]
{{
  "format_pass": true|false,
  "language_pass": true|false,
  "grammar_pass": true|false,
  "relevance_pass": true|false,
  "answerability_pass": true|false,
  "correct_set_pass": true|false,
  "no_four_correct_pass": true|false,
  "answer_not_in_stem_pass": true|false,
  "overall_valid": true|false,
  "fail_reasons": ["<lý do thất bại nếu có>"],
  "quality_score": <0.0-1.0>
}}
"""


EVAL_IWF_SYSTEM = """Bạn là chuyên gia phân tích lỗi viết câu hỏi trắc nghiệm (Item Writing Flaws).
Kiểm tra từng distractor (option SAI) theo 6 loại IWF phổ biến.
Chỉ trả về JSON hợp lệ."""


def build_eval_iwf_prompt(mcq_json: dict[str, Any]) -> str:
    mcq_str = json.dumps(mcq_json, ensure_ascii=False, indent=2)
    return f"""[ROLE]
Bạn là chuyên gia phân tích lỗi viết câu hỏi trắc nghiệm (Item Writing Flaws).
Kiểm tra từng distractor (phương án SAI) theo 6 loại IWF:

1. plausible_distractor: Distractor có sai sót thường gặp ở sinh viên không?
   (nếu quá rõ ràng là sai → NOT plausible = lỗi)
2. vague_terms: Distractor có từ ngữ mơ hồ, không rõ ràng không?
3. grammar_clue: Distractor có lỗi ngữ pháp/format gây clue cho đáp án đúng không?
4. absolute_terms: Distractor có từ tuyệt đối ("luôn", "tất cả", "không bao giờ") không?
5. distractor_length: Distractor có độ dài khác biệt lớn so với các options khác không?
6. k_type_combination: Nếu là multiple_correct, các đáp án đúng có phù hợp kiểu K không?

[INPUT — MCQ cần kiểm tra]
{mcq_str}

[HARD CONSTRAINTS]
- Chỉ đánh giá DISTRACTORS (phương án SAI)
- Đáp án đúng KHÔNG cần kiểm tra IWF
- Đếm tổng số lỗi IWF phát hiện
- Nếu > 3 lỗi → overall_distractor_quality_pass = false

[OUTPUT FORMAT — JSON ONLY]
{{
  "distractor_evaluations": [
    {{
      "option_label": "<A|B|C|D>",
      "option_text": "<text>",
      "plausible_distractor": true|false,
      "vague_terms": "<mô tả nếu có, hoặc 'none'>",
      "grammar_clue": true|false,
      "absolute_terms": "<mô tả nếu có, hoặc 'none'>",
      "distractor_length": "<bình thường / quá dài / quá ngắn>",
      "iwf_count": <0|1|2|...>
    }}
  ],
  "total_iwf_count": <int>,
  "bad_options": ["<label của options có lỗi>"],
  "overall_distractor_quality_pass": true|false
}}
"""


# ==============================================================================
# 3. UTILITY FUNCTIONS
# ==============================================================================

def parse_json_output(raw_text: str) -> dict[str, Any]:
    """Parse JSON from LLM output — handle text before/after JSON block."""
    text = raw_text.strip()
    for match in re.finditer(r'\{', text):
        start = match.start()
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text[:500], "error": "parse_failed"}


def _to_seconds(ts: str | float | None) -> str:
    """Convert MM:SS / HH:MM:SS / plain seconds → integer string for YouTube URL."""
    if ts is None or ts == "":
        return ""
    ts = str(ts).strip()
    if not ts:
        return ""
    if ":" in ts:
        parts = ts.split(":")
        try:
            if len(parts) == 2:
                return str(int(parts[0]) * 60 + int(parts[1]))
            elif len(parts) == 3:
                return str(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]))
            else:
                return ""
        except ValueError:
            return ""
    try:
        return str(int(float(ts)))
    except (ValueError, TypeError):
        return ""


def _format_timestamp(ts: str | float | None) -> str:
    """Format timestamp as MM:SS or HH:MM:SS for human-readable display."""
    if ts is None or ts == "":
        return ""
    ts_str = str(ts).strip()
    if not ts_str:
        return ""
    if ":" in ts_str:
        return ts_str
    # Plain seconds → convert to MM:SS
    try:
        total = int(float(ts_str))
        if total >= 3600:
            hh = total // 3600
            mm = (total % 3600) // 60
            ss = total % 60
            return f"{hh}:{mm:02d}:{ss:02d}"
        else:
            mm = total // 60
            ss = total % 60
            return f"{mm}:{ss:02d}"
    except (ValueError, TypeError):
        return ""


def format_context_block(chunk: dict[str, Any]) -> str:
    """
    Format một concept chunk thành context block cho prompt.

    Citation format (theo tieplm):
      - Slide:  <(Trích dẫn: CS116-Bai04-Data preprocessing.pdf | Trang 3)>
      - Video:   <(Trích dẫn: https://youtu.be/abc&t=90 | 1:30 - 3:45)>
                 Khi bấm link → YouTube nhảy đến đúng giây 90.
    """
    parts = []

    # Header với chapter
    if chunk.get("chapter_id"):
        parts.append(f"[{chunk['chapter_id']}]")

    # ── Nguồn trích dẫn ───────────────────────────────────────────
    source_type = chunk.get("source_type", "")
    if source_type == "slide_pdf":
        src_parts = []
        if chunk.get("source_file"):
            src_parts.append(chunk["source_file"])
        if chunk.get("page_number"):
            src_parts.append(f"Trang {chunk['page_number']}")
        if src_parts:
            parts.append(f"<(Trích dẫn: {', '.join(src_parts)}>)")

    elif source_type == "video_transcript":
        yt_url = chunk.get("youtube_url", "")
        # Ưu tiên timestamp_start (float seconds) > youtube_ts_start (MM:SS string)
        ts_start_raw = chunk.get("timestamp_start")
        ts_end_raw = chunk.get("timestamp_end")
        # Fallback sang string versions
        if ts_start_raw is None:
            ts_start_raw = chunk.get("youtube_ts_start", "")
        if ts_end_raw is None:
            ts_end_raw = chunk.get("youtube_ts_end", "")

        ts_start_sec = _to_seconds(ts_start_raw)
        ts_end_sec = _to_seconds(ts_end_raw)

        if yt_url:
            base_url = yt_url.split("&t=")[0].split("?t=")[0]
            full_url = f"{base_url}&t={ts_start_sec}" if ts_start_sec else yt_url

            ts_display = _format_timestamp(ts_start_raw)
            ts_end_display = _format_timestamp(ts_end_raw)
            if ts_end_display and ts_end_display != ts_display:
                ts_display = f"{ts_display} - {ts_end_display}"
            elif ts_end_sec and ts_end_sec != ts_start_sec:
                ts_display = f"{ts_display} - {ts_end_display}"

            if ts_display:
                parts.append(f"<(Trích dẫn: {full_url} | {ts_display})>")
            else:
                parts.append(f"<(Trích dẫn: {full_url})>")
        elif ts_start_sec:
            parts.append(f"<(Trích dẫn: {_format_timestamp(ts_start_raw)} - {_format_timestamp(ts_end_raw)}>)")

    # Section title và topic
    if chunk.get("section_title"):
        parts.append(f"Tiêu đề: {chunk['section_title']}")
    if chunk.get("topic"):
        parts.append(f"Chủ đề: {chunk['topic']}")

    # Text content
    if chunk.get("text"):
        parts.append(chunk["text"])
    return "\n".join(parts)


def load_topic_list() -> list[dict[str, Any]]:
    """Load topic list từ JSON file."""
    with open(Config.TOPIC_LIST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load all records from a .jsonl file."""
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict[str, Any]], path: Path):
    """Save records to .jsonl file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def make_vllm_sampling(temperature: float, max_tokens: int) -> dict[str, Any]:
    """Tạo sampling params cho vLLM."""
    return {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

#def _get_dynamic_gpu_utilization(model_name: str = "Qwen2.5-14B-Instruct") -> float:
#    try:
#        import subprocess
#        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
#        result = subprocess.run(
#            ["nvidia-smi", "--query-gpu=memory.total,memory.free",
#             "--format=csv,noheader,nounits", "-i", gpu_id],
#            capture_output=True, text=True, timeout=5
#        )
#        if result.returncode == 0:
#            lines = result.stdout.strip().split("\n")
#            if lines:
#                parts = lines[0].split(",")
#                if len(parts) >= 2:
#                    total_mb = int(parts[0].strip())
#                    free_mb = int(parts[1].strip())
#                    # Dùng 70% free VRAM, giữ 30% buffer
#                    util = (free_mb * 0.70) / total_mb
#                    util = round(max(0.50, min(0.70, util)), 2)
#                    print(f"[vLLM GPU] GPU {gpu_id}: {free_mb}/{total_mb} MiB free | "
#                          f"Utilization: {util}")
#                    return util
#    except Exception as e:
#        print(f"[vLLM GPU] Could not query VRAM: {e}")
#    return 0.60

def _get_dynamic_gpu_utilization(model_name: str = "Qwen2.5-14B-Instruct") -> float:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.60
        device = torch.device("cuda:0")
        total = torch.cuda.get_device_properties(device).total_mem / 1024**2  # MiB
        free, _ = torch.cuda.mem_get_info(device)
        free_mb = free / 1024**2
        # Dùng free VRAM trừ 3GB buffer, chia cho total
        util = (free_mb - 3000) / total
        util = round(max(0.50, min(0.90, util)), 2)
        print(f"[vLLM GPU] CUDA device 0: {free_mb:.0f}/{total:.0f} MiB free | "
              f"Utilization: {util}")
        return util
    except Exception as e:
        print(f"[vLLM GPU] Could not query VRAM: {e}")
    return 0.60

def init_vllm_gen() -> Any:
    """Khởi tạo vLLM cho generation model (Qwen2.5-14B-Instruct)."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("❌ vllm not installed. Run: pip install vllm")
        sys.exit(1)
    gpu_util = _get_dynamic_gpu_utilization("Qwen2.5-14B-Instruct")
    llm = LLM(
        model=str(Config.MODEL_GEN),
        tensor_parallel_size=Config.GEN_TP,
        trust_remote_code=True,
        gpu_memory_utilization=0.40,
        max_model_len=8192,
        # enforce_eager=True,            # tắt CUDA graph → tránh OOM khi warm-up
    )
    return llm, SamplingParams


def init_vllm_eval() -> Any:
    """Khởi tạo vLLM cho evaluation model (Gemma-3-12b-it)."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("❌ vllm not installed. Run: pip install vllm")
        sys.exit(1)
    gpu_util = _get_dynamic_gpu_utilization("gemma-3-12b-it")
    llm = LLM(
        model=str(Config.MODEL_EVAL),
        tensor_parallel_size=Config.EVAL_TP,
        trust_remote_code=True,
        gpu_memory_utilization=0.40,
        max_model_len=8192,
        # enforce_eager=True,
    )
    return llm, SamplingParams
