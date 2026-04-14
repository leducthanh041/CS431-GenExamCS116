"""
explain_mcq.py — Explanation Generation for Accepted MCQs
===========================================================
Chạy SAU Step 08 (eval_iwf.py) khi câu hỏi được accept.

Mỗi câu hỏi đã accept sẽ được bổ sung explanation gồm:
  1. correct_answer_rationale — vì sao đáp án đúng là đúng
  2. distractor_explanations — vì sao từng distractor là sai
  3. knowledge_context — câu hỏi nằm trong phạm vi môn học nào,
     nâng cao kiến thức gì
  4. sources — trích dẫn: Slide (số trang) + YouTube bài giảng liên quan

TRÍCH DẪN YOUTUBE:
  - YouTube được trích dẫn từ video/slide mapping trong input/video/videos1.txt
  - Mỗi slide có mapping tới YouTube URL → trích dẫn theo đó
  - KHÔNG dùng external web search
  - Trích dẫn slide KHÔNG trùng lặp

Output:
  output/exp_XX/09_explain/
      explanations.jsonl   ← bản gốn JSONL với explanation
"""

from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

# NOTE: Must add 'src' to path so 'from common' and 'from gen.' work
_pdir = str(Path(__file__).resolve().parent.parent)  # .../CS431MCQGen/src/
if _pdir not in sys.path:
    sys.path.insert(0, _pdir)

from common import Config, config, save_jsonl, load_jsonl, format_context_block
from gen.retrieval_hybrid import HybridRetriever

# ── Override EXP_NAME from environment ────────────────────────────────────────
import os as _os
_exp_name = _os.environ.get("EXP_NAME", "")
if _exp_name:
    Config.EXP_NAME = _exp_name
    Config.OUTPUT_DIR = Config.PROJECT_ROOT / "output" / Config.EXP_NAME
    Config.EVAL_IWF_OUTPUT = Config.OUTPUT_DIR / "08_eval_iwf"
    Config.EXPLAIN_OUTPUT = Config.OUTPUT_DIR / "09_explain"
    print(f"[explain_mcq] EXP_NAME overridden: {Config.EXP_NAME}")


# ── YouTube ↔ Slide mapping from videos1.txt ───────────────────────────────────

_YOUTUBE_MAP: dict[str, str] = {}   # "ch04_CS116-Bai04-Data preprocessing.pdf_p22" → "https://youtu.be/..."


def _load_youtube_map() -> dict[str, str]:
    """
    Parse input/video/videos1.txt to build a mapping from
    chapter + slide_file + page → YouTube URL.

    Format each line:  chapter|yt_url|optional_slide_info
    Lines with "slide: CS116-Bai04-Data preprocessing.pdf, trang 22"
      → map that chapter+file+page → yt_url
    """
    global _YOUTUBE_MAP
    if _YOUTUBE_MAP:
        return _YOUTUBE_MAP

    videos_file = Config.PROJECT_ROOT / "input" / "video" / "videos1.txt"
    if not videos_file.exists():
        print(f"⚠️  videos1.txt not found at {videos_file}")
        return _YOUTUBE_MAP

    chapter_map = {
        "ch01": 1, "ch02": 2, "ch03": 3, "ch04": 4,
        "ch05": 5, "ch06": 6, "ch07": 7, "ch08": 8,
        "ch09": 9, "ch10": 10, "ch11": 11,
    }
    num_to_ch = {v: k for k, v in chapter_map.items()}

    with open(videos_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith('"""'):
                continue
            # Format: chapter|yt_url[, extra_info]
            # Note: URL may contain commas, so split only on first |
            pipe_parts = line.split("|", 1)
            if len(pipe_parts) < 2:
                continue
            try:
                chapter_num = int(pipe_parts[0].strip())
            except ValueError:
                continue

            # yt_url is everything before the first comma after the pipe
            # (URL may contain commas like "https://youtu.be/eY--6WX1hAQ, None")
            rest = pipe_parts[1]
            first_comma = rest.find(",")
            if first_comma >= 0:
                yt_url = rest[:first_comma].strip()
                extra  = rest[first_comma + 1:].strip()
            else:
                yt_url = rest.strip()
                extra  = ""

            # Skip lines without slide info (coding tutorials, etc.)
            if not extra or "trang" not in extra.lower():
                continue

            # Extract page number: "trang N"
            page_num: int | None = None
            m = re.search(r"trang\s+(\d+)", extra, re.IGNORECASE)
            if m:
                page_num = int(m.group(1))

            # Extract slide file name: "CS116-Bai04-Data preprocessing.pdf"
            slide_file: str | None = None
            sm = re.search(r"(CS116-Bai\d+.+\.pdf)", extra)
            if sm:
                slide_file = sm.group(1).strip()

            chapter_id = num_to_ch.get(chapter_num, f"ch{chapter_num:02d}")

            if page_num is not None and slide_file:
                # Chapter 8 in videos1.txt = BOTH ch08 AND ch07b (Bai07b slides)
                # Chapter 7 in videos1.txt = BOTH ch07 AND ch07a (Bai07a slides)
                chapter_ids = [chapter_id]
                if chapter_num == 8:
                    chapter_ids.extend(["ch07b", "ch07"])
                elif chapter_num == 7:
                    chapter_ids.extend(["ch07a"])

                for cid in chapter_ids:
                    # Build all key variants: with/without space in " learning"
                    for sf in [slide_file, slide_file.replace(" ", "")]:
                        sm2 = re.search(r"(CS116-Bai\d+.+\.pdf)", sf)
                        if sm2:
                            bare = sm2.group(1)
                            keys_to_store = [
                                f"{cid}_{bare}_p{page_num}",
                                f"{cid}_{bare.replace(' ', '')}_p{page_num}",
                                f"{bare}_p{page_num}",
                                f"{bare.replace(' ', '')}_p{page_num}",
                            ]
                            for key in keys_to_store:
                                _YOUTUBE_MAP[key] = yt_url

    print(f"  📺 YouTube map loaded: {len(_YOUTUBE_MAP)} slide→video entries")
    return _YOUTUBE_MAP


def _chunk_to_map_key(chunk: dict) -> tuple[str, int] | None:
    """
    Extract (slide_file, page_number) from a slide_pdf chunk,
    normalized for YouTube map lookup.
    Returns (normalized_file, page) or None.
    """
    slide_file = chunk.get("source_file", "") or chunk.get("slide_file", "")
    page_raw   = chunk.get("page_number", 0) or chunk.get("slide_start_page", 0)
    if not slide_file or not page_raw:
        return None
    try:
        page = int(page_raw)
    except (ValueError, TypeError):
        return None
    # Normalize: extract bare CS116-BaiXX filename (handles mangled Unicode like v#U1edbi)
    m = re.search(r"(CS116-Bai\d+.+\.pdf)", slide_file)
    if m:
        return (m.group(1).strip(), page)
    return (slide_file.strip(), page)


def get_youtube_for_chunk(
    chunk: dict,
    primary_chapter: str = "",
) -> str:
    """
    Get YouTube URL for a slide_pdf chunk by looking it up in videos1.txt.

    Keys tried (in order):
      1. <primary_chapter>_<file>_p<page>   ← preferred (same chapter as question)
      2. <chapter_id>_<file>_p<page>        ← fallback to chunk's own chapter
      3. <file>_p<page>                     ← bare (no chapter prefix)
      4. ch07_<file>_p<page>  /  ch07a_<file>_p<page>  (sibling chapters)

    primary_chapter: the chapter_id from the question topic (e.g. "ch07a")
      — used to prefer the question's own chapter's video over siblings.
    """
    _load_youtube_map()

    if chunk.get("source_type") != "slide_pdf":
        return ""

    key_data = _chunk_to_map_key(chunk)
    if not key_data:
        return ""
    slide_file, page = key_data

    # Sibling chapter variants (e.g. ch07 ↔ ch07a)
    def _siblings(cid: str) -> list[str]:
        if cid in ("ch07a", "ch07b"):
            return ["ch07a", "ch07b", "ch07"]
        return [cid]

    def _try_keys(prefixes: list[str]) -> str:
        for prefix in prefixes:
            for variant in (f"{prefix}_{slide_file}_p{page}",
                             f"{prefix}_{slide_file.replace(' ', '')}_p{page}",
                             f"{slide_file}_p{page}",
                             f"{slide_file.replace(' ', '')}_p{page}"):
                if variant in _YOUTUBE_MAP:
                    return _YOUTUBE_MAP[variant]
        return ""

    # Priority 1: primary chapter (question's own chapter)
    if primary_chapter:
        result = _try_keys([primary_chapter])
        if result:
            return result

    # Priority 2: chunk's own chapter (including sibling expansion)
    chunk_ch = chunk.get("chapter_id", "")
    result = _try_keys(_siblings(chunk_ch))
    if result:
        return result

    # Priority 3: bare lookup (no chapter prefix)
    return _try_keys([""])


# ── Format citations (deduplicated, with YouTube per slide) ────────────────────


def _to_seconds(ts: str | float | int | None) -> int:
    """Convert '1:30' or float seconds → integer."""
    if ts is None or ts == "":
        return 0
    try:
        return int(float(str(ts)))
    except (ValueError, TypeError):
        return 0


def _format_ts(seconds: int) -> str:
    """Convert integer seconds → MM:SS or HH:MM:SS."""
    if seconds >= 3600:
        hh = seconds // 3600
        mm = (seconds % 3600) // 60
        ss = seconds % 60
        return f"{hh}:{mm:02d}:{ss:02d}"
    else:
        mm = seconds // 60
        ss = seconds % 60
        return f"{mm}:{ss:02d}"


def build_source_citations(
    context_blocks: list[dict],
    max_slides: int = 4,
    primary_chapter: str = "",
) -> list[dict]:
    """
    Build citation list from context blocks — QUANTITY-CONTROLLED.

    Rules:
    - Slides: top-ranked blocks (max max_slides), primary-chapter slides first,
      then cross-chapter if needed. Deduplicated by (file, page).
      Earlier in context_blocks = higher relevance from hybrid retriever.
    - Video: get YouTube URL from videos1.txt mapping of the top-ranked slide.
      If NO slides exist in blocks (all video_transcript → cross-encoder dominates),
      use primary_chapter to look up the chapter's video from videos1.txt directly.

    Returns: [{type, url, file, page, description, timestamp}]
    """
    CHAPTER_NAMES = {
        # Chapter number → display name (matches videos1.txt chapter numbers)
        1: "Thư viện phổ biến", 2: "Pipeline & EDA",
        3: "Tiền xử lý dữ liệu", 4: "Đánh giá mô hình",
        5: "Unsupervised Learning",
        6: "Supervised Learning — Regression",
        7: "Supervised Learning — Classification",  # videos1 chapter 7 = Bai07a Regression
        8: "Deep Learning với CNN",                  # videos1 chapter 9 = Bai08 = ch08
        9: "Parameter Tuning",
        10: "Ensemble Models", 11: "Model Deployment",
        # chapter_id → chapter NUMBER (for chapter_title lookups)
        "ch01": 1, "ch02": 2, "ch03": 3, "ch04": 4,
        "ch05": 5, "ch06": 6, "ch07": 6,
        "ch07a": 6,  # Bai07a → chapter 6 = Supervised Regression
        "ch07b": 7,  # Bai07b → chapter 7 = Supervised Classification
        "ch08": 8,   # Bai08 → chapter 8 = Deep Learning CNN
        "ch09": 9, "ch10": 10, "ch11": 11,
    }

    def chapter_title(cid: str) -> str:
        n = CHAPTER_NAMES.get(cid, 0)
        return CHAPTER_NAMES.get(n, cid)

    _PARENT_MAP_VIDEO = {"ch07a": "ch07", "ch07b": "ch07", "ch08": "ch09"}

    # ── Phase 1: do NOT override primary_chapter (it IS the question's chapter) ─
    # Only use slide chapters to sort/prioritise slides within the citation list.
    # primary_chapter param is the QUESTION's chapter — the single source of truth.

    # ── Phase 2: collect slides (primary chapter first, max max_slides) ────
    seen_slides: set[tuple[str, str]] = set()
    primary_slides: list[dict] = []
    cross_slides: list[dict] = []

    for blk in context_blocks:
        if blk.get("source_type") != "slide_pdf":
            continue

        slide_file = blk.get("source_file", "") or blk.get("slide_file", "")
        page_raw   = blk.get("page_number", "") or blk.get("slide_start_page", "")
        page       = str(page_raw) if page_raw else ""
        section    = blk.get("section_title", "")

        if not slide_file or not page or page == "0":
            continue

        key = (slide_file, page)
        if key in seen_slides:
            continue
        seen_slides.add(key)

        label = f"📄 {slide_file}"
        if page:
            label += f", Trang {page}"
        if section:
            label += f" — {section[:35]}"

        entry = {
            "type":        "slide",
            "file":        slide_file,
            "page":        page,
            "chapter_id":  blk.get("chapter_id", ""),
            "description": label,
            "url":         "",
        }

        if blk.get("chapter_id") == primary_chapter:
            primary_slides.append(entry)
        else:
            cross_slides.append(entry)

    # Primary slides first, then cross-chapter, capped at max_slides
    all_slides = primary_slides[:max_slides]
    if len(all_slides) < max_slides:
        for s in cross_slides:
            if len(all_slides) >= max_slides:
                break
            all_slides.append(s)

    # ── Phase 3: get YouTube from the top slide's chapter mapping ────────────
    best_video: dict | None = None

    def _find_timestamp(pc: str) -> tuple[int, int]:
        """Find timestamp from video_transcript block of the given chapter."""
        # video_transcript blocks use parent chapter (ch07 not ch07a/ch07b)
        _PARENT_MAP = {"ch07a": "ch07", "ch07b": "ch07", "ch08": "ch09"}
        candidates = [pc, _PARENT_MAP.get(pc, pc)]
        for blk in context_blocks:
            if (blk.get("source_type") == "video_transcript" and
                    blk.get("chapter_id") in candidates):
                raw = blk.get("timestamp_start", "")
                start = _to_seconds(raw) if raw else 0
                if start > 0:
                    end_raw = blk.get("timestamp_end", "")
                    end = _to_seconds(end_raw) if end_raw else 0
                    return start, end
        return 0, 0

    if all_slides:
        # ── Path A: look up video from videos1.txt using slides from blocks ───
        _load_youtube_map()
        _PARENT_MAP = {"ch07a": "ch07", "ch07b": "ch07", "ch08": "ch09"}

        # ── PRIORITY-BASED VIDEO SELECTION ──────────────────────────────────────
        # Priority: 1) question chapter slides → 2) parent chapter slides → 3) any slides
        # This ensures the video always matches the QUESTION's chapter, not the retrieval order.
        _load_youtube_map()

        # Build ordered slide list by chapter priority
        def _slide_priority(slide_entry: dict) -> int:
            ch = slide_entry.get("chapter_id", "")
            if ch == primary_chapter:
                return 0   # question's own chapter → highest priority
            parent = _PARENT_MAP_VIDEO.get(ch, ch)
            if parent == primary_chapter or _PARENT_MAP_VIDEO.get(primary_chapter, primary_chapter) == ch:
                return 1   # parent/sibling chapter → second priority
            return 2      # cross-chapter → last resort

        sorted_slides = sorted(all_slides, key=_slide_priority)

        # Try each slide in priority order; for each, try exact page then nearby pages
        for slide_entry in sorted_slides:
            yt_url = get_youtube_for_chunk(slide_entry, primary_chapter=primary_chapter)
            if not yt_url:
                # Slide's exact page not in videos1.txt → try nearby pages in same chapter
                slide_file = slide_entry["file"]
                slide_ch   = slide_entry["chapter_id"]
                slide_page = int(slide_entry["page"])

                # Pages to try: same page, then ±1, ±2, ±3
                pages_to_try = [slide_page]
                pages_to_try.extend([slide_page + d for d in range(1, 5)])
                pages_to_try.extend([slide_page - d for d in range(1, 5)])
                pages_to_try = [p for p in pages_to_try if p >= 1]

                best_url_for_slide: str = ""
                best_dist: int = 9999

                bai_m = re.search(r"(CS116-Bai\d+)", slide_file)
                if bai_m:
                    bai_pattern = bai_m.group(1)
                    for key in _YOUTUBE_MAP:
                        if bai_pattern in key:
                            km = re.search(r"_p(\d+)$", key)
                            if km:
                                key_page = int(km.group(1))
                                dist = abs(key_page - slide_page)
                                if dist < best_dist:
                                    best_dist = dist
                                    best_url_for_slide = _YOUTUBE_MAP[key]

                if best_url_for_slide:
                    yt_url = best_url_for_slide

            if yt_url:
                # Find timestamp from video_transcript block matching the slide's chapter
                slide_ch = slide_entry["chapter_id"]
                # Include parent in candidates so ch07a slides get timestamp from ch07 video
                candidates = [slide_ch, _PARENT_MAP_VIDEO.get(slide_ch, slide_ch)]
                ts_start, ts_end = 0, 0
                for blk in context_blocks:
                    if (blk.get("source_type") == "video_transcript" and
                            blk.get("chapter_id") in candidates):
                        raw = blk.get("timestamp_start", "")
                        ts_start = _to_seconds(raw) if raw else 0
                        if ts_start > 0:
                            end_raw = blk.get("timestamp_end", "")
                            ts_end = _to_seconds(end_raw) if end_raw else 0
                            break

                base_url  = yt_url.split("&t=")[0].split("?t=")[0]
                ts_disp   = _format_ts(ts_start) if ts_start else ""
                ts_end_d  = _format_ts(ts_end)   if ts_end   else ""

                ts_label = f" [{ts_disp}" if ts_disp else " ["
                if ts_end_d and ts_end_d != ts_disp:
                    ts_label += f" → {ts_end_d}"
                ts_label += "]"

                best_video = {
                    "type":          "video",
                    "url":           base_url,
                    "description":   f"▶️ Video bài giảng: {chapter_title(primary_chapter)}{ts_label}",
                    "chapter_id":    primary_chapter,
                    "timestamp":     ts_disp,
                    "timestamp_end": ts_end_d,
                }
                break   # found a video

    else:
        # ── Path B: no slides in blocks (cross-encoder dominates) ─────────────
        # → Fall back: use primary_chapter to look up the chapter's first video
        _load_youtube_map()
        if primary_chapter:
            # Try: primary_chapter + first slide file for that chapter
            first_slide = next(
                (blk for blk in context_blocks
                 if blk.get("source_type") == "slide_pdf"),
                None,
            )
            if first_slide:
                yt_url = get_youtube_for_chunk(
                    {
                        "source_type": "slide_pdf",
                        "source_file": first_slide.get("source_file", ""),
                        "chapter_id":  primary_chapter,
                        "page_number": (first_slide.get("page_number", "") or
                                        first_slide.get("slide_start_page", "")),
                    },
                    primary_chapter=primary_chapter,
                )
            else:
                yt_url = ""

            if not yt_url:
                # Last resort: look up any page-1 video for primary chapter
                # Try chXX_BaiXX*.pdf as file pattern for the chapter
                chapter_num_m = re.match(r"ch0?(\d+)", primary_chapter)
                if chapter_num_m:
                    cn = chapter_num_m.group(1)
                    # Find slide→video mapping for page 1 of this chapter
                    candidates = [
                        k for k in _YOUTUBE_MAP
                        if f"Bai{cn}" in k and "_p1" in k and primary_chapter in k
                    ]
                    if candidates:
                        yt_url = _YOUTUBE_MAP[candidates[0]]

            if yt_url:
                ts_start, ts_end = _find_timestamp(primary_chapter)
                ts_disp  = _format_ts(ts_start) if ts_start else ""
                ts_end_d = _format_ts(ts_end)   if ts_end   else ""

                ts_label = f" [{ts_disp}" if ts_disp else " ["
                if ts_end_d and ts_end_d != ts_disp:
                    ts_label += f" → {ts_end_d}"
                ts_label += "]"

                best_video = {
                    "type":          "video",
                    "url":           yt_url,
                    "description":   f"▶️ Video bài giảng: {chapter_title(primary_chapter)}{ts_label}",
                    "chapter_id":    primary_chapter,
                    "timestamp":     ts_disp,
                    "timestamp_end": ts_end_d,
                }

    # ── Phase 4: build final citation list ───────────────────────────────────
    citations: list[dict] = []
    citations.extend(all_slides)     # slides first
    if best_video:
        citations.append(best_video)  # then the one best video

    return citations


# ── Explanation prompt ───────────────────────────────────────────────────────


EXPLAIN_SYSTEM_PROMPT = (
    "Bạn là giảng viên đại học chuyên giải thích câu hỏi trắc nghiệm cho sinh viên. "
    "Nhiệm vụ: tạo explanation rõ ràng, có trích dẫn, giúp sinh viên HIỂU "
    "vì sao đáp án đúng và vì sao các đáp án sai KHÔNG đúng. "
    "Không dùng markup phức tạp — chỉ plain text tiếng Việt."
)


EXPLAIN_USER_PROMPT_TEMPLATE = """[ROLE]
Bạn là giảng viên đại học giỏi giải thích câu hỏi trắc nghiệm.
Mỗi câu hỏi bạn giải thích cần giúp sinh viên HIỂU
vì sao đáp án đúng đúng và vì sao các đáp án sai không đúng.

[NHIỆM VỤ]
Hãy giải thích câu hỏi MCQ dưới đây. Viết 4 phần:
  1. TẠI SAO RA CÂU HỎI NÀY — câu hỏi này kiểm tra khái niệm gì, vì sao SV cần biết, tại sao lại hỏi theo cách này
  2. Tại sao đáp án đúng là đúng — kèm TRÍCH DẪN cụ thể (video YouTube bài giảng, slide số trang)
  3. Tại sao từng distractor là sai — kèm TRÍCH DẪN + giải thích confusion point
  4. Knowledge context: phạm vi môn học, nâng cao kiến thức gì, prerequisites

[CÂU HỎI]
{question_text}

[OPTIONS]
{options_str}

[ĐÁP ÁN ĐÚNG: {correct_letters}]
{correct_answers_str}

[KHOÁI KIẾN THỨC TỪ COURSE MATERIAL — CÓ TRÍCH DẪN VIDEO/SLIDE]
{course_context}

[YÊU CẦU — BẮT BUỘC CÓ TRÍCH DẪN]
- **Phần 1 (đáp án đúng):** Giải thích 2-4 câu, SAU MỖI câu giải thích phải ghi trích dẫn:
    VD: "Gradient descent cập nhật trọng số theo hướng giảm gradient, được giải thích trong slide CS116-Bai07a, trang 5."
- **Phần 2 (distractors):** Mỗi distractor 1-2 câu, ghi rõ VÌ SAO sai + confusion point + trích dẫn.
    VD: "Đáp án B sai vì nó mô tả hàm activation ReLU, không phải sigmoid. Xem slide CS116-Bai08, trang 3."
- **Phần 3 (knowledge context):** 2-3 câu, nêu phạm vi + prerequisites + advanced knowledge.
- **TRÍCH DẪN BẮT BUỘC:** Ghi rõ tên slide + số trang.
  Nếu có video YouTube liên quan → ghi thêm URL video.
- VIẾT BẰNG TIẾNG VIỆT, rõ ràng, có trích dẫn rõ ràng sau mỗi ý.
- KHÔNG trích dẫn nguồn internet bên ngoài (web search).

[OUTPUT FORMAT — JSON ONLY]
{{
  "question_motivation": "<2-3 câu: TẠI SAO ra câu hỏi này, câu hỏi kiểm tra khái niệm gì, vì sao SV cần nắm vững, tại sao hỏi theo cách này (đặc biệt nếu là câu hỏi sai/thiên lệch cần tránh)>",
  "correct_answer_rationale": "<2-4 câu giải thích TẠI SAO đúng, kèm trích dẫn cụ thể (tên slide + số trang, video YouTube URL)>",
  "distractor_explanations": {{
    "<letter>": "<1-2 câu: vì sao sai + confusion point + trích dẫn cụ thể (slide/trang, video URL)>",
    "<letter>": "...",
    "<letter>": "..."
  }},
  "knowledge_context": {{
    "topic_scope": "<mô tả phạm vi topic>",
    "prerequisites": ["<khái niệm 1>", "<khái niệm 2>"],
    "advanced_knowledge": "<kiến thức nâng cao liên quan>",
    "learning_value": "<câu hỏi này giúp sinh viên hiểu được gì>"
  }},
  "sources_used": [
    {{
      "type": "slide|video",
      "url": "<YouTube URL hoặc ''>",
      "file": "<slide file name nếu là slide, hoặc ''>",
      "page": "<số trang nếu là slide, hoặc ''>",
      "description": "<mô tả ngắn>"
    }}
  ]
}}
"""


# ── Generate explanation for a single MCQ ───────────────────────────────────


def generate_explanation_for_mcq(
    mcq: dict[str, Any],
    context_blocks: list[dict],
    llm,            # vLLM LLM instance
    SamplingParams,
    config: dict[str, Any],
    primary_chapter: str = "",
) -> dict[str, Any]:
    """
    Generate explanation for a single accepted MCQ.
    - NO external web search
    - Slide citations deduplicated
    - YouTube lecture video cited per slide (from videos1.txt mapping)
    """
    question_text   = mcq.get("question_text", "")
    options         = mcq.get("options", {})
    correct_letters = mcq.get("correct_answers", [])
    correct_content = mcq.get("correct_answers_content", {})

    # ── Build options string ────────────────────────────────────────────────
    letters = ["A", "B", "C", "D"]
    opt_lines = []
    for letter in letters:
        text = options.get(letter, "")
        if text:
            opt_lines.append(f"  {letter}. {text}")
    options_str = "\n".join(opt_lines)

    # ── Correct answers string ──────────────────────────────────────────────
    if isinstance(correct_content, dict):
        correct_ans_lines = []
        for letter in correct_letters:
            text = correct_content.get(letter, options.get(letter, ""))
            correct_ans_lines.append(f"  {letter}. {text}")
        correct_answers_str = "\n".join(correct_ans_lines)
    else:
        correct_answers_str = "\n".join(f"  {c}" for c in (correct_content or []))

    # ── Build course context string ──────────────────────────────────────────
    if context_blocks:
        ctx_lines = []
        for b in context_blocks[:5]:  # top 5 blocks for richer context
            ctx_lines.append(format_context_block(b)[:600])
        course_context = "\n\n".join(ctx_lines)
    else:
        course_context = "(không có course material — dùng kiến thức của bạn)"

    # ── Build prompt (NO external web sources) ───────────────────────────────
    prompt = EXPLAIN_USER_PROMPT_TEMPLATE.format(
        question_text=question_text,
        options_str=options_str,
        correct_letters=", ".join(correct_letters),
        correct_answers_str=correct_answers_str,
        course_context=course_context,
    )

    # ── Call LLM ───────────────────────────────────────────────────────────
    messages = [
        {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    sp = SamplingParams(
        max_tokens=1200,
        temperature=0.3,
        stop=None,
    )
    outputs = llm.chat(messages, sampling_params=sp)
    raw = outputs[0].outputs[0].text

    # ── Parse JSON ───────────────────────────────────────────────────────────
    parsed = _parse_json_output(raw)
    if "error" in parsed:
        print(f"  ⚠️  Explanation parse error: {parsed['error']}")
        return {"error": parsed["error"]}

    # ── Build sources (max 4 slides + 1 video from top slide's chapter) ─────
    citations = build_source_citations(
        context_blocks,
        max_slides=4,
        primary_chapter=primary_chapter,
    )
    parsed["sources"] = citations
    return parsed


def _parse_json_output(raw_text: str) -> dict[str, Any]:
    """Parse JSON from LLM output."""
    text = raw_text.strip()
    for match in re.finditer(r'\{', text):
        try:
            return json.loads(text[match.start():])
        except json.JSONDecodeError:
            continue
    return {"error": f"Cannot parse JSON from: {text[:200]}"}


# ── Entry point ───────────────────────────────────────────────────────────────

def run_explain_mcq(
    accepted_jsonl: Path | None = None,
    output_dir: Path | None = None,
    config: dict[str, Any] | None = None,
):
    """
    Generate explanations for all accepted MCQs.
    """
    if config is None:
        from gen.prompt_config import load_generation_config
        config = load_generation_config()

    if accepted_jsonl is None:
        accepted_jsonl = Config.EVAL_IWF_OUTPUT / "final_accepted_questions.jsonl"
    accepted_jsonl = Path(accepted_jsonl)

    if output_dir is None:
        output_dir = Config.OUTPUT_DIR / "09_explain"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not accepted_jsonl.exists():
        print(f"⚠️  Accepted questions not found: {accepted_jsonl}")
        return

    # ── Load questions ───────────────────────────────────────────────────────
    questions = load_jsonl(accepted_jsonl)
    print(f"📋 Loading {len(questions)} accepted questions from {accepted_jsonl}")

    # ── Init hybrid retriever ────────────────────────────────────────────────
    print("🔄 Initializing hybrid retriever for context...")
    hr = HybridRetriever()
    hr.build_bm25_index()

    # ── Load vLLM ───────────────────────────────────────────────────────────
    print("🔄 Loading vLLM for explanation generation...")
    from common import init_vllm_gen
    llm, SamplingParams = init_vllm_gen()
    print("✅ vLLM loaded")

    # ── Pre-load YouTube map ───────────────────────────────────────────────
    _load_youtube_map()

    results = []
    for i, q in enumerate(questions, 1):
        topic   = q.get("topic", "")
        meta    = q.get("_meta", {})
        topic_id = meta.get("topic_id", "")
        difficulty = q.get("difficulty_label", "")

        print(f"\n  [{i}/{len(questions)}] Generating explanation for: {topic_id}")

        # ── Hybrid retrieval: NO chapter filter to preserve slide diversity ──
        # Cross-encoder biases toward transcript blocks; we compensate by
        # retrieving enough blocks (top_k=15) so slides remain in the top list.
        primary_chapter_raw = topic_id.split("_")[0]   # e.g. "ch07a", "ch07b", "ch04"
        # Normalise to parent chapter for ChromaDB filter: ch07a→ch07, ch07b→ch07
        primary_chapter_for_filter = re.sub(r'^(ch\d+)[ab]$', r'\1', primary_chapter_raw)

        blocks = hr.retrieve(
            query=topic,
            top_k=15,
            chapter_filter=None,   # removed chapter filter — slides must appear
        )

        # If too few slides, supplement with chapter-filtered retrieval
        # (chapter filter is needed because cross-encoder dominates for dense topics)
        slide_blocks = [b for b in blocks if b.get("source_type") == "slide_pdf"]
        if len(slide_blocks) < 2:
            chapter_blocks = hr.retrieve(
                query=topic,
                top_k=20,
                chapter_filter=[primary_chapter_raw],
            )
            # Merge chapter slides that aren't already present
            existing_ids = {b["chunk_id"] for b in blocks}
            for cb in chapter_blocks:
                if cb.get("source_type") == "slide_pdf" and cb["chunk_id"] not in existing_ids:
                    blocks.append(cb)

        # Generate explanation (blocks = slides + video transcripts for rich citations)
        try:
            explanation = generate_explanation_for_mcq(
                q, blocks, llm, SamplingParams, config,
                primary_chapter=primary_chapter_raw,
            )
        except Exception as e:
            print(f"  ⚠️  Explanation error: {e}")
            traceback.print_exc()
            explanation = {"error": str(e)}

        # Merge
        explained = dict(q)
        explained["explanation"] = explanation
        results.append(explained)

        # Progress indicator
        if explanation.get("error"):
            print(f"  ❌ Failed: {explanation['error']}")
        else:
            n_slides = sum(1 for s in explanation.get("sources", []) if s.get("type") == "slide")
            n_videos = sum(1 for s in explanation.get("sources", []) if s.get("type") == "video")
            print(f"  ✅ Done (slides={n_slides}, videos={n_videos})")

    # ── Save ────────────────────────────────────────────────────────────────
    out_jsonl = output_dir / "explanations.jsonl"
    save_jsonl(results, out_jsonl)
    print(f"\n✅ Explanations saved: {out_jsonl}")

    # ── Cleanup VRAM ─────────────────────────────────────────────────────────
    import gc, torch
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("  [cleanup] VRAM released")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate explanations for accepted MCQs")
    parser.add_argument("--input", default=None, help="Path to final_accepted_questions.jsonl")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--config", default=None, help="Path to generation_config.yaml")
    args = parser.parse_args()

    config = None
    if args.config:
        from gen.prompt_config import load_generation_config
        config = load_generation_config(args.config)

    accepted = Path(args.input) if args.input else None
    output   = Path(args.output_dir) if args.output_dir else None

    run_explain_mcq(accepted_jsonl=accepted, output_dir=output, config=config)


if __name__ == "__main__":
    main()
