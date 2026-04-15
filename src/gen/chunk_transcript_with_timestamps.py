"""
chunk_transcript_with_timestamps.py — Step 01b: Chunk JSON Transcripts with YouTube Timestamps

Input:
  - input/transcribe_data/*.json       (Whisper JSON: segments with word-level timestamps)
  - input/video/videos1.txt           (line-indexed YouTube URL mapping)

Output:
  - data/processed/transcript_chunks_with_timestamps.jsonl
    Mỗi chunk có: chunk_id, text, timestamp_start, timestamp_end, youtube_url,
    chapter_id, source_file, source_type="video_transcript", topics, text_clean

Logic:
  1. Parse videos1.txt → sequential line_number → youtube_url mapping
  2. Parse JSON transcript → flatten word-level timestamps per segment
  3. Deduplicate: loại bỏ đoạn lặp do lỗi nối file ASR
  4. Chunk by words (target=200, min=80, overlap=30) — gom câu nguyên vẹn
  5. Map timestamp → youtube_url
  6. Save JSONL
"""

from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import Config, save_jsonl


# ─── Mapping video → YouTube URL ────────────────────────────────────────────
# videos1.txt: line_N = sequential video index (1-based)
# MP4 naming convention: X.Y.mp4 → video with chapter=X, sub_index=Y
#   → maps to videos1.txt line number computed by cumulative count

# (chapter_id, sub_index) → {url, slide_file, slide_start_page}
# e.g. ("ch04", 1) → {"url": "https://youtu.be/...", "slide_file": "CS116-Bai04-Data preprocessing.pdf", "slide_start_page": 1}
VIDEO_META_MAP: dict[tuple[str, int], dict] = {}


def _parse_slide_metadata(raw: str) -> tuple[str, int]:
    """
    Parse slide metadata from the optional ", slide: filename.pdf, trang N" suffix.

    Returns (slide_file, slide_start_page):
      - "None"           → ("", 0)
      - "slide: foo.pdf, trang 1" → ("foo.pdf", 1)
      - "slide: foo.pdf, trang 10" → ("foo.pdf", 10)
    """
    raw = raw.strip()
    if not raw or raw.lower() == "none":
        return "", 0

    import re
    # Match: "slide: filename, trang N" or just "slide: filename"
    slide_match = re.search(r"slide:\s*([^,]+)", raw, re.IGNORECASE)
    page_match  = re.search(r"trang\s+(\d+)", raw, re.IGNORECASE)

    slide_file    = slide_match.group(1).strip() if slide_match else ""
    slide_start_page = int(page_match.group(1)) if page_match else 0
    return slide_file, slide_start_page


def _build_video_url_map() -> None:
    """
    Build VIDEO_META_MAP: (chapter_id, sub_index) → {url, slide_file, slide_start_page}
    from videos1.txt.

    videos1.txt format per line:
      chapter_number|YouTube_URL[, extra...]
    where extra can be:
      - "None"
      - "slide: CS116-Bai02-Popular Libs.pdf, trang 1"
      - "slide: CS116-Bai02-Popular Libs.pdf, trang 4"
      - "None, Coding Tutorials"

    We assign sequential sub_index per chapter based on line order.
    """
    videos1 = Config.INPUT_DIR / "video" / "videos1.txt"
    if not videos1.exists():
        print(f"⚠️  {videos1} not found — youtube_url will be empty")
        return

    lines = videos1.read_text(encoding="utf-8").strip().split("\n")

    current_chapter: int | None = None
    sub_index: int = 0

    for line in lines:
        if "|" not in line:
            continue
        parts = line.split("|")
        url = parts[1].strip().split(",")[0].strip()
        # Everything after URL is the optional metadata
        extra = parts[2].strip() if len(parts) > 2 else ""
        slide_file, slide_start_page = _parse_slide_metadata(extra)

        try:
            ch_int = int(parts[0].strip())
        except ValueError:
            continue

        if ch_int != current_chapter:
            current_chapter = ch_int
            sub_index = 1
        else:
            sub_index += 1

        chapter_id = f"ch{ch_int:02d}"
        VIDEO_META_MAP[(chapter_id, sub_index)] = {
            "url":              url,
            "slide_file":       slide_file,
            "slide_start_page": slide_start_page,
        }
        # Verbose log for non-None slide entries
        if slide_file:
            print(f"  [video_map] {chapter_id} sub={sub_index}: {url} → slide={slide_file} page={slide_start_page}")


# ─── Deduplication ────────────────────────────────────────────────────────────

def _deduplicate_text(text: str) -> str:
    """
    Remove repeated phrases that Whisper produces due to ASR concatenation.
    Matches 2+ consecutive identical word groups (2-100 chars).
    E.g. "đây đây đây" → "đây", "nội dung nội dung" → "nội dung"
    NOTE: Sentence-level dedup was removed (too slow on long texts).
    """
    if not text:
        return text
    # Remove repeated word groups — fast linear pass
    text = re.sub(r"\b(.{2,100}?)(?:\s+\1\b)+", r"\1", text)
    return text.strip()


# ─── Sentence boundary detection ─────────────────────────────────────────────

# Simple Vietnamese sentence splitter using punctuation
_SENTENCE_DELIMITERS = re.compile(r"(?<=[.?!])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on . ? ! boundaries."""
    sents = _SENTENCE_DELIMITERS.split(text)
    return [s.strip() for s in sents if s.strip()]


# ─── Core chunking ────────────────────────────────────────────────────────────

CHUNK_CONFIG = {
    "target_words": 200,
    "min_words": 80,
    "overlap_words": 30,
}


def _chunk_segments(
    all_words: list[dict],  # [{word, start, end}]
    chapter_id: str,
    video_sub: str,
    source_file: str,
    topics: list[str],
    chapter_title: str,
) -> list[dict]:
    """
    Chunk words into groups of ~200 words, respecting sentence boundaries.
    Each chunk spans [timestamp_start, timestamp_end] in seconds.

    Returns list of chunk dicts ready for JSONL + ChromaDB.
    """
    target = CHUNK_CONFIG["target_words"]
    min_w = CHUNK_CONFIG["min_words"]
    overlap = CHUNK_CONFIG["overlap_words"]

    chunks = []
    n = len(all_words)
    pos = 0
    seq = 1

    while pos < n:
        # Gather words until we hit target size or run out
        end_pos = min(pos + target, n)
        start_ts = all_words[pos]["start"]
        end_ts = all_words[end_pos - 1]["end"]

        # Try to stop at a sentence boundary (within ±20 words tolerance)
        best_cut = end_pos
        if end_pos < n:
            candidates = []
            for i in range(pos, min(pos + target + 20, n)):
                w = all_words[i]["word"].strip()
                if w and w[-1] in ".?!":
                    dist = i - pos
                    candidates.append((abs(dist - target), dist, i + 1))
            if candidates:
                candidates.sort()
                _, _, best_cut = candidates[0]
                best_cut = min(best_cut, n)

        # Build text from pos to best_cut
        chunk_words = all_words[pos:best_cut]
        raw_text = " ".join(w["word"].strip() for w in chunk_words)
        clean_text = _deduplicate_text(raw_text)

        # Fallback: if cleaned text is too short, use original
        if len(clean_text.split()) < min_w and raw_text.strip():
            clean_text = raw_text.strip()

        if clean_text and len(clean_text.split()) >= min_w:
            video_meta = VIDEO_META_MAP.get((chapter_id, int(video_sub)), {})
            youtube_url = video_meta.get("url", "")
            slide_file       = video_meta.get("slide_file", "")
            slide_start_page = video_meta.get("slide_start_page", 0)

            chunk_id = f"cs116_{chapter_id}_transcript_{video_sub}_s{seq:03d}"
            chunk = {
                "chunk_id": chunk_id,
                "course_id": "CS116",
                "chapter_id": chapter_id,
                "chapter_title": chapter_title,
                "topics": topics,
                "source_type": "video_transcript",
                "source_file": source_file,
                "page_number": int(video_sub),
                "section_title": "",
                "text": clean_text,
                "timestamp_start": round(start_ts, 3),
                "timestamp_end": round(end_ts, 3),
                "youtube_url": youtube_url,
                "youtube_timestamp_start": _format_youtube_ts(start_ts),
                "youtube_timestamp_end": _format_youtube_ts(end_ts),
                "slide_file": slide_file,           # e.g. "CS116-Bai04-Data preprocessing.pdf"
                "slide_start_page": slide_start_page,  # e.g. 1 or 7 or 34...
                "word_count": len(clean_text.split()),
                "embedding_ready": True,
            }
            chunks.append(chunk)
            seq += 1

        # Slide with overlap
        advance = target - overlap
        pos += advance
        if pos >= end_pos - 1:
            pos = end_pos  # Prevent infinite loop on tiny sentences

    return chunks


def _format_youtube_ts(seconds: float) -> str:
    """Format seconds as YouTube timestamp: H:MM:SS or M:SS."""
    seconds = max(0, round(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 60}:{seconds % 60:02d}"


# ─── Parse Whisper JSON ────────────────────────────────────────────────────────

def _parse_whisper_json(json_path: str) -> tuple[list[dict], str, str]:
    """
    Parse Whisper JSON transcript:
      - Returns flattened list of word dicts: [{word, start, end}]
      - Returns video_sub (e.g. "1" from "4.1.json") and source_file name
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_words: list[dict] = []
    for seg in data.get("segments", []):
        for w in seg.get("words", []):
            word = w.get("word", "").strip()
            if word:
                all_words.append({
                    "word": word,
                    "start": w.get("start", 0.0),
                    "end": w.get("end", 0.0),
                })
        # Also include segment-level text as fallback for segments without words
        if not seg.get("words") and seg.get("text"):
            txt = seg["text"]
            dur = seg.get("end", 0) - seg.get("start", 0)
            s = seg.get("start", 0)
            for w_txt in txt.split():
                all_words.append({
                    "word": w_txt,
                    "start": s,
                    "end": s + dur / max(1, len(txt.split())),
                })

    video_file = data.get("video_file", Path(json_path).stem + ".mp4")
    return all_words, video_file, Path(json_path).name


# ─── Chapter metadata ─────────────────────────────────────────────────────────

# Reuse same maps from indexing_for_slide.py
SLIDE_NAME_MAP: dict[str, tuple[str, str, str]] = {
    "ch02": ("CS116-Bai02-Popular Libs.pdf",             "ch02", "Popular Libraries"),
    "ch03": ("CS116-Bai03-Pipeline & EDA.pdf",           "ch03", "Pipeline & EDA"),
    "ch04": ("CS116-Bai04-Data preprocessing.pdf",        "ch04", "Tiền xử lý dữ liệu"),
    "ch05": ("CS116-Bai05-Eval model.pdf",               "ch05", "Đánh giá mô hình"),
    "ch06": ("CS116-Bai06-Unsupervised learning.pdf",    "ch06", "Unsupervised Learning"),
    "ch07a": ("CS116-Bai07a-Supervised learning-Regression.pdf", "ch07a", "Supervised Learning - Regression"),
    "ch07b": ("CS116-Bai07b-Supervised learning-Classification.pdf", "ch07b", "Supervised Learning - Classification"),
    "ch08": ("CS116-Bai08-Deep learning với CNN.pdf",    "ch08", "Deep Learning với CNN"),
    "ch09": ("CS116-Bai09-Parameter tuning.pdf",         "ch09", "Parameter Tuning"),
    "ch10": ("CS116-Bai10-Ensemble model.pdf",            "ch10", "Ensemble Models"),
    "ch11": ("CS116-Bai11-Model Deployment.pdf",         "ch11", "Model Deployment"),
}

SLIDE_TOPICS: dict[str, list[str]] = {
    "ch04": ["Missing Data", "Outlier Detection", "Feature Extraction",
             "Feature Transformation", "Feature Selection"],
    "ch02": ["NumPy", "Pandas", "Matplotlib", "Scikit-learn"],
    "ch03": ["Pipeline", "Exploratory Data Analysis"],
    "ch05": ["Classification Metrics", "Regression Metrics", "Cross-validation"],
    "ch06": ["Clustering", "Dimensionality Reduction"],
    "ch07a": ["Linear Regression", "Regularization"],
    "ch07b": ["Logistic Regression", "Decision Trees", "SVM"],
    "ch08": ["Neural Networks", "CNN"],
    "ch09": ["Grid Search", "Random Search", "Bayesian Optimization"],
    "ch10": ["Bagging", "Boosting", "Random Forest"],
    "ch11": ["Model Serving", "API", "Monitoring"],
}


def _get_chapter_id(video_sub: str) -> tuple[str, int]:
    """Parse '4.1' → ('ch04', 1). Returns (chapter_id, sub_index)."""
    try:
        if "." in video_sub:
            parts = video_sub.split(".")
            ch = int(parts[0])
            sub = int(parts[1])
        else:
            ch = int(video_sub)
            sub = 1
        return f"ch{ch:02d}", sub
    except (ValueError, IndexError):
        return "ch00", 1


def run_chunking() -> list[dict]:
    """
    Entry point: chunk all JSON transcripts in input/transcribe_data/.
    Returns list of all transcript chunks.
    """
    _build_video_url_map()

    transcript_dir = Config.INPUT_DIR / "transcribe_data"
    out_file = Config.PROCESSED_DIR / "transcript_chunks_with_timestamps.jsonl"
    Config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []
    json_files = sorted(transcript_dir.glob("*.json"))
    print(f"📂 Found {len(json_files)} JSON transcript files")

    for json_path in json_files:
        fname = json_path.name          # e.g. "4.1.json"
        video_sub_raw = fname.replace(".json", "")   # e.g. "4.1"
        chapter_id, sub_idx = _get_chapter_id(video_sub_raw)

        chapter_title = SLIDE_NAME_MAP.get(chapter_id, (None, chapter_id, "Unknown"))[2]
        topics = SLIDE_TOPICS.get(chapter_id, [])

        print(f"  📝 {fname} → chapter={chapter_id}, sub={sub_idx}")
        try:
            all_words, video_file, source_file = _parse_whisper_json(str(json_path))
        except Exception as e:
            print(f"  ❌ Error parsing {fname}: {e}")
            traceback.print_exc()
            continue

        if not all_words:
            print(f"  ⚠️  No words found in {fname}")
            continue

        chunks = _chunk_segments(
            all_words=all_words,
            chapter_id=chapter_id,
            video_sub=str(sub_idx),
            source_file=source_file,
            topics=topics,
            chapter_title=chapter_title,
        )
        print(f"     → {len(chunks)} chunks | first: {chunks[0]['timestamp_start']:.1f}s | last: {chunks[-1]['timestamp_end']:.1f}s")

        for c in chunks:
            c["video_sub"] = str(sub_idx)
            c["video_file"] = video_file

        all_chunks.extend(chunks)

    print(f"\n📊 Total transcript chunks: {len(all_chunks)}")

    # Save JSONL
    save_jsonl(all_chunks, out_file)
    print(f"✅ Saved: {out_file} ({out_file.stat().st_size / 1024:.0f} KB)")

    # Quick stats
    ts_starts = [c["timestamp_start"] for c in all_chunks]
    ts_ends   = [c["timestamp_end"]   for c in all_chunks]
    wc        = [c["word_count"]      for c in all_chunks]
    print(f"   timestamp range: {min(ts_starts):.1f}s – {max(ts_ends):.1f}s")
    print(f"   avg words/chunk: {sum(wc)/len(wc):.0f} | min: {min(wc)} | max: {max(wc)}")

    urls = [c["youtube_url"] for c in all_chunks if c["youtube_url"]]
    print(f"   chunks with YouTube URL: {len(urls)}/{len(all_chunks)}")

    return all_chunks


if __name__ == "__main__":
    run_chunking()
