"""
indexing.py — Step 01: Indexing Pipeline
MCQGen Pipeline: slide PDF + transcript TXT → concept_chunks.jsonl + ChromaDB

Input:
  - input/slide/*.pdf
  - input/video_transcript/*.txt

Output:
  - data/processed/concept_chunks.jsonl
  - data/indexes/concept_kb/         (ChromaDB)
  - data/indexes/assessment_kb/     (ChromaDB, if assessment items exist)
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import Config, config, save_jsonl


# ─── Chapter → slide filename mapping ──────────────────────────────────────

SLIDE_NAME_MAP: dict[str, tuple[str, str, str]] = {
    "ch02": ("CS116-Bai02-Popular Libs.pdf",             "ch02", "Popular Libraries"),
    "ch03": ("CS116-Bai03-Pipeline & EDA.pdf",           "ch03", "Pipeline & EDA"),
    "ch04": ("CS116-Bai04-Data preprocessing.pdf",        "ch04", "Tiền xử lý dữ liệu"),
    "ch05": ("CS116-Bai05-Eval model.pdf",               "ch05", "Đánh giá mô hình"),
    "ch06": ("CS116-Bai06-Unsupervised learning.pdf",    "ch06", "Unsupervised Learning"),
    "ch07a": ("CS116-Bai07a-Supervised learning-Regression.pdf", "ch07a", "Supervised Learning - Regression"),
    "ch07b": ("CS116-Bai07b-Supervised learning-Classification.pdf", "ch07b", "Supervised Learning - Classification"),
    "ch08": ("CS116-Bai08-Deep learning v#U1edbi CNN.pdf", "ch08", "Deep Learning với CNN"),
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


def extract_slide_pdf(pdf_path: str, chapter_id: str) -> list[dict]:
    """Trích xuất text từ slide PDF, mỗi trang → 1 chunk."""
    chunks = []
    try:
        import fitz
    except ImportError:
        print("⚠️  PyMuPDF not installed. Skipping PDF extraction.")
        return chunks

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  ⚠️  Cannot open PDF {pdf_path}: {e}")
        return chunks

    filename = Path(pdf_path).name
    chapter_title = SLIDE_NAME_MAP.get(chapter_id, (None, chapter_id, "Unknown"))[2]
    topics = SLIDE_TOPICS.get(chapter_id, [])

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if not text:
            continue

        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) < 2:
            continue

        section_title = lines[0]
        body_text = " ".join(lines[1:])

        chunk_id = f"cs116_{chapter_id}_slide_p{page_num+1:03d}"
        chunk = {
            "chunk_id": chunk_id,
            "course_id": "CS116",
            "chapter_id": chapter_id,
            "chapter_title": chapter_title,
            "topics": topics,
            "source_type": "slide_pdf",
            "source_file": filename,
            "page_number": page_num + 1,
            "section_title": section_title,
            "text": body_text,
            "embedding_ready": True,
        }
        chunks.append(chunk)

    doc.close()
    return chunks


def load_transcript_chunks() -> list[dict]:
    """
    Load pre-chunked transcript JSONL từ Step 01b (chunk_transcript_with_timestamps.py).
    Mỗi chunk đã có: timestamp_start, timestamp_end, youtube_url, youtube_timestamp_*.
    """
    transcript_jsonl = Config.PROCESSED_DIR / "transcript_chunks_with_timestamps.jsonl"
    if not transcript_jsonl.exists():
        print(f"  ⚠️  Transcript chunks not found: {transcript_jsonl}")
        print("     Run: python -u src/gen/chunk_transcript_with_timestamps.py first")
        return []

    chunks = []
    with open(transcript_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    print(f"  📝 Loaded {len(chunks)} transcript chunks from JSONL")
    return chunks


def embed_and_store(chunks: list[dict]) -> list[dict]:
    """
    Embed chunks bằng BGE-m3 và lưu vào ChromaDB.
    Nếu ChromaDB đã có data thì xóa và embed lại để đảm bảo metadata mới.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
    except ImportError as e:
        print(f"⚠️  Missing dependency: {e}")
        print("   Run: pip install sentence-transformers chromadb")
        return chunks

    # ─── ChromaDB setup ────────────────────────────────────────────
    client = chromadb.PersistentClient(path=str(Config.INDEX_DIR))
    collection = client.get_or_create_collection("concept_chunks")

    # Nếu đã có data → xóa và embed lại (đảm bảo metadata mới: timestamp, youtube_url...)
    if collection.count() > 0:
        print(f"  ℹ️  ChromaDB has {collection.count()} old chunks — deleting to re-embed with new metadata...")
        client.delete_collection("concept_chunks")
        collection = client.get_or_create_collection("concept_chunks")

    # ─── Embedding model ──────────────────────────────────────────
    print("  🔄 Loading BGE-m3 embedding model...")
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    texts = [c["text"][:2000] for c in chunks]
    print(f"  🔄 Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    # ─── Store in ChromaDB với metadata mở rộng ───────────────────────
    ids = [c["chunk_id"] for c in chunks]
    metadatas = []
    for c in chunks:
        meta = {
            "chunk_id":       c["chunk_id"],
            "chapter_id":     c.get("chapter_id", ""),
            "chapter_title":  c.get("chapter_title", ""),
            "topic":          "|".join(c.get("topics", [])) if c.get("topics") else "",
            "section_title":  c.get("section_title", ""),
            "source_type":    c.get("source_type", ""),
            "source_file":     c.get("source_file", ""),
            # Transcript-specific (convert None → "" for ChromaDB compatibility)
            "youtube_url":      str(c.get("youtube_url", "")) or "",
            "youtube_ts_start": str(c.get("youtube_timestamp_start", "")) or "",
            "youtube_ts_end":   str(c.get("youtube_timestamp_end", "")) or "",
            # Slide citation metadata
            "slide_file":       str(c.get("slide_file", "")) or "",
            # page_number: 1-based page from original slide chunk
            # slide_start_page: 0-based page stored in ChromaDB (use 0-based for 1-based display)
            "slide_start_page": int(c.get("slide_start_page") or c.get("page_number", 0) or 0),
        }
        # timestamp_start/end as float — only include if not None
        ts_start = c.get("timestamp_start")
        ts_end   = c.get("timestamp_end")
        if ts_start is not None:
            meta["timestamp_start"] = float(ts_start)
        if ts_end is not None:
            meta["timestamp_end"] = float(ts_end)
        metadatas.append(meta)

    collection.add(
        ids=ids,
        documents=[c["text"] for c in chunks],
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )
    print(f"  ✅ Stored {len(chunks)} chunks in ChromaDB")
    return chunks


def run_indexing():
    """
    Entry point cho Step 01 — indexing toàn bộ slide + transcript.
    """
    config.makedirs()

    all_chunks = []

    # ─── Index slide PDFs ─────────────────────────────────────────
    slide_dir = Config.INPUT_DIR / "slide"
    if slide_dir.exists():
        print(f"📂 Indexing slides from: {slide_dir}")
        for chapter_id, (filename, _, _) in SLIDE_NAME_MAP.items():
            pdf_path = slide_dir / filename
            if not pdf_path.exists():
                # Thử tìm filename gần đúng
                candidates = list(slide_dir.glob(f"*{filename.split('-')[1]}*"))
                if candidates:
                    pdf_path = candidates[0]
                else:
                    print(f"  ⚠️  Slide not found: {filename}")
                    continue

            print(f"  📄 {chapter_id}: {pdf_path.name}")
            try:
                chunks = extract_slide_pdf(str(pdf_path), chapter_id)
                all_chunks.extend(chunks)
                print(f"     → {len(chunks)} slide chunks")
            except Exception as e:
                print(f"  ❌ Error extracting {filename}: {e}")
                traceback.print_exc()

    # ─── Load pre-chunked video transcripts (Step 01b output) ────────────
    print(f"\n📂 Loading transcript chunks from JSONL...")
    transcript_chunks = load_transcript_chunks()
    all_chunks.extend(transcript_chunks)

    # ─── Deduplicate ───────────────────────────────────────────────
    seen = set()
    unique_chunks = []
    for c in all_chunks:
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            unique_chunks.append(c)

    print(f"\n📊 Total unique chunks: {len(unique_chunks)}")

    # ─── Save concept_chunks.jsonl ────────────────────────────────
    chunks_file = Config.CONCEPT_CHUNKS_FILE
    save_jsonl(unique_chunks, chunks_file)
    print(f"✅ Saved concept_chunks.jsonl: {chunks_file}")

    # ─── Embed + store in ChromaDB ─────────────────────────────────
    print("\n🔄 Embedding chunks into ChromaDB...")
    embed_and_store(unique_chunks)

    print(f"\n✅ Indexing done: {len(unique_chunks)} chunks indexed")


if __name__ == "__main__":
    run_indexing()
