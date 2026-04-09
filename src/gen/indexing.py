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
    "ch08": ("CS116-Bai08-Deep learning với CNN.pdf", "ch08", "Deep Learning với CNN"),
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

# ─── Data Cleaning cho Slide ───────────────────────────────────────────────

def clean_slide_text(text: str) -> str:
    """Dọn dẹp các dữ liệu nhiễu, header, footer và tag hình ảnh rỗng từ slide."""
    if not text:
        return ""
        
    # 1. Xóa Footer cố định và tên giảng viên
    text = re.sub(r'(?i)Thực hiện bởi Trường Đại học Công nghệ Thông tin, ĐHQG-HCM\s*', '', text)
    text = re.sub(r'(?i)ĐẠI HỌC QUỐC GIA TP\. HỒ CHÍ MINH\s*', '', text)
    text = re.sub(r'(?i)TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN\s*', '', text)
    text = re.sub(r'(?i)TS\. Nguyễn Vinh Tiệp\s*', '', text)
    
    # 2. Xóa các tag hình ảnh rác do thư viện pymupdf4llm sinh ra
    text = re.sub(r'\**==> picture.*?<==\**', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\**----- Start of picture text -----\**.*?\**----- End of picture text -----\**', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<br>', '\n', text, flags=re.IGNORECASE)
    
    # 3. Xóa các tag hình ảnh kiểu cũ
    text = re.sub(r'\[Image\s*\d+\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    
    # 4. Xóa các con số chơ vơ trên 1 dòng (thường là số trang bị tách ra)
    text = re.sub(r'(?m)^\s*\d+\s*$\n?', '', text)
    
    return text.strip()


# ─── Hàm trích xuất Slide (Dùng Markdown) ─────────────────────────

def extract_slide_pdf(pdf_path: str, chapter_id: str) -> list[dict]:
    """Trích xuất text từ slide PDF bằng pymupdf4llm, giữ cấu trúc Markdown."""
    chunks = []
    try:
        import pymupdf4llm
    except ImportError:
        print("⚠️  pymupdf4llm not installed. Run: pip install pymupdf4llm")
        return chunks

    filename = Path(pdf_path).name
    chapter_title = SLIDE_NAME_MAP.get(chapter_id, (None, chapter_id, "Unknown"))[2]
    topics = SLIDE_TOPICS.get(chapter_id, [])

    try:
        # Trích xuất toàn bộ PDF thành danh sách các trang chứa Markdown
        md_pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    except Exception as e:
        print(f"  ⚠️  Cannot process PDF {pdf_path}: {e}")
        return chunks

    for page_data in md_pages:
        page_num = page_data.get("metadata", {}).get("page", 0) + 1
        raw_text = page_data.get("text", "")
        
        # 1. Làm sạch text
        cleaned_text = clean_slide_text(raw_text)
        text_lower = cleaned_text.lower()
        
        # 2. Lọc bỏ các slide vô nghĩa, mục lục, hỏi đáp, hoặc quá ngắn
        if len(cleaned_text.split()) < 5 or text_lower.strip() == "nội dung" or "bài quiz và hỏi đáp" in text_lower:
            continue

        # 3. Lấy Section Title từ dòng đầu tiên có nội dung
        lines = cleaned_text.split('\n')
        section_title = ""
        for line in lines:
            line_clean = line.replace("#", "").replace("*", "").strip()
            if line_clean:
                section_title = line_clean
                break

        chunk_id = f"cs116_{chapter_id}_slide_p{page_num:03d}"
        chunk = {
            "chunk_id": chunk_id,
            "course_id": "CS116",
            "chapter_id": chapter_id,
            "chapter_title": chapter_title,
            "topics": topics,
            "source_type": "slide_pdf",
            "source_file": filename,
            "page_number": page_num,
            "section_title": section_title,
            "text": cleaned_text,  
            "embedding_ready": True,
        }
        chunks.append(chunk)

    return chunks


# ─── Cập nhật hàm trích xuất Transcript (Dọn rác & Chống lặp từ) ─────────

def extract_transcript(txt_path: str, chapter_id: str) -> list[dict]:
    """Trích xuất text từ transcript TXT file, dọn dẹp tag phụ đề và chống lặp từ."""
    chunks = []
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"  ⚠️  Cannot read transcript {txt_path}: {e}")
        return chunks

    # 1. Xóa Header của file Subtitle/VTT
    text = re.sub(r'(?i)Kind: captions Language: vi', '', text)
    
    # 2. Xóa toàn bộ tag thời gian và tag định dạng (VD: <00:08:57.320>, <c>, </c>)
    text = re.sub(r'<[^>]+>', '', text)
    
    # 3. Xóa các ghi chú âm thanh trong ngoặc vuông (VD: [Nhạc nền])
    text = re.sub(r'\[.*?\]', '', text)
    
    # 4. Gom khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. [QUAN TRỌNG] Chống lặp từ (Rolling Captions Deduplication)
    # Tìm các cụm từ (từ 2 đến 100 ký tự) bị lặp lại liên tiếp do lỗi nối file và chỉ giữ lại 1 cụm.
    # VD: "có có cái kích kích thước" -> "có cái kích thước"
    # VD: "tham số của mình nó quá đơn giản tham số của mình nó quá đơn giản" -> "tham số của mình nó quá đơn giản"
    text = re.sub(r'\b(.{2,100}?)(?:\s+\1\b)+', r'\1', text)
    
    if not text:
        return chunks

    chapter_title = SLIDE_NAME_MAP.get(chapter_id, (None, chapter_id, "Unknown"))[2]
    topics = SLIDE_TOPICS.get(chapter_id, [])

    # Simple semantic chunking: split by sentences, group ~300-500 tokens
    words = text.split()
    chunk_size = 200  # words per chunk
    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i:i+chunk_size])
        seq = i // chunk_size + 1
        chunk_id = f"cs116_{chapter_id}_transcript_s{seq:03d}"
        chunk = {
            "chunk_id": chunk_id,
            "course_id": "CS116",
            "chapter_id": chapter_id,
            "chapter_title": chapter_title,
            "topics": topics,
            "source_type": "video_transcript",
            "source_file": Path(txt_path).name,
            "page_number": seq,
            "section_title": "",
            "text": chunk_text,
            "embedding_ready": True,
        }
        chunks.append(chunk)

    return chunks


# ─── Hàm Embedding và Lưu trữ ──────────────────────────────────────────────

def embed_and_store(chunks: list[dict]) -> list[dict]:
    """
    Embed chunks bằng BGE-m3 và lưu vào ChromaDB.
    Nếu ChromaDB đã có data thì skip embedding, chỉ verify.
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

    # Check if already embedded
    if collection.count() > 0:
        print(f"  ℹ️  ChromaDB already has {collection.count()} chunks — skipping embed")
        return chunks

    # ─── Embedding model ──────────────────────────────────────────
    print("  🔄 Loading BGE-m3 embedding model...")
    model = SentenceTransformer("BAAI/bge-m3")
    texts = [c["text"][:2000] for c in chunks]  # Truncate long text
    print(f"  🔄 Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    # ─── Store in ChromaDB ─────────────────────────────────────────
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {
            "chunk_id": c["chunk_id"],
            "chapter_id": c["chapter_id"],
            "chapter_title": c.get("chapter_title", ""),
            "topic": "|".join(c.get("topics", [])) if c.get("topics") else "",
            "section_title": c.get("section_title", ""),
            "source_type": c.get("source_type", ""),
            "source_file": c.get("source_file", ""),
        }
        for c in chunks
    ]

    collection.add(
        ids=ids,
        documents=[c["text"] for c in chunks],
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )
    print(f"  ✅ Stored {len(chunks)} chunks in ChromaDB")
    return chunks


# ─── Hàm Main ──────────────────────────────────────────────────────────────

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
                print(f"     → {len(chunks)} slide chunks (Clean Markdown)")
            except Exception as e:
                print(f"  ❌ Error extracting {filename}: {e}")
                traceback.print_exc()

    # ─── Index video transcripts ────────────────────────────────────
    transcript_dir = Config.INPUT_DIR / "video_transcript"
    if transcript_dir.exists():
        print(f"\n📂 Indexing transcripts from: {transcript_dir}")
        for chapter_id in SLIDE_NAME_MAP:
            chapter_num = chapter_id.lstrip("ch0")
            chapter_glob = chapter_id.lstrip("ch")  

            for txt_file in sorted(transcript_dir.glob(f"{chapter_glob}.*.txt")):
                print(f"  📝 {chapter_id}: {txt_file.name}")
                try:
                    chunks = extract_transcript(str(txt_file), chapter_id)
                    all_chunks.extend(chunks)
                    print(f"     → {len(chunks)} transcript chunks")
                except Exception as e:
                    print(f"  ❌ Error: {e}")

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