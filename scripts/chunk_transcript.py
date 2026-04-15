import json
import re
from pathlib import Path

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


# ==========================================================
# ==========================================================
# CONFIG
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "cleaned_transcript"
OUTPUT_DIR = BASE_DIR / "data" / "chunked_transcript"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_FILES = 3

# semantic chunk params
BREAKPOINT_PERCENTILE = 85

# post-processing params
MIN_WORDS = 50
MAX_WORDS = 250
OVERLAP_SENTENCES = 1


# ==========================================================
# HELPERS
# ==========================================================
def normalize_text(text: str) -> str:
    """Làm sạch khoảng trắng trước khi chunk."""
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    return text.strip()


def word_count(text: str) -> int:
    return len(text.split())


def split_sentences(text: str):
    """
    Tách câu đơn giản theo . ? !
    Giữ lại dấu câu.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def split_large_chunk(text: str, max_words: int = MAX_WORDS):
    """
    Nếu chunk quá lớn thì cắt tiếp theo sentence boundary.
    """
    sentences = split_sentences(text)

    chunks = []
    current = []

    for sentence in sentences:
        temp = " ".join(current + [sentence])

        if word_count(temp) > max_words and current:
            chunks.append(" ".join(current).strip())
            current = [sentence]
        else:
            current.append(sentence)

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def merge_small_chunks(chunks, min_words: int = MIN_WORDS):
    """
    Gộp chunk quá ngắn vào chunk trước.
    """
    merged = []

    for chunk in chunks:
        if merged and word_count(chunk) < min_words:
            merged[-1] = merged[-1].strip() + " " + chunk.strip()
        else:
            merged.append(chunk.strip())

    return merged


def add_overlap(chunks, overlap_sentences: int = OVERLAP_SENTENCES):
    """
    Thêm 1 câu cuối của chunk trước vào chunk hiện tại.
    """
    if overlap_sentences <= 0:
        return chunks

    result = []

    for i, chunk in enumerate(chunks):
        if i == 0:
            result.append(chunk)
            continue

        prev_sentences = split_sentences(chunks[i - 1])
        overlap = " ".join(prev_sentences[-overlap_sentences:])

        new_chunk = overlap + " " + chunk
        result.append(new_chunk.strip())

    return result


# ==========================================================
# EMBEDDING + CHUNKER
# ==========================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

chunker = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=BREAKPOINT_PERCENTILE,
)


# ==========================================================
# PROCESS FILES
# ==========================================================
json_files = sorted(INPUT_DIR.glob("*.json"))

for file_path in json_files:
    output_path = OUTPUT_DIR / file_path.name
    if output_path.exists():
        print(f"\nSkipping (already exists): {file_path.name}")
        continue

    print(f"\nProcessing: {file_path.name}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_text = normalize_text(data["cleaned_text"])

    # ------------------------------------------------------
    # Semantic chunking từ LangChain
    # ------------------------------------------------------
    docs = chunker.create_documents([cleaned_text])

    raw_chunks = [
        doc.page_content.strip()
        for doc in docs
        if doc.page_content.strip()
    ]

    print(f"  Raw chunks: {len(raw_chunks)}")

    # ------------------------------------------------------
    # Merge chunk quá ngắn
    # ------------------------------------------------------
    chunks = merge_small_chunks(raw_chunks, MIN_WORDS)

    # ------------------------------------------------------
    # Split chunk quá dài
    # ------------------------------------------------------
    split_chunks = []

    for chunk in chunks:
        if word_count(chunk) > MAX_WORDS:
            split_chunks.extend(split_large_chunk(chunk, MAX_WORDS))
        else:
            split_chunks.append(chunk)

    chunks = split_chunks

    # ------------------------------------------------------
    # Add overlap
    # ------------------------------------------------------
    chunks = add_overlap(chunks, OVERLAP_SENTENCES)

    # ------------------------------------------------------
    # Convert sang JSON output
    # ------------------------------------------------------
    output = []

    for idx, chunk in enumerate(chunks):
        output.append({
            "chunk_id": idx,
            "text": chunk,
            "word_count": word_count(chunk)
        })

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  Final chunks: {len(output)}")
    print(f"  Saved: {output_path}")

    # preview vài chunk đầu
    for chunk in output[:3]:
        print("\n---")
        print(f"Chunk {chunk['chunk_id']} ({chunk['word_count']} words)")
        print(chunk["text"][:300])