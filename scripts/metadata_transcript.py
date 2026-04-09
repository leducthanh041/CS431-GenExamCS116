import json
import re
import os
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Fix credential path if relative
cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if cred_path and not Path(cred_path).is_absolute():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(BASE_DIR / cred_path)

# =========================
# Paths
# =========================
TRANSCRIPT_DIR = BASE_DIR / "data" / "chunked_transcript"
SLIDE_FILE = BASE_DIR / "data" / "concept_chunks.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "transcript_metadata_preview.jsonl"

# Set to None to process all files, or a set like {"10.1", "11.1"} to filter
TARGET_SECTIONS = None

# =========================
# Load prompt template from external file
# =========================
PROMPT_TEMPLATE_FILE = BASE_DIR / "prompts" / "transcript_metadata_prompt.txt"
if not PROMPT_TEMPLATE_FILE.exists():
    raise FileNotFoundError(f"Missing prompt template: {PROMPT_TEMPLATE_FILE}")

PROMPT_TEMPLATE = PROMPT_TEMPLATE_FILE.read_text(encoding="utf-8")


# =========================
# Load slide chunks grouped by chapter_id
# =========================
slide_by_chapter = defaultdict(list)
chapter_title_by_id = {}

with open(SLIDE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        slide = json.loads(line)
        chapter_id = slide.get("chapter_id")
        if not chapter_id:
            continue

        slide_by_chapter[chapter_id].append(slide)

        if chapter_id not in chapter_title_by_id:
            chapter_title_by_id[chapter_id] = slide.get("chapter_title", "")


# =========================
# Utility functions
# =========================
def section_to_chapter_id(section: str) -> str:
    major = int(section.split(".")[0])
    if major == 7:
        return "ch07a"
    elif major == 8:
        return "ch07b"
    elif major in (9, 10):
        return "ch08"
    elif major >= 11:
        return f"ch{major - 2:02d}"
    return f"ch{major:02d}"


def build_slide_context(slides, max_slides=5):
    """
    Build short text context from slide chunks.
    For now simply take first max_slides slide chunks in the chapter.
    You can later replace this with semantic retrieval.
    """
    contexts = []

    for slide in slides[:max_slides]:
        page = slide.get("page_number", "?")
        section_title = slide.get("section_title", "")
        topics = ", ".join(slide.get("topics", []))
        text = slide.get("text", "").strip().replace("\n", " ")

        if len(text) > 300:
            text = text[:300] + "..."

        contexts.append(
            f"[Slide p{page}] {section_title}\n"
            f"Topics: {topics}\n"
            f"Text: {text}"
        )

    return "\n\n".join(contexts)

# =========================
# Load transcript chunks
# Assumes each file in chunked_transcript is json/jsonl containing chunk objects
# with fields such as:
# {
#   "chunk_id": "...",
#   "section": "10.1",
#   "text": "..."
# }
# =========================
all_results = []

def file_sort_key(p):
    name_stripped = p.name.replace(".cleaned.json", "").replace(".json", "")
    try:
        return [int(x) for x in name_stripped.split(".")]
    except ValueError:
        return [0]

all_files = sorted(TRANSCRIPT_DIR.glob("*.json*"), key=file_sort_key)

for file_path in all_files:
    section_from_filename = file_path.name.replace(".cleaned.json", "").replace(".json", "").strip()
    if TARGET_SECTIONS and section_from_filename not in TARGET_SECTIONS:
        continue

    print(f"Processing {file_path.name}")

    chunks = []

    if file_path.suffix == ".jsonl":
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # support either list or dict with key 'chunks'
        if isinstance(data, list):
            chunks = data
        elif isinstance(data, dict) and "chunks" in data:
            chunks = data["chunks"]
        else:
            print(f"Skip unsupported format: {file_path}")
            continue

    for idx, chunk in enumerate(chunks):
        # The section is part of the filename: e.g., "10.1.cleaned.json" -> "10.1"
        section = file_path.name.replace(".cleaned.json", "").replace(".json", "").strip()

        if TARGET_SECTIONS and section not in TARGET_SECTIONS:
            continue

        chapter_id = section_to_chapter_id(section)
        if chapter_id == "ch01":
            chapter_title = "Introduction"
        else:
            chapter_title = chapter_title_by_id.get(chapter_id, "")
        
        chapter_slides = slide_by_chapter.get(chapter_id, [])

        slide_context = build_slide_context(chapter_slides)

        prev_chunk_id = chunks[idx - 1]["chunk_id"] if idx > 0 else None
        next_chunk_id = chunks[idx + 1]["chunk_id"] if idx < len(chunks) - 1 else None

        result = {
            "chunk_id": chunk["chunk_id"],
            "course_id": "CS116",
            "chapter_id": chapter_id,
            "chapter_title": chapter_title,
            "source_type": "transcript",
            "source_file": file_path.name,
            "section": section,
            "text": chunk.get("text", ""),
            "related_chunks": [
                cid for cid in [prev_chunk_id, next_chunk_id] if cid is not None
            ],
            "slide_context": slide_context,
            "llm_prompt": PROMPT_TEMPLATE.replace(
                "{chapter_id}", str(chapter_id)
            ).replace(
                "{chapter_title}", str(chapter_title)
            ).replace(
                "{transcript_text}", str(chunk.get("text", ""))
            ).replace(
                "{slide_context}", str(slide_context)
            )
        }

        all_results.append(result)

# =========================
# Save preview output
# =========================
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in all_results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Saved {len(all_results)} preview items to {OUTPUT_FILE}")

# =========================
# Optional: write prompts to separate file for inspection
# =========================
PROMPT_FILE = BASE_DIR / "data" / "transcript_metadata_prompts.jsonl"

with open(PROMPT_FILE, "w", encoding="utf-8") as f:
    for item in all_results:
        f.write(json.dumps({
            "chunk_id": item["chunk_id"],
            "chapter_id": item["chapter_id"],
            "prompt": item["llm_prompt"]
        }, ensure_ascii=False) + "\n")

print(f"Saved prompts to {PROMPT_FILE}")



# =========================
# Optional: call Gemini on Vertex AI
# Requires:
#   pip install google-genai
#   export GOOGLE_CLOUD_PROJECT=...
#   export GOOGLE_CLOUD_LOCATION=us-central1
# =========================
RUN_LLM = True
MODEL_NAME = "gemini-2.5-flash"

if RUN_LLM:
    from google import genai
    from google.genai import types

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        print("Skipping LLM generation: GOOGLE_CLOUD_PROJECT is not set.")
    else:
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        )

        FINAL_OUTPUT = BASE_DIR / "data" / "transcript_metadata_generated.jsonl"
        
        # Hàm xử lý cho 1 chunk
        def process_chunk(item):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=item["llm_prompt"],
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json"
                    )
                )
                raw_text = response.text.strip()
                try:
                    metadata = json.loads(raw_text)
                except Exception:
                    metadata = {"parse_error": True, "raw_response": raw_text}
            except Exception as e:
                metadata = {"api_error": str(e)}

            return {
                "chunk_id": item["chunk_id"],
                "course_id": item["course_id"],
                "chapter_id": item["chapter_id"],
                "chapter_title": item["chapter_title"],
                "source_type": item["source_type"],
                "source_file": item["source_file"],
                "related_chunks": item["related_chunks"],
                # Lấy text gốc trực tiếp từ item, không cần chờ LLM gen lại
                "text": item.get("text", ""),
                **metadata
            }

        # Chạy đa luồng
        MAX_WORKERS = 10 # Số lượng request gọi cùng lúc. Có thể tăng/giảm tùy quota Vertex AI của bạn.
        
        unprocessed_results = all_results
        total_unprocessed = len(unprocessed_results)
        
        if not unprocessed_results:
            print("Không tìm thấy file json nào để xử lý!")
        else:
            print(f"Bắt đầu xử lý TOÀN BỘ {total_unprocessed} chunks từ đầu với {MAX_WORKERS} luồng...")
            
            start_time = time.time()
            
            # Ghi đè file từ đầu (mode 'w')
            with open(FINAL_OUTPUT, "w", encoding="utf-8") as fout:
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    for i, final_obj in enumerate(executor.map(process_chunk, unprocessed_results), 1):
                        fout.write(json.dumps(final_obj, ensure_ascii=False) + "\n")
                        fout.flush() # Ghi ngay lập tức xuống đĩa
                        
                        elapsed = time.time() - start_time
                        remaining = total_unprocessed - i
                        eta_seconds = int((elapsed / i) * remaining)
                        eta_str = f"{eta_seconds // 60}m{eta_seconds % 60:02d}s"
                        percent = (i / total_unprocessed) * 100
                        
                        print(f"[{percent:5.1f}% | Xong: {i} | Còn lại: {remaining} | Tổng: {total_unprocessed}] Ghi: {final_obj['chunk_id']} | ETA: {eta_str}")
                        
            print(f"Hoàn thành! Đã lưu mới toàn bộ metadata vào {FINAL_OUTPUT}")