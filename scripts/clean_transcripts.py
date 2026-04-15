import json
import os
from pathlib import Path
import re

from dotenv import load_dotenv
from google import genai
from tqdm import tqdm


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw_transcript"
CLEAN_DIR = BASE_DIR / "data" / "cleaned_transcript"
PROMPT_PATH = BASE_DIR / "prompts" / "transcript_cleanup.txt"

# =========================
# Load environment variables
# =========================
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Fix credential path if relative
cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if cred_path and not Path(cred_path).is_absolute():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(BASE_DIR / cred_path)

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

CLEAN_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Initialize Vertex AI client
# =========================
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

# =========================
# Load cleanup prompt template
# =========================
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    CLEANUP_PROMPT = f.read().strip()


# Find all transcript files
all_transcript_files = sorted(RAW_DIR.glob("*.json"))

# Filter to only unprocessed files
transcript_files = [
    f for f in all_transcript_files
    if not (CLEAN_DIR / f"{f.stem}.cleaned.json").exists()
]

if len(all_transcript_files) == 0:
    raise FileNotFoundError(
        f"No transcript files found in: {RAW_DIR}"
    )

print(f"Found {len(all_transcript_files)} total transcript files. {len(transcript_files)} need processing.")

if len(transcript_files) == 0:
    print("All files are already cleaned. Exiting.")
    import sys
    sys.exit(0)

# =========================
# Process each transcript
# =========================
from concurrent.futures import ThreadPoolExecutor
import time

def process_transcript(transcript_path):
    print(f"\n[START] Đang xử lý: {transcript_path.name}")
    try:
        output_path = CLEAN_DIR / f"{transcript_path.stem}.cleaned.json"

        # Load transcript JSON
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

        raw_text = transcript_data.get("text", "").strip()

        if not raw_text:
            return f"[SKIP] Empty text in {transcript_path.name}"

        # Build final prompt
        full_prompt = f"""
{CLEANUP_PROMPT}

Transcript:
{raw_text}
"""

        # Call Gemini with auto-retry for 503 errors
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=full_prompt,
                    config={
                        "temperature": 0,
                        "response_mime_type": "application/json",
                    },
                )
                break  # Thành công thì thoát vòng lặp retry
            except Exception as e:
                if attempt < 2 and "503" in str(e):
                    time.sleep(10)
                    continue
                else:
                    raise e

        # Chỉ lấy text một lần duy nhất
        raw_output = response.text.strip()

        # Dùng Regex xóa mọi loại Markdown Code Block nếu có
        clean_json_str = re.sub(r"^```(?:json)?|```$", "", raw_output, flags=re.MULTILINE).strip()

        # Fix potentially truncated JSON (Gemini API sometimes cuts off long responses)
        if clean_json_str.startswith('{') and not clean_json_str.endswith('}'):
            if clean_json_str.endswith('"'):
                clean_json_str += '}'
            else:
                clean_json_str += '"}'

        try:
            result = json.loads(clean_json_str)

        except json.JSONDecodeError:
            # Save raw response for debugging
            debug_path = CLEAN_DIR / f"{transcript_path.stem}.raw_response.txt"
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(clean_json_str)

            # Try extracting first JSON object
            match = re.search(r"\{.*\}", clean_json_str, re.DOTALL)

            if not match:
                raise ValueError("No JSON object found in response")

            json_part = match.group(0)

            try:
                result = json.loads(json_part)

            except json.JSONDecodeError:
                # Final fallback: extract cleaned_text directly
                match_text = re.search(
                    r'"cleaned_text"\s*:\s*"(.*)"',
                    json_part,
                    re.DOTALL
                )

                if not match_text:
                    raise

                cleaned_text = match_text.group(1)
                cleaned_text = cleaned_text.replace('\\"', '"')
                cleaned_text = cleaned_text.replace("\\n", "\n")

                result = {"cleaned_text": cleaned_text}

        cleaned_text = result["cleaned_text"].strip()

        # Save cleaned transcript
        output_data = {
            "source_file": transcript_path.name,
            "cleaned_text": cleaned_text,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        return f"[DONE] {transcript_path.name} -> {output_path.name}"

    except Exception as e:
        return f"[ERROR] Failed processing {transcript_path.name}: {e}"


MAX_WORKERS = 2
print(f"Bắt đầu gọi API với {MAX_WORKERS} luồng...")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    for res in tqdm(executor.map(process_transcript, transcript_files), total=len(transcript_files)):
        if res and res.startswith("[ERROR]"):
            tqdm.write(res)
        elif res and res.startswith("[SKIP]"):
            tqdm.write(res)

print("Hoàn thành!")