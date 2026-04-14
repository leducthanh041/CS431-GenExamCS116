import os
import json
import re
import time
import threading
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from google import genai
from google.api_core.exceptions import ResourceExhausted

# =========================
# Load environment variables
# =========================
load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# =========================
# Init client
# =========================
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_PATH = BASE_DIR / "prompts" / "misconception_generate_prompt.txt"
INPUT_DIR = BASE_DIR / "input" / "question_bank" / "multiple_choice"
OUTPUT_DIR = BASE_DIR / "data" / "misconceptions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Load prompt
# =========================
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# =========================
# Logging helper
# =========================
_log_lock = threading.Lock()

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    tid = threading.current_thread().name
    with _log_lock:
        print(f"[{ts}] [{tid}] {msg}", flush=True)

# =========================
# Helper: strip markdown code block if present
# =========================
def strip_markdown_json(text: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text

# =========================
# Call model with retry on 429
# =========================
MAX_RETRIES = 5
RETRY_BASE_DELAY = 10  # seconds

def call_model_with_retry(prompt: str, file_name: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            return response.text.strip()
        except ResourceExhausted:
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))  # 10, 20, 40, 80s
            log(f"RETRY  {file_name} — 429 rate limit, waiting {wait}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait)
        except Exception as e:
            # Fallback: catch raw 429 from google.genai if not wrapped
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt == MAX_RETRIES:
                    raise
                wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                log(f"RETRY  {file_name} — 429 rate limit, waiting {wait}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise

# =========================
# Process one file
# =========================
def process_file(input_file: Path):
    t0 = time.time()
    log(f"START  {input_file.name}")

    with open(input_file, "r", encoding="utf-8") as f:
        mcq_data = json.load(f)

    n_questions = len(mcq_data) if isinstance(mcq_data, list) else "?"
    log(f"LOADED {input_file.name} ({n_questions} questions) — calling model...")

    prompt = f"{SYSTEM_PROMPT}\n\nInput:\n{json.dumps(mcq_data, ensure_ascii=False, indent=2)}"
    raw_text = call_model_with_retry(prompt, input_file.name)

    elapsed_api = time.time() - t0
    log(f"RECV   {input_file.name} — API response in {elapsed_api:.1f}s")

    output_text = strip_markdown_json(raw_text)
    output_file = OUTPUT_DIR / input_file.name

    try:
        parsed = json.loads(output_text)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        elapsed = time.time() - t0
        log(f"OK     {input_file.name} -> {output_file.relative_to(BASE_DIR)} ({elapsed:.1f}s total)")
    except json.JSONDecodeError:
        raw_output_file = OUTPUT_DIR / f"{input_file.stem}_raw.txt"
        with open(raw_output_file, "w", encoding="utf-8") as f:
            f.write(output_text)
        elapsed = time.time() - t0
        log(f"FAIL   {input_file.name} — JSON parse failed, saved raw ({elapsed:.1f}s total)")

# =========================
# Run all files with 10 threads
# =========================
files = sorted(INPUT_DIR.glob("*.json"))

if not files:
    raise ValueError("No JSON files found in input folder")

MAX_WORKERS = 5
print(f"Found {len(files)} file(s). Starting {MAX_WORKERS} threads...\n", flush=True)

t_start = time.time()
done = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="worker") as executor:
    futures = {executor.submit(process_file, f): f for f in files}
    for future in as_completed(futures):
        done += 1
        try:
            future.result()
        except Exception as e:
            with _log_lock:
                print(f"[ERROR] {futures[future].name}: {e}", flush=True)
        with _log_lock:
            print(f"  --> Progress: {done}/{len(files)} done", flush=True)

print(f"\nAll done in {time.time() - t_start:.1f}s.")
