import os
import json
import re
import time
import threading
import argparse
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
PROMPT_PATH = BASE_DIR / "prompts" / "socratic_generate.txt"
MISCONCEPTIONS_DIR = BASE_DIR / "data" / "misconceptions"
QUESTION_BANK_DIR = BASE_DIR / "input" / "question_bank" / "multiple_choice"
OUTPUT_DIR = BASE_DIR / "data" / "dialogues"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Load prompt template
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
MAX_RETRIES = 3
RETRY_BASE_DELAY = 10  # seconds

def call_model_with_retry(prompt: str, item_id: str) -> str:
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
            wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            log(f"RETRY  {item_id} — 429 rate limit, waiting {wait}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt == MAX_RETRIES:
                    raise
                wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                log(f"RETRY  {item_id} — 429 rate limit, waiting {wait}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise

# =========================
# Validate and normalize dialogue output
# =========================
def _extract_dialogue_list(parsed) -> list | None:
    """
    Extract the dialogue list from parsed JSON.
    Handles:
    - {"dialogue": [...]}
        ← expected format
    - [{"role": .., "text": ..}, ...]
        ← model returned flat list of turns
    - [{"dialogue_id": .., "dialogue": [...]}, ...]
        ← model returned multiple variations; take the first
    """
    # Case 1: {"dialogue": [...]}
    if isinstance(parsed, dict) and "dialogue" in parsed:
        d = parsed["dialogue"]
        return d if isinstance(d, list) else None

    if isinstance(parsed, list) and parsed:
        first = parsed[0]
        # Case 2: list of variation objects — each item has its own "dialogue"
        if isinstance(first, dict) and "dialogue" in first:
            d = first["dialogue"]
            return d if isinstance(d, list) else None
        # Case 3: flat list of turn objects [{"role": .., "text": ..}, ...]
        return parsed

    return None


def is_valid_dialogue(data) -> bool:
    dialogue = _extract_dialogue_list(data)
    if dialogue is None:
        return False
    return len(dialogue) >= 6


# =========================
# Generate dialogue for one question item
# =========================
def generate_dialogue_for_item(item_data: dict, item_id: str):
    input_payload = {
        "question_text": item_data["question_text"],
        "correct_answer": item_data["correct_answer"],
        "student_reasoning": item_data["student_reasoning"],
        "misconception": item_data["misconception"],
    }
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Input:\n{json.dumps(input_payload, ensure_ascii=False, indent=2)}"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = call_model_with_retry(prompt, item_id)
            text = strip_markdown_json(raw)
            parsed = json.loads(text)
            dialogue = _extract_dialogue_list(parsed)
            if dialogue is not None and len(dialogue) >= 6:
                return {"dialogue": dialogue}
            turns = len(dialogue) if dialogue is not None else f"parse_ok/bad_shape:{type(parsed).__name__}"
            log(f"INVALID {item_id} — {turns} turns < 6 (attempt {attempt}/{MAX_RETRIES}), retrying...")
            if attempt == 1:
                sample = json.dumps(dialogue[:2] if dialogue else parsed, ensure_ascii=False)
                log(f"         sample: {sample[:300]}")
        except json.JSONDecodeError as e:
            log(f"JSON_ERR {item_id} — parse failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            log(f"         raw snippet: {raw[:200]!r}")
        except Exception as e:
            log(f"ERR {item_id} — {e} (attempt {attempt}/{MAX_RETRIES})")
            if attempt == MAX_RETRIES:
                raise

    return None

# =========================
# Build input items from misconceptions + question bank
# =========================

def _normalize_qbank(raw) -> dict:
    """
    Normalize question bank into a lookup dict: question_id (str) -> {question_text, correct_answer}.

    Handles two formats:
    - Old (tuan1): list of {question_number, question_title, answers:[{answer_text, is_correct}]}
    - New (tuan2+): {GENERATED_QUESTIONS: [{question_id, question_text, options:{A:..}, correct_answer:[..]}]}
    """
    lookup = {}

    # New format: dict with GENERATED_QUESTIONS key
    if isinstance(raw, dict):
        questions = raw.get("GENERATED_QUESTIONS", [])
        for q in questions:
            qid = str(q["question_id"])
            # correct_answer is a list of option keys e.g. ["B"]
            correct_keys = q.get("correct_answer", [])
            options = q.get("options", {})
            correct_text = options.get(correct_keys[0], "") if correct_keys else ""
            lookup[qid] = {
                "question_text": q.get("question_text", ""),
                "correct_answer": correct_text,
            }
        return lookup

    # Old format: plain list
    if isinstance(raw, list):
        for q in raw:
            # Support both question_number (int) and question_id (str)
            if "question_id" in q:
                qid = str(q["question_id"])
            else:
                qid = str(q["question_number"])

            if "answers" in q:
                correct_answers = [a["answer_text"] for a in q["answers"] if a["is_correct"]]
                correct_answer = correct_answers[0] if correct_answers else ""
            elif "options" in q and "correct_answer" in q:
                correct_keys = q["correct_answer"]
                correct_answer = q["options"].get(correct_keys[0], "") if correct_keys else ""
            else:
                correct_answer = ""

            question_text = q.get("question_title") or q.get("question_text", "")
            lookup[qid] = {
                "question_text": question_text,
                "correct_answer": correct_answer,
            }
        return lookup

    return lookup


def build_items_for_file(misconceptions_path: Path) -> list:
    with open(misconceptions_path, "r", encoding="utf-8") as f:
        misconceptions = json.load(f)

    stem = misconceptions_path.stem  # e.g., "tuan1"
    qbank_path = QUESTION_BANK_DIR / f"{stem}.json"

    if not qbank_path.exists():
        log(f"WARN  No question bank for {stem}, skipping file")
        return []

    with open(qbank_path, "r", encoding="utf-8") as f:
        qbank_lookup = _normalize_qbank(json.load(f))

    items = []
    for m in misconceptions:
        qid = str(m["question_id"])
        q = qbank_lookup.get(qid)
        if not q:
            log(f"WARN  {stem}/{qid} not in question bank, skipping")
            continue

        misconceptions_list = m.get("misconceptions", [])
        student_reasoning = "; ".join(misconceptions_list) if misconceptions_list else ""
        primary_misconception = misconceptions_list[0] if misconceptions_list else ""

        items.append({
            "question_id": qid,
            "question_text": q["question_text"],
            "correct_answer": q["correct_answer"],
            "student_reasoning": student_reasoning,
            "misconception": primary_misconception,
        })

    return items

# =========================
# Global progress counters (thread-safe)
# =========================
_q_ok = 0
_q_fail = 0
_q_total = 0  # set after counting all items

def _update_progress(ok_delta: int = 0, fail_delta: int = 0):
    global _q_ok, _q_fail
    with _log_lock:
        _q_ok += ok_delta
        _q_fail += fail_delta
        done = _q_ok + _q_fail
        pct = done / _q_total * 100 if _q_total else 0
        print(
            f"  [PROGRESS] {done}/{_q_total} questions ({pct:.1f}%) "
            f"— ok: {_q_ok}, fail: {_q_fail}",
            flush=True,
        )

# =========================
# Process one misconceptions file
# =========================
def process_file(misconceptions_path: Path, limit: int | None = None):
    t0 = time.time()
    stem = misconceptions_path.stem
    limit_tag = f" (limit={limit})" if limit else ""
    log(f"START  {stem}{limit_tag}")

    items = build_items_for_file(misconceptions_path)
    if limit:
        items = items[:limit]

    if not items:
        log(f"SKIP   {stem} — no items to process")
        return

    log(f"LOADED {stem} ({len(items)} questions)")

    results = []
    failed = 0

    for item in items:
        qid = item["question_id"]
        item_id = f"{stem}/{qid}"
        log(f"GEN    {item_id}")

        dialogue_data = generate_dialogue_for_item(item, item_id)

        if dialogue_data is not None:
            results.append({
                "question_id": qid,
                "dialogue": dialogue_data["dialogue"],
            })
            log(f"OK     {item_id} ({len(dialogue_data['dialogue'])} turns)")
            _update_progress(ok_delta=1)
        else:
            failed += 1
            log(f"FAIL   {item_id} — skipped after validation retries")
            _update_progress(fail_delta=1)

    output_file = OUTPUT_DIR / f"{stem}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    log(
        f"DONE   {stem} — {len(results)} ok, {failed} failed "
        f"→ {output_file.relative_to(BASE_DIR)} ({elapsed:.1f}s)"
    )

# =========================
# CLI arguments
# =========================
parser = argparse.ArgumentParser(description="Generate Socratic dialogues via Claude on Vertex AI")
parser.add_argument("--file",    type=str, default=None, help="Process a single file, e.g. tuan1")
parser.add_argument("--limit",   type=int, default=None, help="Max questions per file (for testing)")
parser.add_argument("--workers", type=int, default=5,    help="Number of parallel threads (default: 5)")
args = parser.parse_args()

# =========================
# Resolve file list
# =========================
if args.file:
    target = MISCONCEPTIONS_DIR / f"{args.file}.json"
    if not target.exists():
        raise FileNotFoundError(f"File not found: {target}")
    files = [target]
else:
    files = sorted(MISCONCEPTIONS_DIR.glob("*.json"))

if not files:
    raise ValueError(f"No JSON files found in {MISCONCEPTIONS_DIR}")

# Pre-count total questions for progress display
_q_total = sum(
    len(build_items_for_file(f)[: args.limit] if args.limit else build_items_for_file(f))
    for f in files
)

MAX_WORKERS = args.workers if not args.file else 1
print(
    f"Files: {len(files)} | Questions: {_q_total} | Workers: {MAX_WORKERS} | Model: {MODEL_NAME}\n",
    flush=True,
)

t_start = time.time()
files_done = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="worker") as executor:
    futures = {executor.submit(process_file, f, args.limit): f for f in files}
    for future in as_completed(futures):
        files_done += 1
        fname = futures[future].stem
        try:
            future.result()
        except Exception as e:
            with _log_lock:
                print(f"  [ERROR] {fname}: {e}", flush=True)
        with _log_lock:
            print(f"  [FILES]  {files_done}/{len(files)} files completed", flush=True)

elapsed_total = time.time() - t_start
print(f"\nAll done in {elapsed_total:.1f}s — {_q_ok} ok, {_q_fail} failed out of {_q_total} questions.")
