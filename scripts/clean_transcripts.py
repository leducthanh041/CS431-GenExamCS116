import json
import os
from pathlib import Path
import re

from dotenv import load_dotenv
from google import genai
from tqdm import tqdm


# =========================
# Load environment variables
# =========================
load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

RAW_DIR = Path("data/raw_transcript")
CLEAN_DIR = Path("data/cleaned_transcript")
PROMPT_PATH = Path("prompts/transcript_cleanup.txt")

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
transcript_files = sorted(RAW_DIR.glob("*.json"))

if len(transcript_files) == 0:
    raise FileNotFoundError(
        f"No transcript files found in: {RAW_DIR}"
    )

print(f"Found {len(transcript_files)} transcript files")

# =========================
# Process each transcript
# =========================
for transcript_path in tqdm(transcript_files):
    try:
        # Load transcript JSON
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

        raw_text = transcript_data.get("text", "").strip()

        if not raw_text:
            print(f"[SKIP] Empty text in {transcript_path.name}")
            continue

        # Build final prompt
        full_prompt = f"""
{CLEANUP_PROMPT}

Transcript:
{raw_text}
"""

        # Call Gemini
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt,
            config={
                "temperature": 0,
                "response_mime_type": "application/json",
            },
        )

        response_text = response.text.strip()

        # Remove markdown wrapper if Gemini returns ```json ... ```
        if response_text.startswith("```json"):
            response_text = response_text[len("```json"):].strip()
            if response_text.endswith("```"):
                response_text = response_text[:-3].strip()
        elif response_text.startswith("```"):
            response_text = response_text[3:].strip()
            if response_text.endswith("```"):
                response_text = response_text[:-3].strip()

        # Parse JSON response
        response_text = response.text.strip()

        # Remove markdown fences if model adds them
        response_text = (
            response_text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        try:
            result = json.loads(response_text)

        except json.JSONDecodeError:
            # Save raw response for debugging
            debug_path = CLEAN_DIR / f"{transcript_path.stem}.raw_response.txt"
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(response_text)

            # Try extracting first JSON object
            match = re.search(r"\{.*\}", response_text, re.DOTALL)

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

        output_path = CLEAN_DIR / f"{transcript_path.stem}.cleaned.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"[DONE] {transcript_path.name} -> {output_path.name}")

    except Exception as e:
        print(f"[ERROR] Failed processing {transcript_path.name}: {e}")