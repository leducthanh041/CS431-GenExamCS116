import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
MODEL_NAME = os.getenv("GEMINI_MODEL")

RAW_DIR = "data/raw_transcript"
CLEAN_DIR = "data/cleaned_transcript"