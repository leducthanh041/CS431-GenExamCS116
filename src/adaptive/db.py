"""
src/adaptive/db.py — JSON File-Based Storage Utilities
====================================================
Provides simple file-based storage for student data.
No external database required — portable and simple.

Functions:
    ensure_student_dir(user_id)  → Path
    load_json / save_json        → JSON read/write
    append_jsonl / load_jsonl    → JSONL append-only / read
    load_mcq_pool()              → Load accepted questions pool
    _student_dir(user_id)        → Path
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

_PROJECT_ROOT: Path | None = None


def _get_project_root() -> Path:
    """Get PROJECT_ROOT lazily — avoids importing common.py at module level."""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        try:
            from src.common import Config
            _PROJECT_ROOT = Config.PROJECT_ROOT
        except Exception:
            _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    return _PROJECT_ROOT


def _student_dir(user_id: str) -> Path:
    """Return the data directory path for a student."""
    return _get_project_root() / "data" / "students" / user_id


# ─── Directory management ─────────────────────────────────────────────────────

def ensure_student_dir(user_id: str) -> Path:
    """Create and return the data directory for user_id.

    Creates:
        data/students/{user_id}/
            profile.json
            interactions.jsonl   (append-only)
            sessions/
                {session_id}.json
    """
    student_dir = _student_dir(user_id)
    student_dir.mkdir(parents=True, exist_ok=True)
    (student_dir / "sessions").mkdir(exist_ok=True)
    return student_dir


def get_profile_path(user_id: str) -> Path:
    """Return path to student's profile.json."""
    return _student_dir(user_id) / "profile.json"


def get_interactions_path(user_id: str) -> Path:
    """Return path to student's interactions.jsonl."""
    return _student_dir(user_id) / "interactions.jsonl"


def get_sessions_dir(user_id: str) -> Path:
    """Return path to student's sessions/ directory."""
    return _student_dir(user_id) / "sessions"


def get_session_path(user_id: str, session_id: str) -> Path:
    """Return path to a specific session JSON file."""
    return get_sessions_dir(user_id) / f"{session_id}.json"


# ─── JSON helpers ────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file. Returns None if file doesn't exist."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Write data to a JSON file (creates parent dirs if needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load all records from a .jsonl file (one JSON object per line)."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append a single record to a .jsonl file (append-only log)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Overwrite a .jsonl file with all records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ─── MCQ Pool helpers ─────────────────────────────────────────────────────────

def get_mcq_pool_path() -> Path | None:
    """Find the latest accepted_questions.jsonl from completed experiments.

    Priority:
        1. output/{EXP_NAME}/08_eval_iwf/final_accepted_questions.jsonl
        2. exp_04_full_v2 / exp_03_test_15q / ... (fallback)
        3. Most recent experiment by mtime
    """
    root = _get_project_root()

    # Try active EXP_NAME first
    try:
        from src.common import Config
        exp_name = Config.EXP_NAME
        candidate = root / "output" / exp_name / "08_eval_iwf" / "final_accepted_questions.jsonl"
        if candidate.exists():
            return candidate
    except Exception:
        pass

    # Known experiment names (priority order)
    candidates = [
        root / "output" / "exp_04_full_v2" / "08_eval_iwf" / "final_accepted_questions.jsonl",
        root / "output" / "exp_03_test_15q" / "08_eval_iwf" / "final_accepted_questions.jsonl",
        root / "output" / "exp_02_baseline" / "08_eval_iwf" / "final_accepted_questions.jsonl",
        root / "output" / "exp_01_baseline" / "08_eval_iwf" / "final_accepted_questions.jsonl",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback: find most recent experiment by mtime
    if output_dir := root / "output":
        for exp_dir in sorted(
            output_dir.glob("exp_*/08_eval_iwf/final_accepted_questions.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            return exp_dir

    return None


def load_mcq_pool() -> list[dict[str, Any]]:
    """Load all accepted questions from the MCQ pool.

    Returns an empty list if no pool exists.
    """
    pool_path = get_mcq_pool_path()
    if pool_path is None:
        return []
    return load_jsonl(pool_path)


# ─── Utility ─────────────────────────────────────────────────────────────────

def new_uuid() -> str:
    """Generate a new UUID v4 string."""
    return str(uuid.uuid4())


def now_iso() -> str:
    """Return current time in ISO 8601 format (UTC+7)."""
    tz = timezone(timedelta(hours=7))
    return datetime.now(tz).isoformat()
