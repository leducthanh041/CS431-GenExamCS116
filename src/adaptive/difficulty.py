"""
src/adaptive/difficulty.py — Difficulty Controller
====================================================
Adapts question difficulty (G1/G2/G3) based on student performance.

Functions:
    get_adaptive_difficulty(topic_id, profile, topic_list) → str
    update_topic_difficulty(topic_id, profile, is_correct) → StudentProfile
    get_difficulty_display_name(difficulty) → str
"""

from __future__ import annotations

from typing import Any

from .profile import StudentProfile, load_topic_list

# ─── Difficulty levels ────────────────────────────────────────────────────────

DIFFICULTY_LEVELS = ["G1", "G2", "G3"]

# ─── Display names ────────────────────────────────────────────────────────────

DIFFICULTY_DISPLAY = {
    "G1": "G1 – Nhớ/Hiểu",
    "G2": "G2 – Áp dụng/Phân tích",
    "G3": "G3 – Đánh giá/Sáng tạo",
    "G1–G2": "G1–G2 – Nhớ đến Áp dụng",
    "G2–G3": "G2–G3 – Áp dụng đến Sáng tạo",
}


def get_difficulty_display_name(difficulty: str) -> str:
    """Return human-readable display name for a difficulty level."""
    return DIFFICULTY_DISPLAY.get(difficulty, difficulty)


# ─── Guess difficulty from topic_list ───────────────────────────────────────

def _guess_difficulty_from_topic(
    topic_id: str,
    topic_list: list[dict[str, Any]],
) -> str:
    """Guess initial difficulty for a topic from topic_list.json."""
    for chapter in topic_list:
        for topic in chapter.get("topics", []):
            if topic.get("topic_id") == topic_id:
                return topic.get("difficulty", "G2")
    return "G2"


# ─── Core functions ─────────────────────────────────────────────────────────

def get_adaptive_difficulty(
    topic_id: str,
    profile: StudentProfile,
    topic_list: list[dict[str, Any]] | None = None,
) -> str:
    """Determine the optimal difficulty for the next question on a topic.

    Logic:
        - No attempts → guess from topic_list.json (base difficulty)
        - accuracy < 0.3  → G1  (build confidence)
        - accuracy 0.3–0.5 → reduce by 1 level (if not already G1)
        - accuracy >= 0.85 → increase by 1 level (max G3)
        - others          → keep current difficulty

    Returns:
        "G1" | "G2" | "G3"
    """
    if topic_list is None:
        topic_list = load_topic_list()

    stats = profile.topic_stats.get(topic_id)

    if stats is None or stats.attempts == 0:
        return _guess_difficulty_from_topic(topic_id, topic_list)

    accuracy = stats.accuracy
    mastery = stats.mastery_level
    current = stats.current_difficulty

    if accuracy < 0.3:
        return "G1"
    elif accuracy < 0.5:
        idx = DIFFICULTY_LEVELS.index(current) if current in DIFFICULTY_LEVELS else 1
        return DIFFICULTY_LEVELS[max(0, idx - 1)]
    elif accuracy >= 0.85:
        if mastery >= 3:
            idx = DIFFICULTY_LEVELS.index(current) if current in DIFFICULTY_LEVELS else 1
            return DIFFICULTY_LEVELS[min(2, idx + 1)]
        return current
    else:
        return current


def update_topic_difficulty(
    topic_id: str,
    profile: StudentProfile,
    is_correct: bool,
) -> StudentProfile:
    """Apply difficulty adjustment rule after each interaction.

    Updates current_difficulty and difficulty_history in the profile.
    Logic:
        - Correct + mastery >= 3 → increase level (if not G3)
        - Incorrect              → decrease level (if not G1)

    Returns:
        Updated StudentProfile (profile is also saved to disk)
    """
    from .profile import save_profile

    stats = profile.topic_stats.get(topic_id)
    if stats is None:
        return profile

    current_idx = (
        DIFFICULTY_LEVELS.index(stats.current_difficulty)
        if stats.current_difficulty in DIFFICULTY_LEVELS else 1
    )

    if is_correct:
        if stats.mastery_level >= 3 and current_idx < 2:
            new_diff = DIFFICULTY_LEVELS[current_idx + 1]
        else:
            new_diff = stats.current_difficulty
        stats.mastery_level = min(5, stats.mastery_level + 1)
    else:
        if current_idx > 0:
            new_diff = DIFFICULTY_LEVELS[current_idx - 1]
        else:
            new_diff = stats.current_difficulty
        stats.mastery_level = max(0, stats.mastery_level - 1)

    stats.difficulty_history.append({
        "step": stats.attempts,
        "difficulty": new_diff,
        "is_correct": is_correct,
    })
    stats.current_difficulty = new_diff

    save_profile(profile)
    return profile
