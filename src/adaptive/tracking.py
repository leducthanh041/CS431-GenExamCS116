"""
src/adaptive/tracking.py — Interaction Recording & Accuracy
==========================================================
Records student quiz interactions and recalculates accuracy metrics.

Functions:
    record_interaction(...)   → tuple[bool, StudentProfile]
    get_topic_accuracy(...)   → float
    get_chapter_accuracy(...)→ float
    get_overall_accuracy(...) → float
"""

from __future__ import annotations

from . import db
from .profile import load_profile, save_profile, StudentProfile, load_topic_list


def record_interaction(
    user_id: str,
    question_id: str,
    chapter_id: str,
    topic_id: str,
    topic_name: str,
    difficulty_label: str,
    difficulty_score: int,
    user_answer: list[str],
    correct_answers: list[str],
    time_spent_seconds: int,
) -> tuple[bool, StudentProfile]:
    """Record a single question interaction.

    1. Appends to interactions.jsonl
    2. Updates profile topic_stats[topic_id]
    3. Recalculates accuracy
    4. Saves profile

    Args:
        user_id: Student identifier
        question_id: Unique question ID
        chapter_id: Chapter identifier (e.g. "ch04")
        topic_id: Topic identifier (e.g. "ch04_t01")
        topic_name: Human-readable topic name
        difficulty_label: "G1" | "G2" | "G3"
        difficulty_score: Numeric score (1=G1, 2=G2, 3=G3)
        user_answer: List of selected option letters (e.g. ["A"])
        correct_answers: List of correct option letters
        time_spent_seconds: Time taken to answer

    Returns:
        (is_correct: bool, updated_profile: StudentProfile)
    """
    # ── Determine correctness ─────────────────────────────────────────────
    user_set = set(u.upper() for u in user_answer)
    correct_set = set(c.upper() for c in correct_answers)
    is_correct = user_set == correct_set

    # ── Append to interaction log ───────────────────────────────────────────
    interaction = {
        "interaction_id": db.new_uuid(),
        "user_id": user_id,
        "question_id": question_id,
        "chapter_id": chapter_id,
        "topic_id": topic_id,
        "topic_name": topic_name,
        "difficulty_label": difficulty_label,
        "difficulty_score": difficulty_score,
        "user_answer": user_answer,
        "correct_answers": correct_answers,
        "is_correct": is_correct,
        "correct_answer_count": len(correct_answers),
        "answered_at": db.now_iso(),
        "time_spent_seconds": time_spent_seconds,
    }
    interactions_path = db.get_interactions_path(user_id)
    db.append_jsonl(interactions_path, interaction)

    # ── Update profile ──────────────────────────────────────────────────────
    profile = load_profile(user_id)
    ts = profile.topic_stats.get(topic_id)
    if ts is None:
        from .profile import TopicStats

        ts = TopicStats(
            topic_id=topic_id,
            topic_name=topic_name,
            chapter_id=chapter_id,
            current_difficulty=difficulty_label,
        )
        profile.topic_stats[topic_id] = ts

    # Update stats
    ts.attempts += 1
    if is_correct:
        ts.correct += 1
    ts.accuracy = ts.correct / ts.attempts if ts.attempts > 0 else 0.0
    ts.last_attempted_at = db.now_iso()

    # Update average time (cumulative moving average)
    total_time = ts.avg_time_spent * (ts.attempts - 1) + time_spent_seconds
    ts.avg_time_spent = total_time / ts.attempts

    # Update difficulty history
    ts.difficulty_history.append({
        "step": ts.attempts,
        "difficulty": difficulty_label,
        "is_correct": is_correct,
    })

    # ── Persist is_weak flag ───────────────────────────────────────────────
    was_weak = ts.is_weak
    ts.is_weak = ts.accuracy < 0.5
    if ts.is_weak and not was_weak:
        ts.weakness_detected_at = db.now_iso()

    # ── Recalculate chapter stats ───────────────────────────────────────────
    _recalculate_chapter_stats(profile, chapter_id)

    # ── Recalculate overall stats ──────────────────────────────────────────
    _recalculate_overall_stats(profile)

    save_profile(profile)
    return is_correct, profile


def _recalculate_chapter_stats(profile: StudentProfile, chapter_id: str) -> None:
    """Recalculate stats for a chapter based on its topics."""
    from .profile import ChapterStats

    chapter_topics = [
        ts for ts in profile.topic_stats.values() if ts.chapter_id == chapter_id
    ]
    if not chapter_topics:
        return

    attempted = [ts for ts in chapter_topics if ts.attempts > 0]
    if not attempted:
        return

    total_correct = sum(ts.correct for ts in attempted)
    total_attempts = sum(ts.attempts for ts in attempted)
    weak = [ts for ts in attempted if ts.is_weak]
    mastered = [ts for ts in attempted if ts.accuracy >= 0.85]

    chapter_name = ""
    for ch in load_topic_list():
        if ch.get("chapter_id") == chapter_id:
            chapter_name = ch.get("chapter_name", "")
            break

    cs = ChapterStats(
        chapter_id=chapter_id,
        chapter_name=chapter_name,
        topics_attempted=len(attempted),
        topics_mastered=len(mastered),
        topics_weak=len(weak),
        overall_accuracy=(
            total_correct / total_attempts if total_attempts > 0 else 0.0
        ),
        weak_topics=[ts.topic_id for ts in weak],
    )
    profile.chapter_stats[chapter_id] = cs


def _recalculate_overall_stats(profile: StudentProfile) -> None:
    """Recalculate overall statistics across all topics."""
    from .profile import OverallStats

    all_attempted = [ts for ts in profile.topic_stats.values() if ts.attempts > 0]
    if not all_attempted:
        return

    total_correct = sum(ts.correct for ts in all_attempted)
    total_attempts = sum(ts.attempts for ts in all_attempted)
    weak_count = sum(1 for ts in all_attempted if ts.is_weak)
    strong_count = sum(1 for ts in all_attempted if ts.accuracy >= 0.85)

    mastery_score = (
        sum(ts.accuracy for ts in all_attempted) / len(all_attempted)
        if all_attempted else 0.0
    )

    profile.overall_stats = OverallStats(
        total_attempts=total_attempts,
        total_correct=total_correct,
        overall_accuracy=(
            total_correct / total_attempts if total_attempts > 0 else 0.0
        ),
        weak_topics_count=weak_count,
        strong_topics_count=strong_count,
        mastery_score=mastery_score,
        last_session_at=db.now_iso(),
    )


# ─── Accuracy query functions ───────────────────────────────────────────────

def get_topic_accuracy(user_id: str, topic_id: str) -> float:
    """Return accuracy for a topic (0.0–1.0). Returns 0.0 if no attempts."""
    profile = load_profile(user_id)
    ts = profile.topic_stats.get(topic_id)
    if ts is None or ts.attempts == 0:
        return 0.0
    return ts.accuracy


def get_chapter_accuracy(user_id: str, chapter_id: str) -> float:
    """Return aggregated accuracy for a chapter (0.0–1.0)."""
    profile = load_profile(user_id)
    chapter_topics = [
        ts for ts in profile.topic_stats.values()
        if ts.chapter_id == chapter_id and ts.attempts > 0
    ]
    if not chapter_topics:
        return 0.0
    total_correct = sum(ts.correct for ts in chapter_topics)
    total_attempts = sum(ts.attempts for ts in chapter_topics)
    return total_correct / total_attempts if total_attempts > 0 else 0.0


def get_overall_accuracy(user_id: str) -> float:
    """Return overall accuracy across all topics (0.0–1.0)."""
    profile = load_profile(user_id)
    all_attempted = [ts for ts in profile.topic_stats.values() if ts.attempts > 0]
    if not all_attempted:
        return 0.0
    total_correct = sum(ts.correct for ts in all_attempted)
    total_attempts = sum(ts.attempts for ts in all_attempted)
    return total_correct / total_attempts if total_attempts > 0 else 0.0
