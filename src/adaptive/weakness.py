"""
src/adaptive/weakness.py — Weakness Detection
============================================
Detects weak topics based on accuracy thresholds.

Constants:
    WEAK_THRESHOLD = 0.5     # accuracy < 50% → weak
    CRITICAL_THRESHOLD = 0.3  # accuracy < 30% → critical
    MIN_ATTEMPTS = 2         # min attempts before marking as weak

Functions:
    detect_weak_topics(user_id) → list[TopicWeakness]
    get_weakness_report(user_id) → WeaknessReport
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import db
from .profile import load_profile, load_topic_list, StudentProfile

# ─── Thresholds ───────────────────────────────────────────────────────────────

WEAK_THRESHOLD = 0.5
CRITICAL_THRESHOLD = 0.3
MIN_ATTEMPTS = 2

# ─── Dataclasses ─────────────────────────────────────────────────────────────


@dataclass
class TopicWeakness:
    """A weak topic with priority classification."""
    topic_id: str
    topic_name: str
    chapter_id: str
    chapter_name: str
    accuracy: float
    attempts: int
    priority: str  # "critical" | "medium" | "low"
    suggested_difficulty: str


@dataclass
class WeaknessReport:
    """Full weakness report for a student."""
    user_id: str
    weak_topics: list[TopicWeakness] = field(default_factory=list)
    weak_by_chapter: dict[str, list[TopicWeakness]] = field(default_factory=dict)
    total_weak: int = 0
    total_critical: int = 0
    total_medium: int = 0
    total_low: int = 0
    generated_at: str = ""


# ─── Helper ───────────────────────────────────────────────────────────────────

def _get_topic_name(topic_id: str) -> tuple[str, str]:
    """Return (topic_name, chapter_name) from topic_list.json."""
    for chapter in load_topic_list():
        for topic in chapter.get("topics", []):
            if topic.get("topic_id") == topic_id:
                return (
                    topic.get("topic_name", topic_id),
                    chapter.get("chapter_name", ""),
                )
    return topic_id, ""


def _classify_priority(accuracy: float) -> str:
    """Classify weakness priority based on accuracy."""
    if accuracy < CRITICAL_THRESHOLD:
        return "critical"
    elif accuracy < WEAK_THRESHOLD:
        return "medium"
    return "low"


# ─── Main functions ───────────────────────────────────────────────────────────

def detect_weak_topics(
    user_id: str,
    threshold: float = WEAK_THRESHOLD,
    min_attempts: int = MIN_ATTEMPTS,
) -> list[TopicWeakness]:
    """Detect all weak topics for a student.

    Args:
        user_id: Student identifier
        threshold: Accuracy threshold (default WEAK_THRESHOLD = 0.5)
        min_attempts: Minimum attempts before declaring weakness

    Returns:
        Sorted list of TopicWeakness (critical → medium → low)
    """
    profile = load_profile(user_id)
    weak_list: list[TopicWeakness] = []

    for topic_id, ts in profile.topic_stats.items():
        if ts.attempts < min_attempts:
            continue
        if ts.accuracy < threshold:
            priority = _classify_priority(ts.accuracy)
            topic_name, chapter_name = _get_topic_name(topic_id)

            suggested_diff = ts.current_difficulty
            if ts.accuracy < CRITICAL_THRESHOLD:
                suggested_diff = "G1"
            else:
                levels = ["G1", "G2", "G3"]
                idx = levels.index(suggested_diff) if suggested_diff in levels else 1
                suggested_diff = levels[max(0, idx - 1)]

            weak_list.append(TopicWeakness(
                topic_id=topic_id,
                topic_name=topic_name,
                chapter_id=ts.chapter_id,
                chapter_name=chapter_name,
                accuracy=ts.accuracy,
                attempts=ts.attempts,
                priority=priority,
                suggested_difficulty=suggested_diff,
            ))

    priority_order = {"critical": 0, "medium": 1, "low": 2}
    weak_list.sort(key=lambda w: (priority_order.get(w.priority, 3), w.accuracy))
    return weak_list


def get_weakness_report(user_id: str) -> WeaknessReport:
    """Generate a full weakness report for a student.

    Groups weak topics by chapter and provides summary statistics.
    """
    weak_list = detect_weak_topics(user_id)
    weak_by_chapter: dict[str, list[TopicWeakness]] = {}

    for w in weak_list:
        if w.chapter_id not in weak_by_chapter:
            weak_by_chapter[w.chapter_id] = []
        weak_by_chapter[w.chapter_id].append(w)

    return WeaknessReport(
        user_id=user_id,
        weak_topics=weak_list,
        weak_by_chapter=weak_by_chapter,
        total_weak=len(weak_list),
        total_critical=sum(1 for w in weak_list if w.priority == "critical"),
        total_medium=sum(1 for w in weak_list if w.priority == "medium"),
        total_low=sum(1 for w in weak_list if w.priority == "low"),
        generated_at=db.now_iso(),
    )
