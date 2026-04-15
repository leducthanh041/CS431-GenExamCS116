"""
src/adaptive/profile.py — Student Profile Management
====================================================
Manages student profiles: topic stats, chapter stats, overall stats.
All data stored as JSON files (no database required).

@dataclass StudentProfile:
    user_id: str
    topic_stats: dict[str, TopicStats]   # topic_id → stats
    chapter_stats: dict[str, ChapterStats]
    overall_stats: OverallStats
    study_history: list[dict]

Functions:
    init_profile(user_id)   → StudentProfile (new student)
    load_profile(user_id)   → StudentProfile (from disk, or new)
    save_profile(profile)   → None (to disk)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import db

# ─── Dataclasses ─────────────────────────────────────────────────────────────


@dataclass
class TopicStats:
    """Statistics for a single topic."""
    topic_id: str
    topic_name: str
    chapter_id: str
    attempts: int = 0
    correct: int = 0
    accuracy: float = 0.0
    avg_time_spent: float = 0.0
    current_difficulty: str = "G1"
    mastery_level: int = 0
    is_weak: bool = False
    weakness_detected_at: str | None = None
    last_attempted_at: str | None = None
    difficulty_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "topic_name": self.topic_name,
            "chapter_id": self.chapter_id,
            "attempts": self.attempts,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "avg_time_spent": self.avg_time_spent,
            "current_difficulty": self.current_difficulty,
            "mastery_level": self.mastery_level,
            "is_weak": self.is_weak,
            "weakness_detected_at": self.weakness_detected_at,
            "last_attempted_at": self.last_attempted_at,
            "difficulty_history": self.difficulty_history,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TopicStats":
        return cls(
            topic_id=d.get("topic_id", ""),
            topic_name=d.get("topic_name", ""),
            chapter_id=d.get("chapter_id", ""),
            attempts=d.get("attempts", 0),
            correct=d.get("correct", 0),
            accuracy=d.get("accuracy", 0.0),
            avg_time_spent=d.get("avg_time_spent", 0.0),
            current_difficulty=d.get("current_difficulty", "G1"),
            mastery_level=d.get("mastery_level", 0),
            is_weak=d.get("is_weak", False),
            weakness_detected_at=d.get("weakness_detected_at"),
            last_attempted_at=d.get("last_attempted_at"),
            difficulty_history=d.get("difficulty_history", []),
        )


@dataclass
class ChapterStats:
    """Statistics for a chapter (aggregated from topics)."""
    chapter_id: str
    chapter_name: str
    topics_attempted: int = 0
    topics_mastered: int = 0
    topics_weak: int = 0
    overall_accuracy: float = 0.0
    weak_topics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter_id": self.chapter_id,
            "chapter_name": self.chapter_name,
            "topics_attempted": self.topics_attempted,
            "topics_mastered": self.topics_mastered,
            "topics_weak": self.topics_weak,
            "overall_accuracy": self.overall_accuracy,
            "weak_topics": self.weak_topics,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChapterStats":
        return cls(
            chapter_id=d.get("chapter_id", ""),
            chapter_name=d.get("chapter_name", ""),
            topics_attempted=d.get("topics_attempted", 0),
            topics_mastered=d.get("topics_mastered", 0),
            topics_weak=d.get("topics_weak", 0),
            overall_accuracy=d.get("overall_accuracy", 0.0),
            weak_topics=d.get("weak_topics", []),
        )


@dataclass
class OverallStats:
    """Overall statistics across all topics."""
    total_attempts: int = 0
    total_correct: int = 0
    overall_accuracy: float = 0.0
    weak_topics_count: int = 0
    strong_topics_count: int = 0
    mastery_score: float = 0.0
    last_session_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_attempts": self.total_attempts,
            "total_correct": self.total_correct,
            "overall_accuracy": self.overall_accuracy,
            "weak_topics_count": self.weak_topics_count,
            "strong_topics_count": self.strong_topics_count,
            "mastery_score": self.mastery_score,
            "last_session_at": self.last_session_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OverallStats":
        return cls(
            total_attempts=d.get("total_attempts", 0),
            total_correct=d.get("total_correct", 0),
            overall_accuracy=d.get("overall_accuracy", 0.0),
            weak_topics_count=d.get("weak_topics_count", 0),
            strong_topics_count=d.get("strong_topics_count", 0),
            mastery_score=d.get("mastery_score", 0.0),
            last_session_at=d.get("last_session_at"),
        )


@dataclass
class StudentProfile:
    """Complete student profile."""
    user_id: str
    topic_stats: dict[str, TopicStats] = field(default_factory=dict)
    chapter_stats: dict[str, ChapterStats] = field(default_factory=dict)
    overall_stats: OverallStats = field(default_factory=OverallStats)
    study_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "topic_stats": {
                tid: ts.to_dict() for tid, ts in self.topic_stats.items()
            },
            "chapter_stats": {
                cid: cs.to_dict() for cid, cs in self.chapter_stats.items()
            },
            "overall_stats": self.overall_stats.to_dict(),
            "study_history": self.study_history,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StudentProfile":
        topic_stats = {
            tid: TopicStats.from_dict(v)
            for tid, v in d.get("topic_stats", {}).items()
        }
        chapter_stats = {
            cid: ChapterStats.from_dict(v)
            for cid, v in d.get("chapter_stats", {}).items()
        }
        overall_stats = OverallStats.from_dict(d.get("overall_stats", {}))
        return cls(
            user_id=d.get("user_id", ""),
            topic_stats=topic_stats,
            chapter_stats=chapter_stats,
            overall_stats=overall_stats,
            study_history=d.get("study_history", []),
        )


# ─── Profile I/O ─────────────────────────────────────────────────────────────


def load_topic_list() -> list[dict[str, Any]]:
    """Load topic list from JSON file."""
    root = db._get_project_root()
    path = root / "input" / "topic_list.json"
    data = db.load_json(path)
    if data is None:
        return []
    if isinstance(data, list):
        return data
    return []


def _build_empty_stats_from_topics() -> (
    tuple[dict[str, TopicStats], dict[str, ChapterStats]]
):
    """Build empty topic + chapter stats from topic_list.json."""
    topic_stats: dict[str, TopicStats] = {}
    chapter_stats: dict[str, ChapterStats] = {}
    topic_list = load_topic_list()

    for chapter in topic_list:
        ch_id = chapter.get("chapter_id", "")
        ch_name = chapter.get("chapter_name", "")
        chapter_stats[ch_id] = ChapterStats(
            chapter_id=ch_id,
            chapter_name=ch_name,
        )
        for topic in chapter.get("topics", []):
            t_id = topic.get("topic_id", "")
            t_name = topic.get("topic_name", "")
            diff = topic.get("difficulty", "G1")
            topic_stats[t_id] = TopicStats(
                topic_id=t_id,
                topic_name=t_name,
                chapter_id=ch_id,
                current_difficulty=diff,
            )

    return topic_stats, chapter_stats


def init_profile(user_id: str) -> StudentProfile:
    """Create a brand-new profile for a student (all topics initialized)."""
    db.ensure_student_dir(user_id)
    topic_stats, chapter_stats = _build_empty_stats_from_topics()

    profile = StudentProfile(
        user_id=user_id,
        topic_stats=topic_stats,
        chapter_stats=chapter_stats,
        overall_stats=OverallStats(),
        study_history=[],
    )
    save_profile(profile)
    return profile


def load_profile(user_id: str) -> StudentProfile:
    """Load student profile from disk.

    If no profile exists, creates a new one with all topics initialized
    from topic_list.json.
    """
    profile_path = db.get_profile_path(user_id)
    data = db.load_json(profile_path)
    if data is None:
        return init_profile(user_id)
    try:
        return StudentProfile.from_dict(data)
    except Exception:
        return init_profile(user_id)


def save_profile(profile: StudentProfile) -> None:
    """Save student profile to disk."""
    db.ensure_student_dir(profile.user_id)
    profile_path = db.get_profile_path(profile.user_id)
    db.save_json(profile_path, profile.to_dict())
