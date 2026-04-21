"""
src/adaptive/ — Adaptive MCQ Generation Engine (Phase 2 + 3)
=============================================================
Adaptive loop: tracking → weakness detection → difficulty control → personalized generation.

Modules:
    db          — JSON file-based storage utilities
    profile     — Student profile management
    tracking   — Interaction recording + accuracy calculation
    weakness   — Weakness detection
    difficulty — Difficulty controller (G1/G2/G3)
    recommend  — Adaptive quiz generation + study plan
    quiz       — Quiz session manager (with P9 explanations)
    api        — FastAPI REST endpoints
    on_demand  — MCQGen on-demand generation bridge
"""

from __future__ import annotations

from .profile import (
    StudentProfile,
    TopicStats,
    ChapterStats,
    OverallStats,
    load_profile,
    save_profile,
    init_profile,
    load_topic_list,
)
from .tracking import (
    record_interaction,
    get_topic_accuracy,
    get_chapter_accuracy,
    get_overall_accuracy,
)
from .weakness import (
    TopicWeakness,
    WeaknessReport,
    detect_weak_topics,
    get_weakness_report,
    WEAK_THRESHOLD,
    CRITICAL_THRESHOLD,
    MIN_ATTEMPTS,
)
from .difficulty import (
    get_adaptive_difficulty,
    update_topic_difficulty,
    get_difficulty_display_name,
)
from .recommend import (
    StudyPlan,
    generate_adaptive_quiz,
    get_study_plan,
    load_mcq_pool,
    select_questions_for_topics,
)
from .quiz import (
    QuizSession,
    AnswerResult,
    SessionSummary,
    create_quiz_session,
    submit_answer,
    grade_quiz,
    end_session,
    get_session,
    get_session_path,
)
from .on_demand import (
    PoolStats,
    MissingTopic,
    get_pool_coverage,
    find_missing_for_topics,
    generate_on_demand,
    trigger_pipeline,
    get_coverage_report,
    write_on_demand_config,
    refresh_pool,
)

__all__ = [
    # profile
    "StudentProfile",
    "TopicStats",
    "ChapterStats",
    "OverallStats",
    "load_profile",
    "save_profile",
    "init_profile",
    "load_topic_list",
    # tracking
    "record_interaction",
    "get_topic_accuracy",
    "get_chapter_accuracy",
    "get_overall_accuracy",
    # weakness
    "TopicWeakness",
    "WeaknessReport",
    "detect_weak_topics",
    "get_weakness_report",
    "WEAK_THRESHOLD",
    "CRITICAL_THRESHOLD",
    "MIN_ATTEMPTS",
    # difficulty
    "get_adaptive_difficulty",
    "update_topic_difficulty",
    "get_difficulty_display_name",
    # recommend
    "StudyPlan",
    "generate_adaptive_quiz",
    "get_study_plan",
    "load_mcq_pool",
    "select_questions_for_topics",
    # quiz
    "QuizSession",
    "AnswerResult",
    "SessionSummary",
    "create_quiz_session",
    "submit_answer",
    "grade_quiz",
    "end_session",
    "get_session",
    "get_session_path",
    # on_demand
    "PoolStats",
    "MissingTopic",
    "get_pool_coverage",
    "find_missing_for_topics",
    "generate_on_demand",
    "trigger_pipeline",
    "get_coverage_report",
    "write_on_demand_config",
    "refresh_pool",
]
