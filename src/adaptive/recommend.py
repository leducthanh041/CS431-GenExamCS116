"""
src/adaptive/recommend.py — Adaptive Quiz Generation & Study Plan
=============================================================
Generates adaptive quizzes based on student performance and detects weak topics.

Functions:
    load_mcq_pool()                    → list[dict] (from output experiments)
    select_questions_for_topics(...)  → list[dict]
    generate_adaptive_quiz(...)        → tuple[list[dict], StudyPlan]
    get_study_plan(user_id)            → StudyPlan
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from . import db
from .difficulty import get_adaptive_difficulty
from .profile import load_profile, load_topic_list, StudentProfile
from .weakness import detect_weak_topics, TopicWeakness

# ─── Dataclass ───────────────────────────────────────────────────────────────


@dataclass
class StudyPlan:
    """Personalized study plan for a student."""
    user_id: str
    weak_topics: list[TopicWeakness] = field(default_factory=list)
    priority_topics: list[TopicWeakness] = field(default_factory=list)
    recommended_difficulties: dict[str, str] = field(default_factory=dict)
    total_questions_needed: int = 0
    notes: list[str] = field(default_factory=list)
    generated_at: str = ""


# ─── MCQ Pool ─────────────────────────────────────────────────────────────────

def load_mcq_pool() -> list[dict[str, Any]]:
    """Load accepted questions from the MCQ pool."""
    return db.load_mcq_pool()


def select_questions_for_topics(
    topic_ids: list[str],
    difficulties: dict[str, str],
    num_questions: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Select questions from MCQ pool for specified topics.

    Args:
        topic_ids: List of topic_ids to draw questions from
        difficulties: Map of topic_id → difficulty label ("G1"|"G2"|"G3")
        num_questions: Total number of questions to select

    Returns:
        (selected_questions: list[dict], missing_topics: list[str])
    """
    pool = load_mcq_pool()
    if not pool:
        return [], topic_ids

    # ── Step 1: Deduplicate by question_id ─────────────────────────────────
    seen_qids: dict[str, dict[str, Any]] = {}
    for q in pool:
        qid = q.get("question_id", "")
        if not qid:
            continue
        score = q.get("evaluation", {}).get("quality_score", 0.5)
        if qid not in seen_qids or score > seen_qids[qid].get("evaluation", {}).get("quality_score", 0):
            seen_qids[qid] = q
    unique_pool = list(seen_qids.values())

    # ── Step 2: Build candidate list ─────────────────────────────────────────
    candidates: list[dict[str, Any]] = []
    covered_topics: set[str] = set()

    # First pass: exact difficulty match
    for q in unique_pool:
        topic_id = q.get("_meta", {}).get("topic_id") or q.get("topic_id", "")
        if topic_id in topic_ids:
            target_diff = difficulties.get(topic_id, "G2")
            q_diff = q.get("difficulty_label", "G2")
            if q_diff == target_diff:
                candidates.append(q)
                covered_topics.add(topic_id)

    # Second pass: any difficulty (relaxed)
    if len(candidates) < num_questions:
        for q in unique_pool:
            q_topic = q.get("_meta", {}).get("topic_id") or q.get("topic_id", "")
            if q_topic in topic_ids and q not in candidates:
                candidates.append(q)
                covered_topics.add(q_topic)

    # Third pass: fill remaining slots
    if len(candidates) < num_questions:
        for q in sorted(
            unique_pool,
            key=lambda x: x.get("evaluation", {}).get("quality_score", 0.5),
            reverse=True,
        ):
            if len(candidates) >= num_questions:
                break
            q_topic = q.get("_meta", {}).get("topic_id") or q.get("topic_id", "")
            if q_topic in topic_ids and q not in candidates:
                candidates.append(q)

    # Sort by quality score — highest quality first
    def quality_score(q: dict[str, Any]) -> float:
        return q.get("evaluation", {}).get("quality_score", 0.5)

    candidates.sort(key=quality_score, reverse=True)
    random.seed(42)  # Deterministic selection within same quality
    random.shuffle(candidates)
    selected = candidates[:num_questions]

    # Topics with no questions at all
    all_pool_topic_ids = {
        q.get("_meta", {}).get("topic_id") or q.get("topic_id", "")
        for q in unique_pool
    }
    missing = [tid for tid in topic_ids if tid not in all_pool_topic_ids]

    return selected, missing


# ─── Main generation functions ─────────────────────────────────────────────────

def get_study_plan(user_id: str) -> StudyPlan:
    """Generate a personalized study plan for a student."""
    profile = load_profile(user_id)
    topic_list = load_topic_list()
    weak_list = detect_weak_topics(user_id)

    recommended_difficulties: dict[str, str] = {}
    for w in weak_list:
        diff = get_adaptive_difficulty(w.topic_id, profile, topic_list)
        recommended_difficulties[w.topic_id] = diff

    priority = [w for w in weak_list if w.priority == "critical"]
    priority += [w for w in weak_list if w.priority == "medium"]

    pool = load_mcq_pool()
    notes: list[str] = []
    if not pool:
        notes.append("⚠️ MCQ pool trống. Cần chạy pipeline MCQGen trước.")

    return StudyPlan(
        user_id=user_id,
        weak_topics=weak_list,
        priority_topics=priority,
        recommended_difficulties=recommended_difficulties,
        total_questions_needed=len(weak_list) * 2,
        notes=notes,
        generated_at=db.now_iso(),
    )


def generate_adaptive_quiz(
    user_id: str,
    num_questions: int = 10,
    focus_topics: list[str] | None = None,
    mode: str = "adaptive",
) -> tuple[list[dict[str, Any]], StudyPlan]:
    """Generate an adaptive quiz for a student.

    Args:
        user_id: Student identifier
        num_questions: Number of questions to generate
        focus_topics: List of specific topic_ids to focus on
        mode: "adaptive" | "mixed" | "focus_weak"

    Returns:
        (questions: list[dict], study_plan: StudyPlan)
    """
    profile = load_profile(user_id)
    topic_list = load_topic_list()
    plan = get_study_plan(user_id)

    def _try_select(target_ids: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
        difficulties = {
            tid: get_adaptive_difficulty(tid, profile, topic_list)
            for tid in target_ids
        }
        return select_questions_for_topics(target_ids, difficulties, num_questions)

    questions: list[dict[str, Any]] = []
    missing: list[str] = []

    if mode == "mixed":
        if focus_topics:
            target_ids = focus_topics[:num_questions]
        else:
            attempted = [
                tid for tid, ts in profile.topic_stats.items() if ts.attempts > 0
            ]
            if not attempted:
                tl = load_topic_list()
                attempted = [
                    t.get("topic_id", "") for ch in tl for t in ch.get("topics", [])
                ]
            target_ids = attempted[:num_questions]
        questions, missing = _try_select(target_ids)
        if missing:
            plan.notes.append(f"⚠️ Thiếu câu cho topics: {missing}. Cần generate thêm.")
        return questions, plan

    elif mode == "focus_weak":
        if focus_topics:
            target_ids = focus_topics
        else:
            weak_ids = [w.topic_id for w in plan.weak_topics]
            if not weak_ids:
                plan.notes.append("✅ Không có topic yếu. Khuyến khích ôn tập đa dạng.")
                return [], plan
            target_ids = weak_ids
        questions, missing = _try_select(target_ids)
        if missing:
            plan.notes.append(f"⚠️ MCQ pool thiếu cho: {missing}. Cần generate on-demand.")
        return questions, plan

    else:
        # Adaptive: prioritize weak + recommended difficulty
        if focus_topics:
            target_ids = focus_topics
        else:
            weak_ids = [w.topic_id for w in plan.priority_topics]
            strong_ids = [
                tid for tid, ts in profile.topic_stats.items()
                if ts.attempts > 0 and not ts.is_weak and ts.accuracy >= 0.5
            ]
            if not weak_ids and not strong_ids:
                tl = load_topic_list()
                attempted = [
                    t.get("topic_id", "") for ch in tl for t in ch.get("topics", [])
                ]
                target_ids = attempted[:num_questions]
            else:
                weak_count = min(int(num_questions * 0.7), len(weak_ids))
                strong_count = num_questions - weak_count
                target_ids = weak_ids[:weak_count] + strong_ids[:strong_count]

        questions, missing = _try_select(target_ids)
        if missing:
            plan.notes.append(f"⚠️ MCQ pool thiếu cho: {missing}. Cần generate on-demand.")

        return questions, plan
