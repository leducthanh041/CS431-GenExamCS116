"""
src/adaptive/quiz.py — Quiz Session Manager
==========================================
Manages quiz sessions: create, submit answer, grade, end session.

@dataclass QuizSession:
    session_id: str; user_id: str
    questions: list[dict]
    current_index: int
    answers: dict[str, list[str]]  # question_id → user_answer
    started_at: str; ended_at: str | None

Functions:
    create_quiz_session(user_id, questions) → QuizSession
    submit_answer(session_id, question_id, user_answer) → AnswerResult
    grade_quiz(session_id) → dict
    end_session(session_id) → SessionSummary
    get_session(session_id) → QuizSession | None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from typing import Any

from . import db
from .difficulty import get_difficulty_display_name
from .profile import load_profile, save_profile
from .tracking import record_interaction

# ─── Dataclasses ─────────────────────────────────────────────────────────────


@dataclass
class AnswerResult:
    """Result of submitting an answer to a question."""
    is_correct: bool
    correct_answers: list[str]
    user_answer: list[str]
    explanation: str = ""
    topic_id: str = ""
    difficulty: str = ""


@dataclass
class SessionSummary:
    """Summary of a completed quiz session."""
    session_id: str
    user_id: str
    total_questions: int
    correct_count: int
    accuracy: float
    weak_topics_detected: list[str]
    duration_seconds: float
    questions_detail: list[dict] = field(default_factory=list)


@dataclass
class QuizSession:
    """Active quiz session state."""
    session_id: str
    user_id: str
    questions: list[dict[str, Any]]
    current_index: int = 0
    answers: dict[str, list[str]] = field(default_factory=dict)
    question_times: dict[str, int] = field(default_factory=dict)
    started_at: str = ""
    ended_at: str | None = None
    mode: str = "adaptive"
    session_questions_order: list[str] = field(default_factory=list)
    explanations: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "questions": self.questions,
            "current_index": self.current_index,
            "answers": self.answers,
            "question_times": self.question_times,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "mode": self.mode,
            "session_questions_order": self.session_questions_order,
            "explanations": self.explanations,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QuizSession":
        return cls(
            session_id=d.get("session_id", ""),
            user_id=d.get("user_id", ""),
            questions=d.get("questions", []),
            current_index=d.get("current_index", 0),
            answers=d.get("answers", {}),
            question_times=d.get("question_times", {}),
            started_at=d.get("started_at", ""),
            ended_at=d.get("ended_at"),
            mode=d.get("mode", "adaptive"),
            session_questions_order=d.get("session_questions_order", []),
            explanations=d.get("explanations", {}),
        )


# ─── Session I/O ──────────────────────────────────────────────────────────────

def get_session(session_id: str) -> QuizSession | None:
    """Find a session by scanning all student session directories."""
    root = db._get_project_root()
    sessions_base = root / "data" / "students"

    if not sessions_base.exists():
        return None

    for student_dir in sessions_base.iterdir():
        if not student_dir.is_dir():
            continue
        sessions_dir = student_dir / "sessions"
        if not sessions_dir.exists():
            continue
        for session_file in sessions_dir.glob("*.json"):
            data = db.load_json(session_file)
            if data and data.get("session_id") == session_id:
                return QuizSession.from_dict(data)
    return None


def get_session_path(user_id: str, session_id: str) -> Any:
    """Return path to a specific session JSON file."""
    return db.get_session_path(user_id, session_id)


def save_session(session: QuizSession) -> None:
    """Save session state to disk."""
    db.ensure_student_dir(session.user_id)
    path = db.get_session_path(session.user_id, session.session_id)
    db.save_json(path, session.to_dict())


# ─── Core session functions ──────────────────────────────────────────────────

def create_quiz_session(
    user_id: str,
    questions: list[dict[str, Any]],
    mode: str = "adaptive",
) -> QuizSession:
    """Create a new quiz session.

    Args:
        user_id: Student identifier
        questions: List of MCQ question dicts
        mode: "adaptive" | "mixed" | "focus_weak"

    Returns:
        QuizSession (saved to disk)
    """
    db.ensure_student_dir(user_id)
    session_id = f"session_{db.new_uuid()[:8]}"

    # Load P9 explanations from the latest experiment's explain output
    explanations: dict[str, str] = {}
    root = db._get_project_root()
    for exp_dir in sorted(
        (root / "output").glob("exp_*/09_explain"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        for explain_file in exp_dir.glob("*.jsonl"):
            for line in explain_file.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    qid = rec.get("question_id", "")
                    expl = rec.get("explanation", "")
                    if qid and expl:
                        explanations[qid] = expl
                except json.JSONDecodeError:
                    continue
        if explanations:
            break  # Use latest experiment's explanations

    session = QuizSession(
        session_id=session_id,
        user_id=user_id,
        questions=questions,
        current_index=0,
        answers={},
        question_times={},
        started_at=db.now_iso(),
        mode=mode,
        session_questions_order=[
            q.get("question_id", f"q_{i}") for i, q in enumerate(questions)
        ],
        explanations=explanations,
    )
    save_session(session)
    return session


def submit_answer(
    session_id: str,
    question_id: str,
    user_answer: list[str],
    time_spent_seconds: int = 0,
) -> AnswerResult:
    """Grade a single answer and record the interaction.

    Args:
        session_id: Active session ID
        question_id: Question being answered
        user_answer: List of selected option letters (e.g. ["A", "C"])
        time_spent_seconds: Time taken to answer

    Returns:
        AnswerResult with grading info
    """
    session = get_session(session_id)
    if session is None:
        return AnswerResult(
            is_correct=False,
            correct_answers=[],
            user_answer=user_answer,
            explanation="Session not found",
        )

    question: dict[str, Any] | None = None
    for q in session.questions:
        if q.get("question_id") == question_id:
            question = q
            break

    if question is None:
        return AnswerResult(
            is_correct=False,
            correct_answers=[],
            user_answer=user_answer,
            explanation="Question not found in session",
        )

    correct_answers = question.get("correct_answers", [])
    user_set = set(u.upper() for u in user_answer)
    correct_set = set(c.upper() for c in correct_answers)
    is_correct = user_set == correct_set

    topic_id = question.get("_meta", {}).get("topic_id") or question.get("topic_id", "")
    topic_name = (
        question.get("_meta", {}).get("topic_name")
        or question.get("topic_name", "")
    )
    chapter_id = question.get("chapter_id", "")
    difficulty_label = question.get("difficulty_label", "G2")
    difficulty_score = {"G1": 1, "G2": 2, "G3": 3}.get(difficulty_label, 2)

    # Record interaction (updates profile + interaction log)
    record_interaction(
        user_id=session.user_id,
        question_id=question_id,
        chapter_id=chapter_id,
        topic_id=topic_id,
        topic_name=topic_name,
        difficulty_label=difficulty_label,
        difficulty_score=difficulty_score,
        user_answer=user_answer,
        correct_answers=correct_answers,
        time_spent_seconds=time_spent_seconds,
    )

    # Build explanation
    explanation = ""
    if is_correct:
        explanation = "✅ Chính xác!"
    else:
        explanation = f"❌ Sai. Đáp án đúng: {', '.join(correct_answers)}"

    p9_explanation = session.explanations.get(question_id, "")
    if p9_explanation:
        explanation = f"{explanation}\n\n📖 Giải thích:\n{p9_explanation}"

    # Update session
    session.answers[question_id] = user_answer
    session.question_times[question_id] = time_spent_seconds
    session.current_index += 1
    save_session(session)

    return AnswerResult(
        is_correct=is_correct,
        correct_answers=correct_answers,
        user_answer=user_answer,
        explanation=explanation,
        topic_id=topic_id,
        difficulty=difficulty_label,
    )


def grade_quiz(session_id: str) -> dict[str, Any]:
    """Grade all questions in a session.

    Returns:
        {
            "session_id": ...,
            "total": N,
            "correct": K,
            "accuracy": K/N,
            "question_results": [...],
            "weak_topics": [...]
        }
    """
    session = get_session(session_id)
    if session is None:
        return {"error": "Session not found"}

    results: list[dict[str, Any]] = []
    weak_topics: list[str] = []

    for q in session.questions:
        qid = q.get("question_id", "")
        user_ans = session.answers.get(qid, [])
        correct_ans = q.get("correct_answers", [])
        user_set = set(u.upper() for u in user_ans)
        correct_set = set(c.upper() for c in correct_ans)
        is_correct = user_set == correct_set

        topic_id = q.get("_meta", {}).get("topic_id") or q.get("topic_id", "")
        if not is_correct:
            weak_topics.append(topic_id)

        results.append({
            "question_id": qid,
            "question_text": q.get("question_text", ""),
            "topic_id": topic_id,
            "topic_name": (
                q.get("_meta", {}).get("topic_name") or q.get("topic_name", "")
            ),
            "user_answer": user_ans,
            "correct_answers": correct_ans,
            "is_correct": is_correct,
            "options": q.get("options", {}),
            "difficulty": q.get("difficulty_label", ""),
        })

    correct_count = sum(1 for r in results if r["is_correct"])
    total = len(results)

    return {
        "session_id": session_id,
        "user_id": session.user_id,
        "total": total,
        "correct": correct_count,
        "accuracy": correct_count / total if total > 0 else 0.0,
        "question_results": results,
        "weak_topics": weak_topics,
        "mode": session.mode,
    }


def end_session(session_id: str) -> SessionSummary:
    """Finalize a quiz session.

    Grades all questions, calculates stats, triggers weak detection,
    and updates profile. Returns a session summary.
    """
    session = get_session(session_id)
    if session is None:
        raise ValueError(f"Session {session_id} not found")

    session.ended_at = db.now_iso()
    save_session(session)

    grade_result = grade_quiz(session_id)

    from .weakness import detect_weak_topics

    weak_list = detect_weak_topics(session.user_id)
    weak_topic_ids = [w.topic_id for w in weak_list]

    try:
        started = datetime.fromisoformat(session.started_at)
        ended = datetime.fromisoformat(session.ended_at)
        duration = (ended - started).total_seconds()
    except Exception:
        duration = 0.0

    return SessionSummary(
        session_id=session_id,
        user_id=session.user_id,
        total_questions=grade_result["total"],
        correct_count=grade_result["correct"],
        accuracy=grade_result["accuracy"],
        weak_topics_detected=weak_topic_ids,
        duration_seconds=duration,
        questions_detail=grade_result["question_results"],
    )
