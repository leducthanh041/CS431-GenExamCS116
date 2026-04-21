"""
src/adaptive/api.py — FastAPI REST Endpoints
============================================
REST API for adaptive quiz engine: quiz sessions, grading, and student progress.

Endpoints:
    POST /quiz/start              — Create a new quiz session
    POST /quiz/{session_id}/answer — Submit an answer to a question
    POST /quiz/{session_id}/end   — End a quiz session and get summary
    GET  /progress/{user_id}      — Get student progress (profile + stats)
    GET  /study-plan/{user_id}    — Get personalized study plan
    GET  /topics                  — List all topics
    GET  /health                  — Health check

Run:
    cd CS431MCQGen
    uvicorn src.adaptive.api:app --reload --port 8000
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import db
from .difficulty import get_adaptive_difficulty, get_difficulty_display_name
from .profile import load_profile, load_topic_list
from .quiz import (
    AnswerResult,
    create_quiz_session,
    end_session,
    get_session,
    submit_answer,
)
from .recommend import StudyPlan, generate_adaptive_quiz, get_study_plan
from .tracking import get_chapter_accuracy, get_overall_accuracy, get_topic_accuracy
from .weakness import (
    TopicWeakness,
    WeaknessReport,
    detect_weak_topics,
    get_weakness_report,
)

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CS431MCQGen — Adaptive Quiz API",
    version="1.0.0",
    description="Adaptive MCQ generation and personalized learning API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic models ─────────────────────────────────────────────────────────


class QuizStartRequest(BaseModel):
    user_id: str = Field(..., description="Student identifier")
    num_questions: int = Field(default=10, ge=1, le=50)
    mode: str = Field(default="adaptive", description='"adaptive"|"mixed"|"focus_weak"')
    focus_topics: list[str] | None = Field(default=None)


class QuizStartResponse(BaseModel):
    session_id: str
    user_id: str
    mode: str
    num_questions: int
    started_at: str
    study_plan_notes: list[str]
    message: str = "Quiz session created. Answer questions sequentially."


class AnswerSubmitRequest(BaseModel):
    question_id: str
    user_answer: list[str] = Field(..., description="Selected option letters")
    time_spent_seconds: int = Field(default=0, ge=0)


class AnswerSubmitResponse(BaseModel):
    question_id: str
    is_correct: bool
    correct_answers: list[str]
    user_answer: list[str]
    explanation: str
    topic_id: str
    difficulty: str
    current_index: int
    total_questions: int


class QuizEndResponse(BaseModel):
    session_id: str
    user_id: str
    total_questions: int
    correct_count: int
    accuracy: float
    weak_topics_detected: list[str]
    duration_seconds: float
    questions_detail: list[dict]


class TopicInfo(BaseModel):
    topic_id: str
    topic_name: str
    chapter_id: str
    chapter_name: str
    difficulty: str


class ChapterProgress(BaseModel):
    chapter_id: str
    chapter_name: str
    topics_attempted: int
    topics_mastered: int
    topics_weak: int
    overall_accuracy: float
    weak_topic_ids: list[str]


class TopicProgress(BaseModel):
    topic_id: str
    topic_name: str
    chapter_id: str
    attempts: int
    correct: int
    accuracy: float
    avg_time_spent: float
    current_difficulty: str
    mastery_level: int
    is_weak: bool


class ProgressResponse(BaseModel):
    user_id: str
    overall_accuracy: float
    total_attempts: int
    total_correct: int
    mastery_score: float
    weak_topics_count: int
    strong_topics_count: int
    chapters: list[ChapterProgress]
    topics: list[TopicProgress]
    last_session_at: str


class StudyPlanResponse(BaseModel):
    user_id: str
    weak_topics: list[dict]
    priority_topics: list[dict]
    recommended_difficulties: dict[str, str]
    total_questions_needed: int
    notes: list[str]
    generated_at: str


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _build_topic_progress(ts) -> TopicProgress:
    return TopicProgress(
        topic_id=ts.topic_id,
        topic_name=ts.topic_name,
        chapter_id=ts.chapter_id,
        attempts=ts.attempts,
        correct=ts.correct,
        accuracy=ts.accuracy,
        avg_time_spent=ts.avg_time_spent,
        current_difficulty=ts.current_difficulty,
        mastery_level=ts.mastery_level,
        is_weak=ts.is_weak,
    )


def _build_chapter_progress(user_id: str, chapter_id: str, chapter_name: str) -> ChapterProgress:
    profile = load_profile(user_id)
    cs = profile.chapter_stats.get(chapter_id)
    if cs:
        return ChapterProgress(
            chapter_id=chapter_id,
            chapter_name=chapter_name,
            topics_attempted=cs.topics_attempted,
            topics_mastered=cs.topics_mastered,
            topics_weak=cs.topics_weak,
            overall_accuracy=cs.overall_accuracy,
            weak_topic_ids=cs.weak_topics,
        )
    return ChapterProgress(
        chapter_id=chapter_id,
        chapter_name=chapter_name,
        topics_attempted=0,
        topics_mastered=0,
        topics_weak=0,
        overall_accuracy=0.0,
        weak_topic_ids=[],
    )


def _build_topic_weakness(w: TopicWeakness) -> dict:
    return {
        "topic_id": w.topic_id,
        "topic_name": w.topic_name,
        "chapter_id": w.chapter_id,
        "chapter_name": w.chapter_name,
        "accuracy": w.accuracy,
        "attempts": w.attempts,
        "priority": w.priority,
        "suggested_difficulty": w.suggested_difficulty,
    }


# ─── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "adaptive-mcq-api"}


@app.get("/topics", response_model=list[TopicInfo])
def list_topics() -> list[TopicInfo]:
    topic_list = load_topic_list()
    result: list[TopicInfo] = []
    for ch in topic_list:
        chapter_id = ch.get("chapter_id", "")
        chapter_name = ch.get("chapter_name", "")
        for topic in ch.get("topics", []):
            result.append(TopicInfo(
                topic_id=topic.get("topic_id", ""),
                topic_name=topic.get("topic_name", ""),
                chapter_id=chapter_id,
                chapter_name=chapter_name,
                difficulty=topic.get("difficulty", "G2"),
            ))
    return result


@app.post("/quiz/start", response_model=QuizStartResponse)
def start_quiz(req: QuizStartRequest):
    db.ensure_student_dir(req.user_id)

    questions, plan = generate_adaptive_quiz(
        user_id=req.user_id,
        num_questions=req.num_questions,
        focus_topics=req.focus_topics,
        mode=req.mode,
    )

    if not questions:
        notes = plan.notes if hasattr(plan, 'notes') else []
        if not notes:
            notes.append("Không có câu hỏi phù hợp trong pool.")
        return QuizStartResponse(
            session_id="",
            user_id=req.user_id,
            mode=req.mode,
            num_questions=0,
            started_at=db.now_iso(),
            study_plan_notes=notes,
            message="Không tìm thấy câu hỏi phù hợp. Vui lòng thử lại sau.",
        )

    session = create_quiz_session(
        user_id=req.user_id,
        questions=questions,
        mode=req.mode,
    )

    notes: list[str] = []
    if hasattr(plan, 'notes'):
        notes = plan.notes

    return QuizStartResponse(
        session_id=session.session_id,
        user_id=session.user_id,
        mode=session.mode,
        num_questions=len(questions),
        started_at=session.started_at,
        study_plan_notes=notes,
    )


@app.post("/quiz/{session_id}/answer", response_model=AnswerSubmitResponse)
def answer_question(session_id: str, req: AnswerSubmitRequest):
    result = submit_answer(
        session_id=session_id,
        question_id=req.question_id,
        user_answer=req.user_answer,
        time_spent_seconds=req.time_spent_seconds,
    )

    session = get_session(session_id)
    current_index = 0
    total = 0
    if session:
        current_index = session.current_index
        total = len(session.questions)

    return AnswerSubmitResponse(
        question_id=req.question_id,
        is_correct=result.is_correct,
        correct_answers=result.correct_answers,
        user_answer=result.user_answer,
        explanation=result.explanation,
        topic_id=result.topic_id,
        difficulty=result.difficulty,
        current_index=current_index,
        total_questions=total,
    )


@app.post("/quiz/{session_id}/end", response_model=QuizEndResponse)
def end_quiz(session_id: str):
    summary = end_session(session_id)
    return QuizEndResponse(
        session_id=summary.session_id,
        user_id=summary.user_id,
        total_questions=summary.total_questions,
        correct_count=summary.correct_count,
        accuracy=summary.accuracy,
        weak_topics_detected=summary.weak_topics_detected,
        duration_seconds=summary.duration_seconds,
        questions_detail=summary.questions_detail,
    )


@app.get("/progress/{user_id}", response_model=ProgressResponse)
def get_progress(user_id: str):
    profile = load_profile(user_id)
    topic_list = load_topic_list()

    chapters: list[ChapterProgress] = []
    for ch in topic_list:
        chapters.append(_build_chapter_progress(
            user_id,
            ch.get("chapter_id", ""),
            ch.get("chapter_name", ""),
        ))

    topics: list[TopicProgress] = [
        _build_topic_progress(ts) for ts in profile.topic_stats.values()
    ]

    overall = profile.overall_stats

    return ProgressResponse(
        user_id=user_id,
        overall_accuracy=overall.overall_accuracy if overall else 0.0,
        total_attempts=overall.total_attempts if overall else 0,
        total_correct=overall.total_correct if overall else 0,
        mastery_score=overall.mastery_score if overall else 0.0,
        weak_topics_count=overall.weak_topics_count if overall else 0,
        strong_topics_count=overall.strong_topics_count if overall else 0,
        chapters=chapters,
        topics=topics,
        last_session_at=overall.last_session_at if overall else "",
    )


@app.get("/study-plan/{user_id}", response_model=StudyPlanResponse)
def get_user_study_plan(user_id: str):
    plan = get_study_plan(user_id)
    return StudyPlanResponse(
        user_id=plan.user_id,
        weak_topics=[_build_topic_weakness(w) for w in plan.weak_topics],
        priority_topics=[_build_topic_weakness(w) for w in plan.priority_topics],
        recommended_difficulties=plan.recommended_difficulties,
        total_questions_needed=plan.total_questions_needed,
        notes=plan.notes,
        generated_at=plan.generated_at,
    )


# ─── Standalone run ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
