"""
src/adaptive/on_demand.py — MCQGen On-Demand Generation Bridge
============================================================
Bridges the adaptive engine with MCQGen pipeline to generate
missing questions when the pool is insufficient for weak topics.

Functions:
    get_pool_coverage()          → PoolStats
    find_missing_for_topics()    → list[MissingTopic]
    generate_on_demand()         → tuple[list[dict], list[MissingTopic]]
    trigger_pipeline()          → str (job_id) | None
    write_on_demand_config()    → Path
    refresh_pool()              → int (new questions added)
    get_coverage_report()       → dict
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import db
from .profile import load_topic_list

# ─── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class PoolStats:
    """Coverage stats for the MCQ pool."""
    total_questions: int = 0
    by_chapter: dict[str, int] = field(default_factory=dict)
    by_topic: dict[str, int] = field(default_factory=dict)
    by_difficulty: dict[str, int] = field(default_factory=dict)
    by_type: dict[str, int] = field(default_factory=dict)
    chapters_covered: list[str] = field(default_factory=list)
    topics_covered: list[str] = field(default_factory=list)
    topics_missing: list[str] = field(default_factory=list)
    quality_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class MissingTopic:
    """A topic missing from the MCQ pool (or insufficient coverage)."""
    topic_id: str
    topic_name: str
    chapter_id: str
    chapter_name: str
    needed_difficulty: str
    needed_count: int
    current_count: int
    priority: str  # "critical" | "medium" | "low"
    reason: str


# ─── Pool coverage analysis ───────────────────────────────────────────────────

def get_pool_coverage() -> PoolStats:
    """Analyze the current MCQ pool and return coverage stats."""
    pool = db.load_mcq_pool()
    stats = PoolStats()

    if not pool:
        topic_list = load_topic_list()
        all_topic_ids = []
        for ch in topic_list:
            for t in ch.get("topics", []):
                all_topic_ids.append(t.get("topic_id", ""))
        stats.topics_missing = all_topic_ids
        return stats

    topic_list = load_topic_list()
    topic_map: dict[str, dict] = {}
    for ch in topic_list:
        for t in ch.get("topics", []):
            topic_map[t.get("topic_id", "")] = {
                "topic_name": t.get("topic_name", ""),
                "chapter_id": ch.get("chapter_id", ""),
                "chapter_name": ch.get("chapter_name", ""),
            }

    for q in pool:
        stats.total_questions += 1

        diff = q.get("difficulty_label", "G2")
        stats.by_difficulty[diff] = stats.by_difficulty.get(diff, 0) + 1

        qtype = q.get("question_type", "single_correct")
        stats.by_type[qtype] = stats.by_type.get(qtype, 0) + 1

        topic_id = q.get("_meta", {}).get("topic_id") or q.get("topic_id", "")
        if not topic_id:
            topic_id = q.get("topic", "")
        if topic_id:
            stats.by_topic[topic_id] = stats.by_topic.get(topic_id, 0) + 1
            if topic_id not in stats.topics_covered:
                stats.topics_covered.append(topic_id)
            ch_info = topic_map.get(topic_id, {})
            ch_id = ch_info.get("chapter_id", "")
            if ch_id:
                stats.by_chapter[ch_id] = stats.by_chapter.get(ch_id, 0) + 1
                if ch_id not in stats.chapters_covered:
                    stats.chapters_covered.append(ch_id)

        eval_data = q.get("evaluation", {})
        quality = eval_data.get("quality_score", 0.5)
        qid = q.get("question_id", "")
        if qid:
            stats.quality_scores[qid] = quality

    all_topic_ids = list(topic_map.keys())
    stats.topics_missing = [tid for tid in all_topic_ids if tid not in stats.topics_covered]

    return stats


def find_missing_for_topics(
    topic_ids: list[str],
    num_per_topic: int = 5,
) -> list[MissingTopic]:
    """Find which topics in topic_ids have insufficient MCQ coverage."""
    pool = db.load_mcq_pool()
    topic_list = load_topic_list()

    topic_map: dict[str, dict] = {}
    for ch in topic_list:
        for t in ch.get("topics", []):
            topic_map[t.get("topic_id", "")] = {
                "topic_name": t.get("topic_name", ""),
                "chapter_id": ch.get("chapter_id", ""),
                "chapter_name": ch.get("chapter_name", ""),
                "difficulty": t.get("difficulty", "G2"),
            }

    topic_counts: dict[str, int] = {}
    for q in pool:
        topic_id = q.get("_meta", {}).get("topic_id") or q.get("topic_id", "") or q.get("topic", "")
        topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1

    missing: list[MissingTopic] = []
    for tid in topic_ids:
        info = topic_map.get(tid, {})
        current = topic_counts.get(tid, 0)
        needed = max(0, num_per_topic - current)
        if needed > 0:
            if current == 0:
                priority = "critical"
                reason = f"Không có câu hỏi nào trong pool cho topic này"
            elif current < 3:
                priority = "medium"
                reason = f"Chỉ có {current} câu, cần thêm {needed} câu"
            else:
                priority = "low"
                reason = f"Cần thêm {needed} câu để đủ {num_per_topic}/topic"

            missing.append(MissingTopic(
                topic_id=tid,
                topic_name=info.get("topic_name", tid),
                chapter_id=info.get("chapter_id", ""),
                chapter_name=info.get("chapter_name", ""),
                needed_difficulty=info.get("difficulty", "G2"),
                needed_count=needed,
                current_count=current,
                priority=priority,
                reason=reason,
            ))

    priority_order = {"critical": 0, "medium": 1, "low": 2}
    missing.sort(key=lambda m: (priority_order.get(m.priority, 3), m.current_count))
    return missing


# ─── On-demand config writer ──────────────────────────────────────────────────

def write_on_demand_config(
    missing_topics: list[MissingTopic],
    output_dir: Path | None = None,
) -> Path:
    """Write a focused generation config for on-demand MCQGen pipeline.

    Creates configs/on_demand_generation.yaml targeting only the missing topics
    with recommended difficulty levels.
    """
    root = db._get_project_root()

    chapter_topics: dict[str, list[str]] = {}
    chapter_difficulties: dict[str, dict[str, str]] = {}
    for m in missing_topics:
        if m.chapter_id not in chapter_topics:
            chapter_topics[m.chapter_id] = []
            chapter_difficulties[m.chapter_id] = {}
        chapter_topics[m.chapter_id].append(m.topic_id)
        chapter_difficulties[m.chapter_id][m.topic_id] = m.needed_difficulty

    focus_chapters = list(chapter_topics.keys())
    topic_weights: dict[str, float] = {}
    for m in missing_topics:
        topic_weights[m.topic_id] = 2.0

    total_needed = sum(m.needed_count for m in missing_topics)

    diff_counts = {"G1": 0, "G2": 0, "G3": 0}
    for m in missing_topics:
        diff_counts[m.needed_difficulty] += m.needed_count

    on_demand_cfg: dict[str, Any] = {
        "_generated_by": "adaptive_on_demand",
        "_generated_at": db.now_iso(),
        "_reason": f"Pool thiếu {len(missing_topics)} topics, cần {total_needed} câu",
        "generation": {
            "target_range": [total_needed, total_needed + 5],
            "single_correct_ratio": 0.6,
            "two_correct_ratio": 0.6,
            "three_correct_ratio": 0.4,
            "focus_chapters": focus_chapters,
            "focus_topics": [m.topic_id for m in missing_topics],
            "topic_weights": topic_weights,
        },
        "difficulty_distribution": {
            "G1": diff_counts["G1"] / total_needed if total_needed > 0 else 0.33,
            "G2": diff_counts["G2"] / total_needed if total_needed > 0 else 0.34,
            "G3": diff_counts["G3"] / total_needed if total_needed > 0 else 0.33,
        },
        "_missing_topics_summary": [
            {
                "topic_id": m.topic_id,
                "topic_name": m.topic_name,
                "chapter_id": m.chapter_id,
                "current": m.current_count,
                "needed": m.needed_count,
                "difficulty": m.needed_difficulty,
                "priority": m.priority,
            }
            for m in missing_topics
        ],
    }

    config_path = root / "configs" / "on_demand_generation.yaml"
    try:
        import yaml
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(on_demand_cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    except ImportError:
        import json
        config_path = root / "configs" / "on_demand_generation.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(on_demand_cfg, f, ensure_ascii=False, indent=2)

    return config_path


# ─── Pipeline trigger ─────────────────────────────────────────────────────────

def trigger_pipeline(
    config_path: Path,
    steps: str = "03-08",
    wait: bool = False,
) -> str | None:
    """Trigger the MCQGen pipeline for on-demand generation.

    Args:
        config_path: Path to the on-demand config YAML
        steps: Pipeline steps to run (default "03-08" = gen + eval only)
        wait: If True, wait for completion (blocking)

    Returns:
        SLURM job ID if submitted, None if failed
    """
    root = db._get_project_root()
    script_path = root / "scripts" / "00_pipeline.sh"

    if not script_path.exists():
        return None

    # Copy active config
    import shutil
    active_cfg = root / "configs" / "generation_config_active.yaml"
    shutil.copy(config_path, active_cfg)

    cmd = [
        "sbatch",
        "--wait" if wait else "--wait=no",
        "--output", str(root / "log" / "on_demand_%j.out"),
        "--error", str(root / "log" / "on_demand_%j.err"),
        str(script_path),
        steps,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            if "Submitted batch job" in line:
                parts = line.strip().split()
                if parts:
                    return parts[-1]
        return None
    except Exception:
        return None


# ─── Pool refresh ────────────────────────────────────────────────────────────

def refresh_pool() -> int:
    """Reload MCQ pool and return total questions available."""
    pool = db.load_mcq_pool()
    return len(pool)


# ─── Main on-demand orchestration ───────────────────────────────────────────

def generate_on_demand(
    topic_ids: list[str],
    num_per_topic: int = 5,
    auto_trigger: bool = False,
    difficulty_overrides: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[MissingTopic]]:
    """Generate questions on-demand for specified topics.

    Args:
        topic_ids: Topics needing questions
        num_per_topic: Minimum questions per topic (default 5)
        auto_trigger: If True, automatically trigger pipeline
        difficulty_overrides: Optional {topic_id: "G1"|"G2"|"G3"} overrides

    Returns:
        (available_questions: list[dict], missing_topics: list[MissingTopic])
    """
    missing = find_missing_for_topics(topic_ids, num_per_topic)
    if difficulty_overrides:
        for m in missing:
            if m.topic_id in difficulty_overrides:
                m.needed_difficulty = difficulty_overrides[m.topic_id]

    pool = db.load_mcq_pool()

    available: list[dict[str, Any]] = []
    for q in pool:
        q_topic = q.get("_meta", {}).get("topic_id") or q.get("topic_id", "") or q.get("topic", "")
        if q_topic in topic_ids:
            available.append(q)

    if not missing:
        return available, []

    config_path = write_on_demand_config(missing)
    job_id = None
    if auto_trigger:
        job_id = trigger_pipeline(config_path, steps="03-08", wait=False)
        if job_id:
            db.ensure_student_dir("_system")
            system_dir = db._student_dir("_system")
            job_log = {
                "job_id": job_id,
                "submitted_at": db.now_iso(),
                "topics": [m.topic_id for m in missing],
                "config": str(config_path),
                "status": "submitted",
            }
            db.append_jsonl(system_dir / "on_demand_jobs.jsonl", job_log)

    return available, missing


# ─── Coverage report ──────────────────────────────────────────────────────────

def get_coverage_report() -> dict[str, Any]:
    """Generate a human-readable coverage report for the MCQ pool."""
    stats = get_pool_coverage()
    topic_list = load_topic_list()

    total_topics = sum(len(ch.get("topics", [])) for ch in topic_list)
    total_chapters = len(topic_list)

    total = stats.total_questions
    g1_ratio = stats.by_difficulty.get("G1", 0) / total if total > 0 else 0
    g2_ratio = stats.by_difficulty.get("G2", 0) / total if total > 0 else 0
    g3_ratio = stats.by_difficulty.get("G3", 0) / total if total > 0 else 0

    scores = list(stats.quality_scores.values())
    avg_quality = sum(scores) / len(scores) if scores else 0.0

    report: dict[str, Any] = {
        "pool_total": total,
        "topics_covered": len(stats.topics_covered),
        "topics_missing": len(stats.topics_missing),
        "total_topics": total_topics,
        "coverage_pct": len(stats.topics_covered) / total_topics * 100 if total_topics > 0 else 0,
        "chapters_covered": len(stats.chapters_covered),
        "total_chapters": total_chapters,
        "chapter_coverage_pct": len(stats.chapters_covered) / total_chapters * 100 if total_chapters > 0 else 0,
        "difficulty_distribution": {
            "G1": {"count": stats.by_difficulty.get("G1", 0), "ratio": g1_ratio},
            "G2": {"count": stats.by_difficulty.get("G2", 0), "ratio": g2_ratio},
            "G3": {"count": stats.by_difficulty.get("G3", 0), "ratio": g3_ratio},
        },
        "type_distribution": dict(stats.by_type),
        "avg_quality_score": avg_quality,
        "missing_topics": stats.topics_missing,
        "recommendations": _build_recommendations(stats, total, total_topics),
    }
    return report


def _build_recommendations(stats: PoolStats, total: int, total_topics: int) -> list[str]:
    """Build actionable recommendations based on pool coverage."""
    recs = []
    if total == 0:
        recs.append("⚠️ Pool trống hoàn toàn. Cần chạy full pipeline (Steps 01-09).")
        return recs

    if stats.by_difficulty.get("G1", 0) == 0:
        recs.append("🔴 Thiếu câu G1 (Nhớ/Hiểu). Cần sinh thêm câu dễ để build confidence.")
    if stats.by_difficulty.get("G3", 0) == 0:
        recs.append("🔴 Thiếu câu G3 (Đánh giá/Sáng tạo). Cần sinh thêm câu khó.")
    if stats.by_type.get("multiple_correct", 0) == 0:
        recs.append("🔴 Không có câu multiple-correct. Nên thêm 30-40% câu nhiều đáp án.")

    coverage_pct = len(stats.topics_covered) / total_topics * 100 if total_topics > 0 else 0
    if coverage_pct < 30:
        recs.append(f"🔴 Coverage chỉ {coverage_pct:.0f}%. Cần expand pool rộng rãi hơn.")
    elif coverage_pct < 70:
        recs.append(f"🟡 Coverage {coverage_pct:.0f}%. Đủ cho demo, cần expand để cover đầy đủ.")

    if len(stats.chapters_covered) < 11:
        all_chapters = {"ch02","ch03","ch04","ch05","ch06","ch07a","ch07b","ch08","ch09","ch10","ch11"}
        missing_ch = all_chapters - set(stats.chapters_covered)
        recs.append(f"🟡 Các chapter chưa có câu: {', '.join(sorted(missing_ch))}")

    return recs
