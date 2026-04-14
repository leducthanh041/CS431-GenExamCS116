"""
mcq_renderer.py — Render MCQ JSONL → beautiful HTML
====================================================
Dùng trong Streamlit để hiển thị câu hỏi đẹp.
"""

from __future__ import annotations

import json
from typing import Any


def render_mcq_card(mcq: dict, index: int | None = None) -> str:
    """
    Render 1 MCQ dict → HTML string for st.markdown().
    Shows: question text, options, correct answers, topic, difficulty.
    """
    q_id     = mcq.get("question_id", f"q{(index+1) if index is not None else '?'}")
    q_text   = mcq.get("question_text", mcq.get("refined_question_text", ""))
    q_type   = mcq.get("question_type", "single_correct")
    options  = mcq.get("options", {})
    correct  = mcq.get("correct_answers", mcq.get("correct_answer_labels", []))
    topic    = mcq.get("topic", "")
    difficulty = mcq.get("difficulty_label", mcq.get("difficulty", "G2"))

    # Type label
    type_label = (
        "✅ [Nhiều đáp án đúng]" if q_type == "multiple_correct"
        else "✅ [Một đáp án đúng]"
    )

    # Status badge
    status = mcq.get("status", "unknown")
    if status == "accepted":
        badge = '<span style="background:#22c55e;color:white;padding:2px 8px;border-radius:12px;font-size:12px">✅ Accepted</span>'
    elif status == "rejected":
        badge = '<span style="background:#ef4444;color:white;padding:2px 8px;border-radius:12px;font-size:12px">❌ Rejected</span>'
    else:
        badge = ""

    # Difficulty badge
    diff_color = {"G1": "#3b82f6", "G2": "#f59e0b", "G3": "#ef4444"}.get(difficulty, "#6b7280")
    diff_badge = f'<span style="background:{diff_color};color:white;padding:2px 8px;border-radius:12px;font-size:12px">{difficulty}</span>'

    # Options
    option_labels = ["A", "B", "C", "D"]
    option_rows = []
    for lbl in option_labels:
        text = options.get(lbl, "")
        if not text:
            continue
        is_correct = lbl in correct
        bg = "#dcfce7" if is_correct else "#f9fafb"
        border_color = "#22c55e" if is_correct else "#d1d5db"
        span_color = "#16a34a" if is_correct else "#6b7280"
        option_rows.append(
            f'<div style="background:{bg};border-radius:6px;padding:8px 12px;margin:4px 0;'
            f'border-left:4px solid {border_color};display:flex;gap:8px">'
            f'<span style="font-weight:700;color:{span_color}">{lbl}.</span>'
            f'<span>{text}</span></div>'
        )

    options_html = "\n".join(option_rows)

    # Evaluation result (if available)
    eval_html = ""
    if "evaluation" in mcq:
        ev = mcq["evaluation"]
        scores = {
            "Format": ev.get("format_pass"),
            "Language": ev.get("language_pass"),
            "Grammar": ev.get("grammar_pass"),
            "Relevance": ev.get("relevance_pass"),
            "Answerability": ev.get("answerability_pass"),
            "Correct set": ev.get("correct_set_pass"),
            "No 4-correct": ev.get("no_four_correct_pass"),
            "Answer not in stem": ev.get("answer_not_in_stem_pass"),
        }
        score_items = []
        for k, v in scores.items():
            icon = "✅" if v else "❌"
            score_items.append(f'<span style="margin-right:8px">{icon} {k}</span>')
        eval_html = (
            f'<details style="margin-top:8px">'
            f'<summary style="cursor:pointer;font-size:13px;color:#6b7280">📋 Evaluation ({ev.get("quality_score", 0):.0%})</summary>'
            f'<div style="padding:8px;background:#f9fafb;border-radius:6px;margin-top:4px">'
            + "".join(score_items) +
            f'</div></details>'
        )

    # IWF result
    iwf_html = ""
    if "distractor_evaluation" in mcq:
        iwf = mcq["distractor_evaluation"]
        iwf_pass = iwf.get("overall_distractor_quality_pass", True)
        bad = iwf.get("bad_options", [])
        iwf_color = "#22c55e" if iwf_pass else "#ef4444"
        iwf_text = "✅ IWF OK" if iwf_pass else f"❌ IWF issues: {', '.join(bad)}"
        iwf_html = (
            f'<div style="margin-top:6px;font-size:13px">'
            f'<span style="color:{iwf_color}">{iwf_text}</span>'
            f'</div>'
        )

    # Explanation
    explanation_html = ""
    if mcq.get("explanation"):
        explanation_html = (
            f'<details style="margin-top:8px">'
            f'<summary style="cursor:pointer;font-size:13px;color:#3b82f6">💡 Explanation</summary>'
            f'<div style="padding:10px;background:#eff6ff;border-radius:6px;margin-top:4px;'
            f'border-left:3px solid #3b82f6;font-size:14px">'
            f'{mcq["explanation"]}</div></details>'
        )

    # Sources
    sources_html = ""
    if mcq.get("sources") and len(mcq["sources"]) > 0:
        src_links = []
        for src in mcq["sources"][:3]:
            label = src.get("label", src.get("source", "source"))
            url = src.get("url", "")
            if url:
                src_links.append(f'<a href="{url}" target="_blank" style="color:#3b82f6">{label}</a>')
            else:
                src_links.append(f'<span style="color:#6b7280">{label}</span>')
        if src_links:
            sources_html = (
                f'<div style="margin-top:8px;font-size:12px;color:#6b7280">'
                f'📚 Sources: {" | ".join(src_links)}</div>'
            )

    html = f"""
<div style="background:white;border-radius:12px;border:1px solid #e5e7eb;
            padding:16px;margin:12px 0;font-family:Segoe UI,sans-serif;max-width:800px">

  <!-- Header -->
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:6px">
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
      <strong style="font-size:15px;color:#374151">{q_id}</strong>
      {diff_badge}
      <span style="font-size:13px;color:#6b7280">{topic}</span>
    </div>
    <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">
      {type_label}
      {badge}
    </div>
  </div>

  <!-- Question text -->
  <div style="font-size:15px;line-height:1.6;margin-bottom:12px;color:#1f2937;white-space:pre-wrap">
    {q_text}
  </div>

  <!-- Options -->
  <div style="margin-bottom:8px">{options_html}</div>

  <!-- Correct answers reveal -->
  <div style="font-size:13px;color:#6b7280;margin-top:8px">
    <strong>Đáp án:</strong> {", ".join(correct)}
  </div>

  {eval_html}
  {iwf_html}
  {explanation_html}
  {sources_html}
</div>
"""
    return html


def render_mcq_list(mcqs: list[dict]) -> str:
    """Render list of MCQs as a batch HTML."""
    if not mcqs:
        return '<p style="color:#6b7280;font-size:14px">Chưa có câu hỏi nào.</p>'
    cards = [render_mcq_card(mcq, i) for i, mcq in enumerate(mcqs)]
    return "\n".join(cards)


def stats_summary(mcqs: list[dict]) -> dict[str, Any]:
    """Compute summary statistics from a list of MCQ dicts."""
    if not mcqs:
        return {"total": 0, "accepted": 0, "rejected": 0, "pass_rate": 0.0,
                "single_correct": 0, "multiple_correct": 0,
                "g1": 0, "g2": 0, "g3": 0, "avg_quality": 0.0}

    total = len(mcqs)
    accepted = sum(1 for m in mcqs if m.get("status") == "accepted")
    rejected = sum(1 for m in mcqs if m.get("status") == "rejected")

    single = sum(1 for m in mcqs if m.get("question_type") == "single_correct")
    multi   = sum(1 for m in mcqs if m.get("question_type") == "multiple_correct")

    diffs = {"G1": 0, "G2": 0, "G3": 0}
    for m in mcqs:
        d = m.get("difficulty_label", m.get("difficulty", ""))
        if d in diffs:
            diffs[d] += 1

    quality_scores = []
    for m in mcqs:
        ev = m.get("evaluation", {})
        if ev and "quality_score" in ev:
            quality_scores.append(ev["quality_score"])
    avg_q = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return {
        "total": total,
        "accepted": accepted,
        "rejected": rejected,
        "pass_rate": accepted / total * 100 if total > 0 else 0,
        "single_correct": single,
        "multiple_correct": multi,
        **diffs,
        "avg_quality": avg_q,
    }


def render_stats_html(stats: dict) -> str:
    """Render stats dict → HTML badge row."""
    return f"""
<div style="display:flex;flex-wrap:wrap;gap:12px;margin:12px 0">
  <div style="background:#f3f4f6;border-radius:8px;padding:10px 16px;text-align:center;min-width:80px">
    <div style="font-size:22px;font-weight:700;color:#374151">{stats['total']}</div>
    <div style="font-size:12px;color:#6b7280">Tổng</div>
  </div>
  <div style="background:#dcfce7;border-radius:8px;padding:10px 16px;text-align:center;min-width:80px">
    <div style="font-size:22px;font-weight:700;color:#16a34a">{stats['accepted']}</div>
    <div style="font-size:12px;color:#16a34a">✅ Accepted</div>
  </div>
  <div style="background:#fee2e2;border-radius:8px;padding:10px 16px;text-align:center;min-width:80px">
    <div style="font-size:22px;font-weight:700;color:#dc2626">{stats['rejected']}</div>
    <div style="font-size:12px;color:#dc2626">❌ Rejected</div>
  </div>
  <div style="background:#fef3c7;border-radius:8px;padding:10px 16px;text-align:center;min-width:80px">
    <div style="font-size:22px;font-weight:700;color:#d97706">{stats['pass_rate']:.0f}%</div>
    <div style="font-size:12px;color:#d97706">Pass rate</div>
  </div>
  <div style="background:#eff6ff;border-radius:8px;padding:10px 16px;text-align:center;min-width:80px">
    <div style="font-size:22px;font-weight:700;color:#2563eb">{stats['single_correct']}</div>
    <div style="font-size:12px;color:#2563eb">Single</div>
  </div>
  <div style="background:#f3e8ff;border-radius:8px;padding:10px 16px;text-align:center;min-width:80px">
    <div style="font-size:22px;font-weight:700;color:#9333ea">{stats['multiple_correct']}</div>
    <div style="font-size:12px;color:#9333ea">Multiple</div>
  </div>
  <div style="background:#f0fdf4;border-radius:8px;padding:10px 16px;text-align:center;min-width:80px">
    <div style="font-size:22px;font-weight:700;color:#22c55e">{stats['avg_quality']:.0%}</div>
    <div style="font-size:12px;color:#22c55e">Avg quality</div>
  </div>
</div>
"""
