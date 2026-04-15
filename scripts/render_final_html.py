#!/usr/bin/env python3
"""
render_final_html.py — Render final_accepted_questions.jsonl → 2 HTML files:
  1. <prefix>_with_answer.html   — correct answers highlighted
  2. <prefix>_no_answer.html     — clean, no answers shown

Sources are joined from all_p1_results.jsonl via (topic_id, seq).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import _to_seconds, _format_timestamp


# ── Paths ────────────────────────────────────────────────────────────────────
P1_RESULTS  = PROJECT_ROOT / "output/exp_02_baseline/03_gen_stem/all_p1_results.jsonl"
FINAL_JSONL = PROJECT_ROOT / "output/exp_02_baseline/08_eval_iwf/final_accepted_questions.jsonl"
OUT_WITH    = PROJECT_ROOT / "output/exp_02_baseline/08_eval_iwf/final_accepted_questions_with_answer.html"
OUT_NO      = PROJECT_ROOT / "output/exp_02_baseline/08_eval_iwf/final_accepted_questions_no_answer.html"


# ── Load P1 sources ─────────────────────────────────────────────────────────

def load_p1_sources(path: Path) -> dict[tuple[str, int], list[dict]]:
    """
    Index P1 results by (topic_id, seq).
    Returns {(topic_id, seq): [sources_list]}.
    """
    sources_map: dict[tuple[str, int], list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = d.get("_meta", {})
            topic_id = meta.get("topic_id", "")
            seq = meta.get("seq", 0)
            sources = d.get("sources", [])
            if topic_id:
                sources_map[(topic_id, seq)] = sources
    return sources_map


def load_final_questions(path: Path) -> list[dict]:
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


# ── Helpers ─────────────────────────────────────────────────────────────────

def format_seconds_display(seconds: str | int | float | None) -> str:
    if seconds is None or seconds == "":
        return ""
    try:
        total = int(float(str(seconds)))
    except (ValueError, TypeError):
        return ""
    if total >= 3600:
        hh = total // 3600
        mm = (total % 3600) // 60
        ss = total % 60
        return f"{hh}:{mm:02d}:{ss:02d}"
    else:
        mm = total // 60
        ss = total % 60
        return f"{mm}:{ss:02d}"


def difficulty_badge(diff: str) -> str:
    colors = {"G1": "#27ae60", "G2": "#f39c12", "G3": "#e74c3c"}
    color = colors.get(diff, "#7f8c8d")
    return (f'<span style="background:{color};color:#fff;padding:2px 10px;'
            f'border-radius:12px;font-size:0.75em;font-weight:700">{diff}</span>')


def qtype_badge(qtype: str) -> str:
    if qtype == "multiple_correct":
        return (f'<span style="background:#8e44ad;color:#fff;padding:2px 10px;'
                f'border-radius:12px;font-size:0.75em;font-weight:700">Nhiều đáp án</span>')
    return (f'<span style="background:#2980b9;color:#fff;padding:2px 10px;'
            f'border-radius:12px;font-size:0.75em;font-weight:700">Một đáp án</span>')


def build_sources_html(sources: list[dict]) -> str:
    """Render video + slide pills."""
    if not sources:
        return ""

    video_item = None
    slide_item = None

    for s in sources:
        source_type = s.get("source_type", "")
        youtube_url = s.get("youtube_url", "")
        label = s.get("label", "")

        if source_type == "video_transcript" and youtube_url and video_item is None:
            ts = ""
            if "&t=" in youtube_url:
                ts_part = youtube_url.split("&t=")[-1].split("&")[0]
                ts = format_seconds_display(ts_part)
            display_url = youtube_url.split("&t=")[0]
            if ts:
                display_text = ts
                full_href = youtube_url
            else:
                display_text = display_url
                full_href = youtube_url

            slide_label = s.get("slide_label", "")
            icon = "&#127909;" if slide_label else "&#9654;"
            title = slide_label or "Không có slide tương ứng (video coding)"

            video_item = (
                f'<span style="display:inline-flex;align-items:center;gap:4px;'
                f'background:#e8f5e9;border:1px solid #a5d6a7;border-radius:16px;'
                f'padding:3px 10px;font-size:0.78em" title="{title}">'
                f'{icon}&nbsp;<a href="{full_href}" target="_blank" '
                f'style="color:#2e7d32;font-weight:600;text-decoration:none">{display_text}</a>'
                f'</span>'
            )

        elif source_type == "slide_pdf" and slide_item is None:
            display_label = label or s.get("chunk_id", "")
            if display_label:
                slide_item = (
                    f'<span style="display:inline-flex;align-items:center;gap:4px;'
                    f'background:#f3e5f5;border:1px solid #ce93d8;border-radius:16px;'
                    f'padding:3px 10px;font-size:0.78em">'
                    f'&#128196;&nbsp;<span style="color:#7b1fa2;font-weight:600">{display_label}</span>'
                    f'</span>'
                )

    items = [p for p in [video_item, slide_item] if p]
    if not items:
        return ""
    return ('<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px">'
            + "".join(items) + "</div>")


def build_options_html(options: dict, correct_letters: list[str],
                       reveal_correct: bool = False) -> str:
    """Render A/B/C/D options; highlight correct ones only when reveal=True."""
    letters = ["A", "B", "C", "D"]
    rows = []
    for letter in letters:
        text = options.get(letter, "")
        if not text:
            continue
        is_correct = letter in correct_letters
        if reveal_correct and is_correct:
            bg = "#d4edda"
            border = "#27ae60"
            marker = "&#10003;"  # ✓
            marker_html = f'<span style="color:#27ae60;font-weight:700">{marker}</span>'
        else:
            bg = "#f8f9fa"
            border = "#dee2e6"
            marker_html = ""
        rows.append(
            f'<div style="background:{bg};border-left:4px solid {border};'
            f'padding:8px 12px;margin:4px 0;border-radius:4px">'
            f'<strong style="font-family:monospace">{letter}.</strong>&nbsp;{text}&nbsp;{marker_html}'
            f'</div>'
        )
    return '<div style="margin-bottom:14px">' + "".join(rows) + "</div>"


def build_explanation_html(q: dict, sources: list[dict]) -> str:
    """
    Build a collapsible 'Giải thích' section from CoT steps (P5-P7).
    Shows which correct options are correct and why each wrong distractor is wrong,
    citing video/slide sources where available.
    """
    cot = q.get("_cot_steps", {})
    if not cot:
        return ""

    p5 = cot.get("p5", {})
    p6 = cot.get("p6", {})
    p7 = cot.get("p7", {})

    evaluations = p5.get("evaluations", [])
    kept_options = p6.get("kept_options", [])
    selected_distractors = p7.get("selected_distractors", [])

    # Build a map: option_text -> is_correct + notes
    opt_map: dict[str, dict] = {}
    for ev in evaluations:
        opt_map[ev.get("option_text", "")] = {
            "is_correct": ev.get("is_correct", False),
            "notes": ev.get("notes", ""),
            "misleading_likelihood": ev.get("misleading_likelihood", 0),
            "error_type": None,
        }

    for d in selected_distractors:
        text = d.get("option_text", "")
        if text in opt_map:
            opt_map[text]["error_type"] = d.get("error_type", "")

    # Collect video/slide URLs for citation
    yt_url = None
    slide_label = None
    for s in sources:
        if s.get("source_type") == "video_transcript" and not yt_url:
            yt_url = s.get("youtube_url", "")
        elif s.get("source_type") == "slide_pdf" and not slide_label:
            slide_label = s.get("label", "")

    # Find the correct answer letters
    correct_letters = q.get("correct_answers", [])
    correct_letter_set = set(correct_letters)
    correct_texts = {q.get("options", {}).get(l, "").strip() for l in correct_letters}

    lines = []

    # Correct answers explanation
    correct_parts = []
    for letter in correct_letters:
        opt_text = q.get("options", {}).get(letter, "").strip()
        if opt_text in opt_map:
            note = opt_map[opt_text].get("notes", "")
            if note:
                correct_parts.append(f"<strong>{letter}.</strong> {opt_text} — {note}")
            else:
                correct_parts.append(f"<strong>{letter}.</strong> {opt_text}")
        else:
            correct_parts.append(f"<strong>{letter}.</strong> {opt_text}")

    if correct_parts:
        lines.append("<p style='margin:0 0 6px'><strong>&#10003; Đáp án đúng:</strong></p>")
        for part in correct_parts:
            lines.append(f"<p style='margin:0 0 4px;padding-left:12px'>{part}</p>")

    # Distractor explanation
    kept = {ko.get("option_text", ""): ko.get("reason", "") for ko in kept_options}
    selected = {d.get("option_text", ""): d for d in selected_distractors}

    wrong_parts = []
    for ev in evaluations:
        if not ev.get("is_correct", False):
            opt_text = ev.get("option_text", "")
            reason = ev.get("notes", "") or kept.get(opt_text, "")
            if opt_text in selected:
                reason = selected[opt_text].get("error_type", "") or reason
            if reason:
                wrong_parts.append(f"• <strong>{opt_text}</strong>: {reason}")
            else:
                wrong_parts.append(f"• <strong>{opt_text}</strong>")

    if wrong_parts:
        lines.append("<p style='margin:8px 0 4px'><strong>&#10007; Loại trừ:</strong></p>")
        for part in wrong_parts:
            lines.append(f"<p style='margin:0 0 3px;padding-left:12px'>{part}</p>")

    # Source citation for explanation
    source_cite = ""
    if yt_url:
        ts = ""
        if "&t=" in yt_url:
            ts = format_seconds_display(yt_url.split("&t=")[-1].split("&")[0])
        display_url = yt_url.split("&t=")[0]
        cite_text = ts if ts else display_url
        source_cite = (
            f'<p style="margin:8px 0 0"><a href="{yt_url}" target="_blank" '
            f'style="color:#c0392b;font-size:0.85em">'
            f'&#127909; Video tham khảo ({cite_text})</a></p>'
        )
    elif slide_label:
        source_cite = (
            f'<p style="margin:8px 0 0;color:#7b1fa2;font-size:0.85em">'
            f'&#128196; Slide: {slide_label}</p>'
        )

    if not lines and not source_cite:
        return ""

    inner = "".join(lines) + source_cite
    return (
        f'<details style="background:#f8f9fa;border:1px solid #ddd;border-radius:6px;'
        f'padding:10px 14px;margin-top:8px">'
        f'<summary style="cursor:pointer;font-weight:600;color:#555;font-size:0.88em">'
        f'&#9432; Xem giải thích</summary>'
        f'<div style="margin-top:8px;font-size:0.88em;color:#444;line-height:1.5">'
        f'{inner}</div></details>'
    )


def eval_badge(passed: bool | None, label: str) -> str:
    if passed is None:
        return ""
    color = "#27ae60" if passed else "#e74c3c"
    icon = "&#10003;" if passed else "&#10007;"
    return (f'<span style="background:{color};color:#fff;padding:1px 6px;'
            f'border-radius:8px;font-size:0.7em" title="{label}">{icon}</span>')


def build_question_card(q: dict, sources: list[dict], idx: int,
                        reveal_correct: bool = False) -> str:
    qid     = q.get("question_id", f"q{idx+1}")
    qtext   = q.get("question_text", "")
    qtype   = q.get("question_type", "single")
    opts    = q.get("options", {})
    correct = q.get("correct_answers", [])
    topic   = q.get("topic", "")
    diff    = q.get("difficulty_label", "G2")
    eval_   = q.get("evaluation", {})
    iwf     = q.get("distractor_evaluation", {})
    meta    = q.get("_meta", {})
    topic_id = meta.get("topic_id", "?")
    seq     = meta.get("seq", "?")
    n_cot   = len(q.get("_cot_steps", {}))

    # Badges
    eval_badges = ""
    for key, label in [
        ("format_pass",              "Format OK"),
        ("language_pass",            "Ngôn ngữ OK"),
        ("grammar_pass",             "Ngữ pháp OK"),
        ("relevance_pass",           "Relevance OK"),
        ("answerability_pass",       "Trả lời được"),
        ("correct_set_pass",         "Đáp án đúng"),
        ("no_four_correct_pass",     "Không 4 đúng"),
        ("answer_not_in_stem_pass",  "Đáp án không trong stem"),
    ]:
        v = eval_.get(key)
        if v is not None:
            eval_badges += eval_badge(bool(v), label)

    score = eval_.get("quality_score", "?")

    iwf_count = q.get("final_iwf_count", iwf.get("total_iwf_count", 0))
    iwf_color = "#27ae60" if iwf_count == 0 else "#e74c3c"
    iwf_html  = (f'<span style="background:{iwf_color};color:#fff;padding:2px 8px;'
                 f'border-radius:8px;font-size:0.75em">IWF={iwf_count}</span>')

    topic_badge = ""
    if topic:
        topic_badge = (f'<span style="background:#2c3e50;color:#fff;padding:2px 8px;'
                       f'border-radius:8px;font-size:0.75em">{topic[:60]}</span>')

    sources_html = build_sources_html(sources)
    options_html = build_options_html(opts, correct, reveal_correct=reveal_correct)
    explanation_html = build_explanation_html(q, sources) if reveal_correct else ""

    # IWF warnings
    iwf_warn = ""
    if iwf_count > 0:
        bad = iwf.get("bad_options", [])
        if bad:
            names = ", ".join(b[:50] for b in bad)
            iwf_warn = (
                f'<div class="iwf-warn" style="background:#fff3cd;border:1px solid #ffc107;'
                f'border-radius:6px;padding:6px 10px;margin-bottom:8px;font-size:0.8em;color:#856404">'
                f'&#9888; Cảnh báo IWF: {names}</div>'
            )

    return f"""
<div class="question-card" style="background:#fff;border:1px solid #ddd;border-radius:10px;padding:20px;margin-bottom:24px;box-shadow:0 2px 6px rgba(0,0,0,.06)">
    <!-- Header -->
    <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;margin-bottom:12px">
        <div>
            <span style="font-size:0.75em;color:#888">#{idx+1}</span>
            <strong style="font-size:1.05em;color:#222;margin-left:6px">{qid}</strong>
            &nbsp;&nbsp;{qtype_badge(qtype)}
            &nbsp;<span style="font-size:0.7em;color:#aaa">topic={topic_id} seq={seq}</span>
        </div>
        <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">
            {difficulty_badge(diff)}
            {iwf_html}
            {topic_badge}
        </div>
    </div>

    {sources_html}
    {iwf_warn}

    <!-- Question text -->
    <p style="font-size:1.05em;font-weight:600;line-height:1.5;margin:0 0 14px;color:#2c3e50">{qtext}</p>

    <!-- Options -->
    {options_html}

    <!-- Explanation (only when answers are shown) -->
    {explanation_html}

    <!-- Eval badges -->
    <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-top:8px">
        <span style="font-size:0.8em;color:#555">Eval:</span>
        {eval_badges}
        <span style="font-size:0.8em;color:#555">score={score}</span>
        <span style="font-size:0.75em;color:#aaa;margin-left:8px">CoT steps={n_cot}</span>
    </div>
</div>"""


def _get_sources(sources_map: dict, q: dict) -> list[dict]:
    return sources_map.get(
        (q.get("_meta", {}).get("topic_id", ""),
         q.get("_meta", {}).get("seq", 0)),
        []
    )


def build_html(questions: list[dict], sources_map: dict,
               reveal_correct: bool, title_suffix: str) -> str:
    # Group by chapter
    grouped: dict[str, list] = defaultdict(list)
    for q in questions:
        topic_id = q.get("_meta", {}).get("topic_id", "unknown")
        ch = topic_id.split("_")[0].upper() if "_" in topic_id else "OTHER"
        grouped[ch].append(q)

    # Stats
    total  = len(questions)
    g1     = sum(1 for q in questions if q.get("difficulty_label") == "G1")
    g2     = sum(1 for q in questions if q.get("difficulty_label") == "G2")
    g3     = sum(1 for q in questions if q.get("difficulty_label") == "G3")
    multi  = sum(1 for q in questions if q.get("question_type") == "multiple_correct")
    with_yt = sum(1 for q in questions
                  if any(s.get("source_type") == "video_transcript" and s.get("youtube_url")
                         for s in _get_sources(sources_map, q)))

    answer_label = "Có đáp án" if reveal_correct else "Không có đáp án"
    reveal_note = ("<b style='color:#27ae60'>&#10003; Đáp án đúng được highlight xanh.</b> "
                   "Nhấn 'Xem giải thích' để xem lý do.") if reveal_correct else ""

    # Build all-cards tab
    all_cards = "\n".join(
        build_question_card(q, _get_sources(sources_map, q), i, reveal_correct)
        for i, q in enumerate(questions)
    )

    chapter_cards = ""
    for ch in sorted(grouped.keys()):
        qs = grouped[ch]
        cards = "\n".join(
            build_question_card(q, _get_sources(sources_map, q), i, reveal_correct)
            for i, q in enumerate(qs)
        )
        chapter_cards += f"""
<div class="chapter-block">
    <h2 style="border-bottom:2px solid #2980b9;padding-bottom:6px;color:#2980b9;margin-top:32px">
        &#128218; {ch} <span style="color:#888;font-size:0.7em">({len(qs)} câu)</span>
    </h2>
    {cards}
</div>"""

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CS431MCQGen — Câu hỏi đã chấp nhận ({total} câu) {title_suffix}</title>
<style>
  * {{ box-sizing: border-box; margin:0; padding:0 }}
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    max-width: 900px;
    margin: 0 auto;
    padding: 20px 16px;
    background: #f0f2f5;
    color: #333;
  }}
  h1 {{ text-align:center; color:#2c3e50; margin-bottom:6px }}
  .subtitle {{ text-align:center; margin-bottom:8px; color:#666; font-size:0.9em }}
  .answer-note {{ text-align:center; margin-bottom:20px; color:#555; font-size:0.88em }}
  .stats {{
    text-align:center;
    margin-bottom:24px;
    color:#666;
    font-size:0.9em;
    display:flex;
    flex-wrap:wrap;
    gap:8px;
    justify-content:center;
  }}
  .stats span {{ background:#2980b9; color:#fff; padding:2px 12px; border-radius:12px }}
  .stats span.yt {{ background:#e74c3c }}
  .tab-bar {{
    display:flex;
    gap:4px;
    margin-bottom:20px;
    border-bottom:2px solid #ddd;
    padding-bottom:0;
    flex-wrap:wrap;
  }}
  .tab-btn {{
    padding:8px 16px;
    border:none;
    background:#e0e0e0;
    cursor:pointer;
    border-radius:6px 6px 0 0;
    font-size:0.85em;
  }}
  .tab-btn.active {{ background:#2980b9; color:#fff; font-weight:600 }}
  .tab-content {{ display:none }}
  .tab-content.active {{ display:block }}
  .legend {{
    background:#fff;
    border:1px solid #ddd;
    border-radius:8px;
    padding:12px 16px;
    margin-bottom:20px;
    font-size:0.82em;
    color:#555;
  }}
  .legend b {{ color:#2c3e50 }}
  a {{ color:#2980b9 }}
  details summary::-webkit-details-marker {{ color:#888 }}
</style>
</head>
<body>

<h1>&#128203; CS431MCQGen — Câu hỏi đã chấp nhận ({total} câu) {title_suffix}</h1>
<div class="subtitle">Pipeline: RAG → P1 Stem+Key → P2 Refine → P4-P8 Distractor CoT → Eval (IWF)</div>
<div class="answer-note">{reveal_note}</div>

<div class="stats">
  Tổng: <span>{total} câu</span>
  G1: <span>{g1}</span>
  G2: <span>{g2}</span>
  G3: <span>{g3}</span>
  Nhiều đáp án: <span>{multi}</span>
  Có YouTube: <span class="yt">{with_yt}</span>
</div>

<div class="legend">
  <b>Chú thích nguồn trích dẫn:</b>&nbsp;&nbsp;
  &#127909; <span style="color:#2e7d32;font-weight:600">Nền xanh lục</span> = Video YouTube&nbsp;&nbsp;
  &#128196; <span style="color:#7b1fa2;font-weight:600">Nền tím</span> = Slide PDF
</div>

<!-- Tabs -->
<div class="tab-bar">
  <button class="tab-btn active" onclick="showTab('all')">Tất cả</button>
  <button class="tab-btn" onclick="showTab('g1')">G1</button>
  <button class="tab-btn" onclick="showTab('g2')">G2</button>
  <button class="tab-btn" onclick="showTab('g3')">G3</button>
  <button class="tab-btn" onclick="showTab('multi')">Nhiều đáp án</button>
  <button class="tab-btn" onclick="showTab('chapter')">Theo chương</button>
  <button class="tab-btn" onclick="showTab('yt')">Có YouTube</button>
</div>

<!-- Tab: All -->
<div id="tab-all" class="tab-content active">
  {all_cards}
</div>

<!-- Tab: G1 -->
<div id="tab-g1" class="tab-content">
  {"".join(build_question_card(q, _get_sources(sources_map, q), i, reveal_correct)
           for i, q in enumerate(questions) if q.get("difficulty_label")=="G1")}
</div>

<!-- Tab: G2 -->
<div id="tab-g2" class="tab-content">
  {"".join(build_question_card(q, _get_sources(sources_map, q), i, reveal_correct)
           for i, q in enumerate(questions) if q.get("difficulty_label")=="G2")}
</div>

<!-- Tab: G3 -->
<div id="tab-g3" class="tab-content">
  {"".join(build_question_card(q, _get_sources(sources_map, q), i, reveal_correct)
           for i, q in enumerate(questions) if q.get("difficulty_label")=="G3")}
</div>

<!-- Tab: Multi-correct -->
<div id="tab-multi" class="tab-content">
  {"".join(build_question_card(q, _get_sources(sources_map, q), i, reveal_correct)
           for i, q in enumerate(questions) if q.get("question_type")=="multiple_correct")}
</div>

<!-- Tab: Chapter -->
<div id="tab-chapter" class="tab-content">
  {chapter_cards}
</div>

<!-- Tab: Has YouTube -->
<div id="tab-yt" class="tab-content">
  {"".join(build_question_card(q, _get_sources(sources_map, q), i, reveal_correct)
           for i, q in enumerate(questions)
           if any(s.get("source_type")=="video_transcript" and s.get("youtube_url")
                  for s in _get_sources(sources_map, q)))}
</div>

<script>
  function showTab(name) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    const target = document.getElementById('tab-' + name);
    if (target) target.classList.add('active');
    if (event && event.target) event.target.classList.add('active');
  }}
</script>
</body>
</html>"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading P1 sources: {P1_RESULTS}")
    sources_map = load_p1_sources(P1_RESULTS)
    print(f"  Indexed {len(sources_map)} P1 results")

    print(f"Loading final questions: {FINAL_JSONL}")
    questions = load_final_questions(FINAL_JSONL)
    print(f"  Loaded {len(questions)} questions")

    # Stats
    total  = len(questions)
    multi  = sum(1 for q in questions if q.get("question_type") == "multiple_correct")
    g1     = sum(1 for q in questions if q.get("difficulty_label") == "G1")
    g2     = sum(1 for q in questions if q.get("difficulty_label") == "G2")
    g3     = sum(1 for q in questions if q.get("difficulty_label") == "G3")
    with_yt = sum(1 for q in questions
                   if any(s.get("source_type") == "video_transcript" and s.get("youtube_url")
                          for s in _get_sources(sources_map, q)))
    print(f"\n  Tổng: {total}  |  G1={g1}  G2={g2}  G3={g3}  |  Nhiều đáp án={multi}  |  Có YouTube={with_yt}")

    # ── With answers ──────────────────────────────────────────────────────────
    print("\nBuilding HTML with answers...")
    html_with = build_html(questions, sources_map,
                            reveal_correct=True, title_suffix="(Có đáp án)")
    OUT_WITH.write_text(html_with, encoding="utf-8")
    print(f"  ✅ {OUT_WITH}")

    # ── No answers ────────────────────────────────────────────────────────────
    print("\nBuilding HTML without answers...")
    html_no = build_html(questions, sources_map,
                          reveal_correct=False, title_suffix="(Không có đáp án)")
    OUT_NO.write_text(html_no, encoding="utf-8")
    print(f"  ✅ {OUT_NO}")


if __name__ == "__main__":
    main()
