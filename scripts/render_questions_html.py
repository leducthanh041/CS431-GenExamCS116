"""
render_questions_html.py — Render final accepted MCQs ra HTML
với clickable YouTube timestamp links.

Data flow:
  • Step 03: all_p1_results.jsonl     ← có sources (youtube_url)
  • Step 08: final_accepted_questions.jsonl ← đã qua P1-P8 nhưng chưa có sources
  → Join qua (topic_id, seq) để đưa sources vào render.

Output:
  output/exp_02_baseline/08_eval_iwf/final_accepted_questions.html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

# ── Setup ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # src/ is a subfolder of PROJECT_ROOT

from src.common import _to_seconds, _format_timestamp


# ── Config ───────────────────────────────────────────────────────────────────
P1_RESULTS   = PROJECT_ROOT / "output/exp_02_baseline/03_gen_stem/all_p1_results.jsonl"
FINAL_JSONL  = PROJECT_ROOT / "output/exp_02_baseline/08_eval_iwf/final_accepted_questions.jsonl"
OUTPUT_HTML  = PROJECT_ROOT / "output/exp_02_baseline/08_eval_iwf/final_accepted_questions.html"


# ── Load & Index P1 sources ──────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def difficulty_badge(diff: str) -> str:
    colors = {"G1": "#27ae60", "G2": "#f39c12", "G3": "#e74c3c"}
    color = colors.get(diff, "#7f8c8d")
    return f'<span style="background:{color};color:#fff;padding:2px 10px;border-radius:12px;font-size:0.75em;font-weight:700">{diff}</span>'


def qtype_badge(qtype: str) -> str:
    if qtype == "multiple_correct":
        return '<span style="background:#8e44ad;color:#fff;padding:2px 10px;border-radius:12px;font-size:0.75em;font-weight:700">Nhiều đáp án</span>'
    return '<span style="background:#2980b9;color:#fff;padding:2px 10px;border-radius:12px;font-size:0.75em;font-weight:700">Một đáp án</span>'


def eval_badge(passed: bool, label: str) -> str:
    color = "#27ae60" if passed else "#e74c3c"
    icon  = "&#10003;" if passed else "&#10007;"  # ✓ ✗
    return f'<span style="background:{color};color:#fff;padding:1px 6px;border-radius:8px;font-size:0.7em" title="{label}">{icon}</span>'


def format_seconds_display(seconds: str | int | float | None) -> str:
    """Convert integer seconds → 'MM:SS' or 'HH:MM:SS'."""
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


def build_sources_html(sources: list[dict]) -> str:
    """
    Render list of sources as clickable citation pills.
    Video: <(Trích dẫn: https://youtu.be/abc&t=148 | 2:28 - 3:25)>
    Slide: <(Trích dẫn: CS116-Bai04-Data preprocessing.pdf | Trang 5)>
    """
    if not sources:
        return ""

    items = []
    for i, s in enumerate(sources, 1):
        source_type = s.get("source_type", "")
        youtube_url = s.get("youtube_url", "")
        label = s.get("label", "")

        if source_type == "video_transcript" and youtube_url:
            # Extract timestamp from URL
            ts = ""
            if "&t=" in youtube_url:
                ts_part = youtube_url.split("&t=")[-1].split("&")[0]
                ts = format_seconds_display(ts_part)

            # Build display URL (remove &t= for display)
            display_url = youtube_url.split("&t=")[0]
            if ts:
                display_label = f"{display_url}&amp;t={ts} ({ts})"
                full_href = youtube_url
            else:
                display_label = display_url
                full_href = youtube_url

            items.append(
                f'<span style="display:inline-flex;align-items:center;gap:4px;background:#e8f5e9;border:1px solid #a5d6a7;border-radius:16px;padding:3px 10px;font-size:0.78em">'
                f'[{i}]&nbsp;<a href="{full_href}" target="_blank" style="color:#2e7d32;font-weight:600;text-decoration:none">{ts or display_url}</a>'
                f'</span>'
            )
        elif source_type == "slide_pdf":
            if label:
                items.append(
                    f'<span style="display:inline-flex;align-items:center;gap:4px;background:#f3e5f5;border:1px solid #ce93d8;border-radius:16px;padding:3px 10px;font-size:0.78em">'
                    f'[{i}]&nbsp;<span style="color:#7b1fa2;font-weight:600">{label}</span>'
                    f'</span>'
                )
            elif s.get("chunk_id"):
                items.append(
                    f'<span style="display:inline-flex;align-items:center;gap:4px;background:#f3e5f5;border:1px solid #ce93d8;border-radius:16px;padding:3px 10px;font-size:0.78em">'
                    f'[{i}]&nbsp;<span style="color:#7b1fa2;font-weight:600">{s["chunk_id"]}</span>'
                    f'</span>'
                )

    if not items:
        return ""
    return '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px">' + "".join(items) + "</div>"


def build_options_html(options: dict, correct_letters: list[str],
                     reveal_correct: bool = False) -> str:
    """Render A/B/C/D options with optional correct answer highlighting."""
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
        elif is_correct:
            bg = "#d4edda"
            border = "#27ae60"
            marker = "&#10003;"
        else:
            bg = "#f8f9fa"
            border = "#dee2e6"
            marker = ""
        marker_html = f'<span style="color:#27ae60;font-weight:700">{marker}</span>' if marker else ""
        rows.append(
            f'<div style="background:{bg};border-left:4px solid {border};padding:8px 12px;margin:4px 0;border-radius:4px">'
            f'<strong style="font-family:monospace">{letter}.</strong>&nbsp;{text}&nbsp;{marker_html}'
            f'</div>'
        )
    return '<div style="margin-bottom:14px">' + "".join(rows) + "</div>"


def build_question_card(q: dict, sources: list[dict], idx: int,
                       reveal_correct: bool = False) -> str:
    qid      = q.get("question_id", f"q{idx+1}")
    qtext    = q.get("question_text", "")
    qtype    = q.get("question_type", "single")
    opts     = q.get("options", {})
    correct  = q.get("correct_answers", [])
    topic    = q.get("topic", "")
    diff     = q.get("difficulty_label", "G2")
    eval_    = q.get("evaluation", {})
    iwf      = q.get("distractor_evaluation", {})
    meta     = q.get("_meta", {})
    topic_id = meta.get("topic_id", "?")
    seq      = meta.get("seq", "?")
    n_cot    = len(q.get("_cot_steps", {}))

    # ── Eval badges ───────────────────────────────────────────────
    eval_badges = ""
    for key, label in [
        ("format_pass",        "Format OK"),
        ("language_pass",       "Ngôn ngữ OK"),
        ("grammar_pass",        "Ngữ pháp OK"),
        ("relevance_pass",     "Relevance OK"),
        ("answerability_pass", "Trả lời được"),
        ("correct_set_pass",   "Đáp án đúng"),
        ("no_four_correct_pass","Không 4 đúng"),
        ("answer_not_in_stem_pass","Đáp án không trong stem"),
    ]:
        v = eval_.get(key)
        if v is not None:
            eval_badges += eval_badge(bool(v), label)
    score = eval_.get("quality_score", "?")

    # ── IWF badge ────────────────────────────────────────────────
    iwf_count = q.get("final_iwf_count", iwf.get("total_iwf_count", 0))
    iwf_color = "#27ae60" if iwf_count == 0 else "#e74c3c"
    iwf_html  = f'<span style="background:{iwf_color};color:#fff;padding:2px 8px;border-radius:8px;font-size:0.75em">IWF={iwf_count}</span>'

    # ── Topic badge ───────────────────────────────────────────────
    topic_badge = ""
    if topic:
        topic_badge = f'<span style="background:#2c3e50;color:#fff;padding:2px 8px;border-radius:8px;font-size:0.75em">{topic[:60]}</span>'

    # ── Sources ─────────────────────────────────────────────────
    sources_html = build_sources_html(sources)

    # ── Options ──────────────────────────────────────────────────
    options_html = build_options_html(opts, correct, reveal_correct=False)

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

    <!-- Sources (clickable YouTube links) -->
    {sources_html}

    <!-- Question text -->
    <p style="font-size:1.05em;font-weight:600;line-height:1.5;margin:0 0 14px;color:#2c3e50">{qtext}</p>

    <!-- Options -->
    {options_html}

    <!-- Eval badges -->
    <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:4px">
        <span style="font-size:0.8em;color:#555">Eval:</span>
        {eval_badges}
        <span style="font-size:0.8em;color:#555">score={score}</span>
        <span style="font-size:0.75em;color:#aaa;margin-left:8px">CoT steps={n_cot}</span>
    </div>
    <p style="font-size:0.75em;color:#aaa;margin:0">
        fail_reasons: {', '.join(eval_.get('fail_reasons', [])) or '—'}
    </p>
</div>"""


# ── HTML builder ─────────────────────────────────────────────────────────────

def _get_sources(sources_map: dict, q: dict) -> list[dict]:
    """Get sources for a question by looking up (topic_id, seq) in sources_map."""
    return sources_map.get(
        (q.get("_meta", {}).get("topic_id", ""),
         q.get("_meta", {}).get("seq", 0)),
        []
    )


def _build_yt_card(q: dict, sources_map: dict, idx: int) -> str:
    """Alias for building a question card in the YouTube tab."""
    return build_question_card(q, _get_sources(sources_map, q), idx)


def build_html(questions: list[dict], sources_map: dict) -> str:
    # Group by chapter prefix
    grouped: dict[str, list] = defaultdict(list)
    for q in questions:
        meta = q.get("_meta", {})
        topic_id = meta.get("topic_id", "unknown")
        ch = topic_id.split("_")[0].upper() if "_" in topic_id else "OTHER"
        grouped[ch].append(q)

    # Stats
    total   = len(questions)
    g1      = sum(1 for q in questions if q.get("difficulty_label") == "G1")
    g2      = sum(1 for q in questions if q.get("difficulty_label") == "G2")
    g3      = sum(1 for q in questions if q.get("difficulty_label") == "G3")
    multi   = sum(1 for q in questions if q.get("question_type") == "multiple_correct")
    with_yt = sum(1 for q in questions
                  if sources_map.get((q.get("_meta", {}).get("topic_id", ""),
                                     q.get("_meta", {}).get("seq", 0))))

    all_cards = "\n".join(
        build_question_card(q, _get_sources(sources_map, q), i)
        for i, q in enumerate(questions)
    )

    chapter_cards = ""
    for ch in sorted(grouped.keys()):
        qs = grouped[ch]
        cards = "\n".join(
            build_question_card(q, _get_sources(sources_map, q), i)
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
<title>CS431MCQGen — Câu hỏi đã chấp nhận ({total} câu)</title>
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
  .subtitle {{ text-align:center; margin-bottom:24px; color:#666; font-size:0.9em }}
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
  .stats span {{
    background:#2980b9;
    color:#fff;
    padding:2px 12px;
    border-radius:12px;
  }}
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
  .iwf-warn {{
    background:#fff3cd;
    border:1px solid #ffc107;
    border-radius:6px;
    padding:10px 14px;
    margin-bottom:24px;
    font-size:0.85em;
  }}
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
</style>
</head>
<body>

<h1>&#128203; CS431MCQGen — Câu hỏi đã chấp nhận</h1>
<div class="subtitle">Pipeline: RAG → P1 Stem+Key → P2 Refine → P4-P8 Distractor CoT → Eval (IWF)</div>

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
  &#127909; <span style="color:#2e7d32;font-weight:600">Nền xanh lục</span> = Video YouTube (bấm → nhảy đến đúng thời điểm)&nbsp;&nbsp;
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
  {''.join(build_question_card(q, _get_sources(sources_map, q), i)
    for i, q in enumerate(questions) if q.get("difficulty_label")=="G1")}
</div>

<!-- Tab: G2 -->
<div id="tab-g2" class="tab-content">
  {''.join(build_question_card(q, _get_sources(sources_map, q), i)
    for i, q in enumerate(questions) if q.get("difficulty_label")=="G2")}
</div>

<!-- Tab: G3 -->
<div id="tab-g3" class="tab-content">
  {''.join(build_question_card(q, _get_sources(sources_map, q), i)
    for i, q in enumerate(questions) if q.get("difficulty_label")=="G3")}
</div>

<!-- Tab: Multi-correct -->
<div id="tab-multi" class="tab-content">
  {''.join(build_question_card(q, _get_sources(sources_map, q), i)
    for i, q in enumerate(questions) if q.get("question_type")=="multiple_correct")}
</div>

<!-- Tab: Chapter -->
<div id="tab-chapter" class="tab-content">
  {chapter_cards}
</div>

<!-- Tab: Has YouTube -->
<div id="tab-yt" class="tab-content">
  {''.join(_build_yt_card(q, sources_map, i) for i, q in enumerate(questions)
    if any(s.get("source_type") == "video_transcript" and s.get("youtube_url")
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

    print("Building HTML...")
    html = build_html(questions, sources_map)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"✅ Saved: {OUTPUT_HTML}")

    # Quick stats
    yt_count = sum(
        1 for q in questions
        if any(s.get("source_type") == "video_transcript" and s.get("youtube_url")
               for s in sources_map.get(
                   (q.get("_meta", {}).get("topic_id", ""),
                    q.get("_meta", {}).get("seq", 0)), []))
    )
    print(f"   Questions with YouTube citation: {yt_count}/{len(questions)}")


if __name__ == "__main__":
    main()
