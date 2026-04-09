"""
render_questions_html.py
Chuyển final_accepted_questions.jsonl → HTML để xem/trình bày câu hỏi MCQ.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE   = PROJECT_ROOT / "output/exp_01_baseline/08_eval_iwf/final_accepted_questions.jsonl"
OUTPUT_FILE  = PROJECT_ROOT / "output/exp_01_baseline/08_eval_iwf/final_accepted_questions.html"


def load_questions(path: Path) -> list[dict]:
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def difficulty_badge(difficulty: str) -> str:
    colors = {"G1": "#27ae60", "G2": "#f39c12", "G3": "#e74c3c"}
    color = colors.get(difficulty, "#7f8c8d")
    return f'<span style="background:{color};color:#fff;padding:2px 10px;border-radius:12px;font-size:0.75em;font-weight:700">{difficulty}</span>'


def question_type_label(qtype: str) -> str:
    if qtype == "multiple_correct":
        return '<span style="background:#8e44ad;color:#fff;padding:2px 10px;border-radius:12px;font-size:0.75em;font-weight:700">Nhiều đáp án</span>'
    return '<span style="background:#2980b9;color:#fff;padding:2px 10px;border-radius:12px;font-size:0.75em;font-weight:700">Một đáp án</span>'


def eval_badge(passed: bool, label: str) -> str:
    color = "#27ae60" if passed else "#e74c3c"
    icon  = "✓" if passed else "✗"
    return f'<span style="background:{color};color:#fff;padding:1px 6px;border-radius:8px;font-size:0.7em" title="{label}">{icon}</span>'


def build_question_card(q: dict, idx: int) -> str:
    qid    = q.get("question_id", f"q{idx+1}")
    qtext  = q.get("question_text", "")
    qtype  = q.get("question_type", "single")
    opts   = q.get("options", {})
    correct = q.get("correct_answers", [])
    topic   = q.get("topic", q.get("_meta", {}).get("topic_name", ""))
    diff    = q.get("difficulty_label", q.get("_meta", {}).get("difficulty", "G2"))
    eval_   = q.get("evaluation", {})
    iwf     = q.get("distractor_evaluation", {})
    status  = q.get("status", "")

    # ── answer toggles ──────────────────────────────────────────────────
    options_html = ""
    for letter in ["A", "B", "C", "D"]:
        text = opts.get(letter, "")
        is_correct = letter in correct
        bg = "#d4edda" if is_correct else "#f8f9fa"
        border = "#27ae60" if is_correct else "#dee2e6"
        marker = "✅ " if is_correct else "  "
        options_html += f"""
        <div style="background:{bg};border-left:4px solid {border};padding:8px 12px;margin:4px 0;border-radius:4px;font-family:monospace">
            <strong>{letter}.</strong> {text}{marker}
        </div>"""

    # ── eval checklist ─────────────────────────────────────────────────
    checks = ""
    checks += eval_badge(eval_.get("format_pass", False), "Format OK")
    checks += eval_badge(eval_.get("language_pass", False), "Ngôn ngữ OK")
    checks += eval_badge(eval_.get("grammar_pass", False), "Ngữ pháp OK")
    checks += eval_badge(eval_.get("relevance_pass", False), "Relevance OK")
    checks += eval_badge(eval_.get("answerability_pass", False), "Trả lời được")
    checks += eval_badge(eval_.get("correct_set_pass", False), "Đáp án đúng")
    checks += f' <span style="font-size:0.8em;color:#555">score={eval_.get("quality_score","?")}</span>'

    # ── IWF badges ─────────────────────────────────────────────────────
    iwf_count = iwf.get("total_iwf_count", 0)
    iwf_color = "#27ae60" if iwf_count == 0 else "#e74c3c"
    iwf_html  = f'<span style="background:{iwf_color};color:#fff;padding:2px 8px;border-radius:8px;font-size:0.75em">IWF={iwf_count}</span>'

    # ── meta chips ─────────────────────────────────────────────────────
    meta = q.get("_meta", {})
    p4_count  = meta.get("p4_candidates_count", "?")
    p7_count  = meta.get("p7_selected_count", "?")
    assembly  = meta.get("p8_assembly_method", "?")

    # ── wrap with collapsible CoT detail ──────────────────────────────
    cot_steps = q.get("_cot_steps", {})
    cot_detail = ""
    if cot_steps:
        p5_evals = cot_steps.get("p5", {}).get("evaluations", [])
        p7_selected = cot_steps.get("p7", {}).get("selected_distractors", [])

        p5_rows = ""
        for ev in p5_evals:
            mark = "❌" if not ev.get("is_correct", True) else "✓"
            p5_rows += f"""
            <tr>
                <td style="padding:4px 8px">{mark}</td>
                <td style="padding:4px 8px;font-size:0.8em">{ev.get("option_text","")[:80]}</td>
                <td style="padding:4px 8px">{ev.get("misleading_likelihood","?")}</td>
                <td style="padding:4px 8px;font-size:0.8em">{ev.get("logical_error","—")}</td>
            </tr>"""

        p7_rows = ""
        for d in p7_selected:
            p7_rows += f"""<li style="font-size:0.85em">"{d.get('option_text','?')[:80]}"
                <em>(error={d.get('error_type','?')}, misleading={d.get('misleading_score','?')})</em></li>"""

        cot_detail = f"""
        <details style="margin-top:10px;background:#f8f9fa;border-radius:6px;padding:8px 12px">
            <summary style="cursor:pointer;font-weight:600;color:#2980b9">🔍 Chi tiết CoT (P4–P7)</summary>
            <div style="margin-top:8px">
                <p style="margin:4px 0"><strong>P4 candidates:</strong> {p4_count} |
                   <strong>P7 selected:</strong> {p7_count} |
                   <strong>Assembly:</strong> {assembly}</p>

                <p style="margin:4px 0;font-weight:600">P5 evaluations:</p>
                <table style="width:100%;border-collapse:collapse;font-size:0.85em">
                    <tr style="background:#bdc3c7">
                        <th>#</th><th>Option</th><th>Mislead</th><th>Error</th>
                    </tr>
                    {p5_rows}
                </table>

                <p style="margin:8px 0 4px;font-weight:600">P7 selected distractors:</p>
                <ul style="margin:0;padding-left:20px">{p7_rows}</ul>
            </div>
        </details>"""

    return f"""
<div class="question-card" style="background:#fff;border:1px solid #ddd;border-radius:10px;padding:20px;margin-bottom:24px;box-shadow:0 2px 6px rgba(0,0,0,.06)">
    <!-- Header -->
    <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;margin-bottom:12px">
        <div>
            <span style="font-size:0.75em;color:#888">#{idx+1}</span>
            <strong style="font-size:1.05em;color:#222;margin-left:6px">{qid}</strong>
            &nbsp;&nbsp;{question_type_label(qtype)}
        </div>
        <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">
            {difficulty_badge(diff)}
            {iwf_html}
            <span style="background:#2c3e50;color:#fff;padding:2px 8px;border-radius:8px;font-size:0.75em">{topic}</span>
        </div>
    </div>

    <!-- Question text -->
    <p style="font-size:1.05em;font-weight:600;line-height:1.5;margin:0 0 14px;color:#2c3e50">{qtext}</p>

    <!-- Options -->
    <div style="margin-bottom:14px">{options_html}</div>

    <!-- Eval checklist -->
    <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:6px">
        <span style="font-size:0.8em;color:#555">Eval:</span>
        {checks}
    </div>
    <p style="font-size:0.75em;color:#aaa;margin:0">
        fail_reasons: {', '.join(eval_.get('fail_reasons', [])) or '—'}
    </p>

    {cot_detail}
</div>"""


def build_html(questions: list[dict]) -> str:
    # Group by chapter/topic
    grouped: dict[str, list] = {}
    for q in questions:
        meta = q.get("_meta", {})
        topic_id = meta.get("topic_id", "unknown")
        # Extract chapter prefix
        if "_" in topic_id:
            ch = topic_id.split("_")[0].upper()
        else:
            ch = "OTHER"
        grouped.setdefault(ch, []).append(q)

    cards_by_chapter = ""
    for ch in sorted(grouped.keys()):
        qs = grouped[ch]
        cards = "\n".join(build_question_card(q, i) for i, q in enumerate(qs))
        cards_by_chapter += f"""
<div class="chapter-block">
    <h2 style="border-bottom:2px solid #2980b9;padding-bottom:6px;color:#2980b9;margin-top:32px">
        📖 {ch} <span style="color:#888;font-size:0.7em">({len(qs)} câu)</span>
    </h2>
    {cards}
</div>"""

    all_cards = "\n".join(build_question_card(q, i) for i, q in enumerate(questions))

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CS431MCQGen — Câu hỏi đã chấp nhận ({len(questions)} câu)</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    max-width: 860px;
    margin: 0 auto;
    padding: 20px 16px;
    background: #f0f2f5;
    color: #333;
  }}
  h1 {{
    text-align: center;
    color: #2c3e50;
    margin-bottom: 6px;
  }}
  .stats {{
    text-align: center;
    margin-bottom: 24px;
    color: #666;
    font-size: 0.9em;
  }}
  .stats span {{
    background: #2980b9;
    color: #fff;
    padding: 2px 10px;
    border-radius: 12px;
    margin: 0 4px;
  }}
  .tab-bar {{
    display: flex;
    gap: 4px;
    margin-bottom: 20px;
    border-bottom: 2px solid #ddd;
    padding-bottom: 0;
  }}
  .tab-btn {{
    padding: 8px 16px;
    border: none;
    background: #e0e0e0;
    cursor: pointer;
    border-radius: 6px 6px 0 0;
    font-size: 0.85em;
  }}
  .tab-btn.active {{
    background: #2980b9;
    color: #fff;
    font-weight: 600;
  }}
  .tab-content {{
    display: none;
  }}
  .tab-content.active {{
    display: block;
  }}
  .iwf-warn {{
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 24px;
    font-size: 0.85em;
  }}
</style>
</head>
<body>

<h1>📝 CS431MCQGen — Câu hỏi đã chấp nhận</h1>
<div class="stats">
  Tổng: <span>{len(questions)} câu</span>
  &nbsp;|&nbsp;
  G1: <span>{sum(1 for q in questions if q.get('difficulty_label','G2')=='G1')}</span>
  &nbsp;|&nbsp;
  G2: <span>{sum(1 for q in questions if q.get('difficulty_label','G2')=='G2')}</span>
  &nbsp;|&nbsp;
  G3: <span>{sum(1 for q in questions if q.get('difficulty_label','G2')=='G3')}</span>
  &nbsp;|&nbsp;
  Multi-correct: <span>{sum(1 for q in questions if q.get('question_type')=='multiple_correct')}</span>
</div>

<!-- Tabs -->
<div class="tab-bar">
  <button class="tab-btn active" onclick="showTab('all')">Tất cả</button>
  <button class="tab-btn" onclick="showTab('g1')">G1</button>
  <button class="tab-btn" onclick="showTab('g2')">G2</button>
  <button class="tab-btn" onclick="showTab('g3')">G3</button>
  <button class="tab-btn" onclick="showTab('multi')">Nhiều đáp án</button>
</div>

<!-- Tab: All -->
<div id="tab-all" class="tab-content active">
  {all_cards}
</div>

<!-- Tab: G1 -->
<div id="tab-g1" class="tab-content">
  {''.join(build_question_card(q, i) for i, q in enumerate(questions) if q.get('difficulty_label','G2')=='G1')}
</div>

<!-- Tab: G2 -->
<div id="tab-g2" class="tab-content">
  {''.join(build_question_card(q, i) for i, q in enumerate(questions) if q.get('difficulty_label','G2')=='G2')}
</div>

<!-- Tab: G3 -->
<div id="tab-g3" class="tab-content">
  {''.join(build_question_card(q, i) for i, q in enumerate(questions) if q.get('difficulty_label','G2')=='G3')}
</div>

<!-- Tab: Multi-correct -->
<div id="tab-multi" class="tab-content">
  {''.join(build_question_card(q, i) for i, q in enumerate(questions) if q.get('question_type')=='multiple_correct')}
</div>

<script>
  function showTab(name) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    event.target.classList.add('active');
  }}
</script>
</body>
</html>"""


def main():
    print(f"Loading: {INPUT_FILE}")
    questions = load_questions(INPUT_FILE)
    print(f"Loaded {len(questions)} questions")

    html = build_html(questions)
    OUTPUT_FILE.write_text(html, encoding="utf-8")
    print(f"✅ Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
