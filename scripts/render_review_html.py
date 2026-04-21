"""
render_review_html.py — Human Review Interface for MCQ Evaluation
================================================================
Exports final_accepted_questions.jsonl → interactive HTML for human evaluation.
Outputs JSON annotation file (Format A) compatible with
compute_human_judgment() in src/eval/eval_metrics.py.

Workflow:
  1. python scripts/render_review_html.py --exp "exp_04"
     → output/exp_04/review/eval_review.html
  2. Open HTML → review questions → vote → Export JSON
     → output/exp_04/review/eval_review_annotations.json
  3. python src/eval/eval_metrics.py --exp "exp_04" --human-json output/exp_04/review/eval_review_annotations.json

Export format (Format A — matches eval_metrics.py):
  {
    "annotator": "...",
    "timestamp": "...",
    "total_annotated": 10,
    "verdicts": {
      "<question_id>": {
        "format_pass": true,
        "language_pass": true,
        ...
        "overall_valid": true
      }
    }
  }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import Config, load_jsonl


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_gen_mcqs() -> list[dict]:
    """Load accepted questions (final output after IWF filter)."""
    path = Config.EVAL_IWF_OUTPUT / "final_accepted_questions.jsonl"
    if not path.exists():
        path = Config.GEN_COT_OUTPUT / "all_final_mcqs.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No questions found at {path}")
    return load_jsonl(path)


def load_evaluated() -> dict[str, dict]:
    """Load LLM Step-07 evaluations → dict keyed by question_id."""
    path = Config.EVAL_OUTPUT / "evaluated_questions.jsonl"
    if path.exists():
        return {
            q.get("question_id", f"q{i}"): q
            for i, q in enumerate(load_jsonl(path))
        }
    return {}


# ── Small UI helpers ─────────────────────────────────────────────────────────

def difficulty_badge(d: str) -> str:
    colors = {"G1": "#27ae60", "G2": "#f39c12", "G3": "#e74c3c"}
    return (
        f'<span class="diff-badge diff-{d.lower()}">{d}</span>'
        if False  # class-based; actual inline below
        else f'<span style="background:{colors.get(d,"#7f8c8d")};color:#fff;'
              f'padding:2px 10px;border-radius:12px;font-size:0.75em;font-weight:700">{d}</span>'
    )


def iwf_badge(passed: bool | None) -> str:
    if passed is True:
        return '<span style="background:#27ae60;color:#fff;padding:2px 8px;border-radius:8px;font-size:0.7em;font-weight:700">IWF ✅</span>'
    if passed is False:
        return '<span style="background:#e74c3c;color:#fff;padding:2px 8px;border-radius:8px;font-size:0.7em;font-weight:700">IWF ❌</span>'
    return '<span style="background:#e9ecef;color:#6c757d;padding:2px 8px;border-radius:8px;font-size:0.7em">IWF —</span>'


# ── Review card builder ──────────────────────────────────────────────────────

def build_review_card(
    q: dict,
    idx: int,
    llm_eval: dict | None,
    llm_iwf: dict | None,
    chapter: str = "",
) -> str:
    qid    = q.get("question_id", f"q{idx+1}")
    qtext  = q.get("question_text", "")
    qtype  = q.get("question_type", "single")
    opts   = q.get("options", {})
    correct = q.get("correct_answers", [])
    topic   = q.get("topic", q.get("_meta", {}).get("topic_name", ""))
    diff   = q.get("difficulty_label", q.get("_meta", {}).get("difficulty", "G2"))
    eval_  = llm_eval.get("evaluation", {}) if llm_eval else {}
    iwf_e  = llm_iwf.get("distractor_evaluation", {}) if llm_iwf else {}
    ch     = chapter or "OTHER"

    # ── Options (show all, green border = correct) ───────────────────────
    options_html = ""
    for letter in ["A", "B", "C", "D"]:
        text       = opts.get(letter, "")
        is_correct = letter in correct
        bg     = "#d4edda" if is_correct else "#f8f9fa"
        border = "#27ae60" if is_correct else "#dee2e6"
        marker = " ✅" if is_correct else ""
        options_html += (
            f'<div style="background:{bg};border-left:4px solid {border};'
            f'padding:7px 12px;margin:3px 0;border-radius:4px">'
            f'<strong>{letter}.</strong> {text}{marker}</div>'
        )

    # ── Criterion rows (6 criteria + overall) ─────────────────────────────
    criteria = [
        ("format_pass",        "1. Format"),
        ("language_pass",      "2. Ngôn ngữ"),
        ("grammar_pass",       "3. Ngữ pháp"),
        ("relevance_pass",     "4. Relevance"),
        ("answerability_pass", "5. Trả lời được"),
        ("correct_set_pass",   "6. Đáp án đúng"),
    ]

    def _row(criterion: str, label: str, llm_val: bool | None) -> str:
        llm_icon  = "✅" if llm_val else "❌" if llm_val is not None else "—"
        llm_class = "llm-pass" if llm_val else "llm-fail" if llm_val is not None else "llm-unknown"
        return (
            f'<tr data-criterion="{criterion}" class="criterion-row">'
            f'<td style="padding:6px 10px;width:38%">'
            f'<strong>{label}</strong>'
            f'<div style="font-size:0.72em;color:#aaa;margin-top:1px">{criterion}</div>'
            f'</td>'
            f'<td style="padding:6px 10px;text-align:center">'
            f'<span class="llm-badge {llm_class}">{llm_icon}</span>'
            f'</td>'
            f'<td style="padding:6px 10px;text-align:center">'
            f'<button class="vote-btn pass" '
            f'onclick="vote(this,\'{qid}\',\'{criterion}\',true)">✅ Pass</button>'
            f'<button class="vote-btn fail" '
            f'onclick="vote(this,\'{qid}\',\'{criterion}\',false)">❌ Fail</button>'
            f'</td>'
            f'<td style="padding:6px 10px;width:22%;text-align:center">'
            f'<span class="verdict-display" id="verdict-{qid}-{criterion}">—</span>'
            f'</td>'
            f'</tr>'
        )

    criterion_rows = "\n".join(
        _row(c, label, eval_.get(c))
        for c, label in criteria
    )

    # ── Overall row ───────────────────────────────────────────────────────
    llm_overall = eval_.get("overall_valid")
    llm_icon  = "✅" if llm_overall else "❌" if llm_overall is not None else "—"
    llm_class = "llm-pass" if llm_overall else "llm-fail" if llm_overall is not None else "llm-unknown"

    quality = eval_.get("quality_score", "?")
    fail_reasons = eval_.get("fail_reasons", [])

    return f"""
<div class="review-card" id="card-{qid}" data-chapter="{ch}"
     style="background:#fff;border:2px solid #ddd;border-radius:10px;
            padding:18px;margin-bottom:20px;box-shadow:0 2px 6px rgba(0,0,0,.06)">

    <!-- Header -->
    <div style="display:flex;justify-content:space-between;align-items:flex-start;
                flex-wrap:wrap;gap:8px;margin-bottom:12px">
        <div>
            <span style="font-size:0.72em;color:#aaa">#{idx+1}</span>
            <strong style="font-size:1em;color:#222;margin-left:6px">{qid}</strong>
        </div>
        <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">
            {difficulty_badge(diff)}
            {iwf_badge(iwf_e.get("overall_distractor_quality_pass"))}
            <span style="background:#2980b9;color:#fff;padding:2px 8px;
                         border-radius:8px;font-size:0.72em">{topic}</span>
        </div>
    </div>

    <!-- Question text -->
    <p style="font-size:1em;font-weight:600;line-height:1.5;margin:0 0 12px;color:#2c3e50">
        {qtext}
    </p>

    <!-- Options (collapsible) -->
    <details>
        <summary style="cursor:pointer;font-size:0.85em;color:#2980b9;
                        font-weight:600;margin-bottom:8px">
            👁 Show Options + Correct Answers
        </summary>
        <div style="margin-bottom:10px">{options_html}</div>
    </details>

    <!-- Review Table -->
    <table style="width:100%;border-collapse:collapse;margin-top:12px">
        <tr style="background:#34495e;color:#fff;font-size:0.78em">
            <th style="padding:6px 10px;text-align:left">Tiêu chí</th>
            <th style="padding:6px 10px;text-align:center">LLM Judge</th>
            <th style="padding:6px 10px;text-align:center">Human Vote</th>
            <th style="padding:6px 10px;text-align:center">Verdict</th>
        </tr>
        {criterion_rows}
        <!-- Overall Valid -->
        <tr style="background:#f0f0f0;font-weight:600">
          <td style="padding:8px 10px">✅ Overall Valid</td>
          <td style="padding:8px 10px;text-align:center">
            <span class="llm-badge {llm_class}">{llm_icon}</span>
          </td>
          <td style="padding:8px 10px;text-align:center">
            <button class="vote-btn pass"
                    onclick="vote(this,\'{qid}\',\'overall_valid\',true)">✅ Pass</button>
            <button class="vote-btn fail"
                    onclick="vote(this,\'{qid}\',\'overall_valid\',false)">❌ Fail</button>
          </td>
          <td style="padding:8px 10px;text-align:center">
            <span class="verdict-display" id="verdict-{qid}-overall_valid">—</span>
          </td>
        </tr>
    </table>

    <!-- LLM info bar -->
    <div style="margin-top:8px;font-size:0.78em;color:#555;display:flex;gap:12px;flex-wrap:wrap">
        <span><strong>Quality Score:</strong> {quality}</span>
        <span><strong>Fail reasons:</strong> {', '.join(fail_reasons) or '—'}</span>
        <span><strong>IWF verdict:</strong> {'Pass' if iwf_e.get('overall_distractor_quality_pass') else 'Fail' if iwf_e.get('overall_distractor_quality_pass') is False else '—'}</span>
    </div>

    <!-- Notes -->
    <div style="margin-top:8px">
        <textarea
            id="notes-{qid}"
            placeholder="Ghi chú của người review..."
            rows="2"
            style="width:100%;padding:6px;border-radius:6px;border:1px solid #ccc;
                   font-size:0.82em;resize:vertical"
        ></textarea>
    </div>

    <!-- Submit -->
    <div style="margin-top:8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <span class="submit-status" id="status-{qid}" style="font-size:0.78em;color:#888"></span>
        <button onclick="submitVerdict(\'{qid}\')"
                style="background:#2980b9;color:#fff;border:none;padding:5px 14px;
                       border-radius:6px;cursor:pointer;font-size:0.82em">
            Submit
        </button>
        <span style="font-size:0.72em;color:#aaa">
            Progress: <span id="progress-{qid}">0/7</span>/7
        </span>
    </div>
</div>"""


# ── HTML page builder ─────────────────────────────────────────────────────────

def build_html(
    questions: list[dict],
    llm_evals: dict[str, dict],
    llm_iwf_map: dict[str, dict],
    exp_name: str,
) -> str:
    # Chapter grouping
    chapter_of: dict[str, str] = {}
    grouped: dict[str, list] = {}
    for q in questions:
        meta = q.get("_meta", {})
        topic_id = meta.get("topic_id", "unknown")
        ch = topic_id.split("_")[0].upper() if "_" in topic_id else "OTHER"
        qid = q.get("question_id", "")
        chapter_of[qid] = ch
        grouped.setdefault(ch, []).append(q)

    all_cards = "\n".join(
        build_review_card(
            q, i,
            llm_evals.get(q.get("question_id", "")),
            llm_iwf_map.get(q.get("question_id", "")),
            chapter_of.get(q.get("question_id", ""), "OTHER"),
        )
        for i, q in enumerate(questions)
    )

    tab_buttons = (
        '<button class="tab-btn active" onclick="showTab(\'all\')">Tất cả</button>'
        + "".join(
            f'<button class="tab-btn" onclick="showTab(\'{ch}\')">{ch} ({len(qs)})</button>'
            for ch, qs in sorted(grouped.items())
        )
    )

    # Count IWF pass/fail
    iwf_pass  = sum(1 for q in questions
                    if llm_iwf_map.get(q.get("question_id", ""), {})
                       .get("distractor_evaluation", {})
                       .get("overall_distractor_quality_pass") is True)
    iwf_fail  = sum(1 for q in questions
                    if llm_iwf_map.get(q.get("question_id", ""), {})
                       .get("distractor_evaluation", {})
                       .get("overall_distractor_quality_pass") is False)
    missing_eval = sum(1 for q in questions if q.get("question_id", "") not in llm_evals)

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Human Review — {exp_name} ({len(questions)} câu)</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    max-width: 960px;
    margin: 0 auto;
    padding: 16px;
    background: #f0f2f5;
    color: #333;
  }}
  h1 {{ text-align: center; color: #2c3e50; margin-bottom: 4px; }}
  .subtitle {{
    text-align: center; color: #888; font-size: 0.82em;
    margin-bottom: 16px;
  }}
  .summary-bar {{
    background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
    padding: 10px 16px; margin-bottom: 16px; display: flex;
    gap: 16px; flex-wrap: wrap; font-size: 0.82em;
  }}
  .summary-bar .stat {{ display: flex; gap: 6px; align-items: center; }}
  .summary-bar .stat-val {{ font-weight: 700; color: #2c3e50; }}
  .instructions {{
    background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 16px;
    font-size: 0.85em; line-height: 1.6;
  }}
  .instructions strong {{ color: #856404; }}
  .tab-bar {{
    display: flex; gap: 4px; margin-bottom: 12px;
    border-bottom: 2px solid #ddd; flex-wrap: wrap;
  }}
  .tab-btn {{
    padding: 7px 14px; border: none; background: #e0e0e0; cursor: pointer;
    border-radius: 6px 6px 0 0; font-size: 0.82em;
  }}
  .tab-btn.active {{ background: #2980b9; color: #fff; font-weight: 600; }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}

  /* Vote buttons */
  .vote-btn {{
    padding: 3px 9px; border-radius: 6px; border: 2px solid transparent;
    cursor: pointer; font-size: 0.80em; margin: 0 2px; transition: all 0.12s;
  }}
  .vote-btn.pass {{ background: #e8f5e9; border-color: #a5d6a7; color: #2e7d32; }}
  .vote-btn.pass.selected {{ background: #4caf50; border-color: #388e3c; color: #fff; }}
  .vote-btn.fail {{ background: #ffebee; border-color: #ef9a9a; color: #c62828; }}
  .vote-btn.fail.selected {{ background: #f44336; border-color: #d32f2f; color: #fff; }}

  /* LLM badge */
  .llm-badge {{
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-weight: 700; font-size: 0.88em;
  }}
  .llm-pass  {{ background: #d4edda; color: #155724; }}
  .llm-fail  {{ background: #f8d7da; color: #721c24; }}
  .llm-unknown {{ background: #e9ecef; color: #6c757d; }}

  /* Verdict display */
  .verdict-display {{ font-weight: 700; font-size: 0.88em; }}
  .verdict-pass {{ color: #4caf50; }}
  .verdict-fail {{ color: #f44336; }}
  .verdict-pending {{ color: #aaa; }}

  /* Card states */
  .criterion-row:hover {{ background: #f8f9fa; }}
  .review-card.submitted {{ opacity: 0.55; }}
  .review-card.no-eval {{ border-color: #f39c12; }}

  /* Status */
  .submit-status.submitted {{ color: #4caf50; font-weight: 600; }}
  .submit-status.in-progress {{ color: #ff9800; }}
  .submit-status.missing {{ color: #e74c3c; }}

  /* Toolbar */
  #toolbar {{
    position: sticky; bottom: 0; background: #2c3e50; color: #fff;
    padding: 10px 16px; border-radius: 10px 10px 0 0;
    display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
    z-index: 100; box-shadow: 0 -2px 10px rgba(0,0,0,.2);
  }}
  #toolbar button {{
    padding: 7px 14px; border-radius: 6px; border: none;
    cursor: pointer; font-size: 0.82em; font-weight: 600;
  }}
  #toolbar .btn-success {{ background: #27ae60; color: #fff; }}
  #toolbar .btn-warning {{ background: #f39c12; color: #fff; }}
  #toolbar .btn-danger  {{ background: #e74c3c; color: #fff; }}
  #toolbar .btn-info    {{ background: #2980b9; color: #fff; }}
  #progress-overall {{
    background: #fff; color: #2c3e50; padding: 4px 12px;
    border-radius: 12px; font-weight: 700;
  }}
  #annotator-input {{
    padding: 6px 10px; border-radius: 6px; border: none;
    font-size: 0.82em; width: 140px;
  }}
</style>
</head>
<body>

<h1>📝 Human Review — MCQGen</h1>
<div class="subtitle">{exp_name} · {len(questions)} câu hỏi</div>

<!-- Summary bar -->
<div class="summary-bar">
  <div class="stat">✅ IWF Pass: <span class="stat-val">{iwf_pass}</span></div>
  <div class="stat">❌ IWF Fail: <span class="stat-val">{iwf_fail}</span></div>
  <div class="stat">❓ Missing LLM eval: <span class="stat-val">{missing_eval}</span></div>
</div>

<div class="instructions">
  <strong>Hướng dẫn:</strong> Mỗi câu hỏi có <strong>6 tiêu chí + 1 overall</strong>.
  Với mỗi tiêu chí, nhấn <strong>✅ Pass</strong> (đạt) hoặc
  <strong>❌ Fail</strong> (không đạt).
  Khi đã vote đủ 7 tiêu chí cho 1 câu, nhấn <strong>Submit</strong>.
  Cuối cùng nhấn <strong>📥 Export JSON</strong> để tải file annotation —
  dùng với <code>eval_metrics.py --human-json</code>.
</div>

<!-- Tabs -->
<div class="tab-bar">
  {tab_buttons}
</div>

<!-- All questions -->
<div id="tab-all" class="tab-content active">
  {all_cards}
</div>

<!-- Sticky toolbar -->
<div id="toolbar">
  <span style="font-size:0.8em">Annotator:</span>
  <input id="annotator-input" type="text" placeholder="Tên của bạn" value="">
  <span id="progress-overall">0/{len(questions)} submitted</span>
  <button class="btn-success" onclick="exportJSON()">📥 Export JSON</button>
  <button class="btn-info"    onclick="markAllSubmitted()">✅ Submit all</button>
  <button class="btn-warning" onclick="markAllPass()">🎯 Auto-pass all</button>
  <button class="btn-danger"  onclick="clearAll()">🗑️ Clear</button>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
const STORAGE_KEY = 'mcq_review_v3';
let votes       = {{}};
let notes       = {{}};
let submitted   = new Set();
let annotator   = '';

// ── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {{
    loadFromStorage();
    updateOverallProgress();
    // Restore annotator
    const saved = localStorage.getItem('mcq_annotator') || '';
    const input = document.getElementById('annotator-input');
    if (input) input.value = annotator || saved;
    input && input.addEventListener('change', () => {{
        annotator = input.value.trim();
        localStorage.setItem('mcq_annotator', annotator);
    }});
    // Restore notes from storage
    const savedNotes = JSON.parse(localStorage.getItem('mcq_notes_v3') || '{{}}');
    for (const [qid, note] of Object.entries(savedNotes)) {{
        const el = document.getElementById('notes-' + qid);
        if (el) el.value = note;
    }}
}});

function loadFromStorage() {{
    try {{
        const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}');
        votes    = saved.votes    || {{}};
        submitted = new Set(saved.submitted || []);
        // Restore button states
        for (const [qid, crits] of Object.entries(votes)) {{
            for (const [criterion, value] of Object.entries(crits)) {{
                restoreButton(qid, criterion, value);
            }}
        }}
        for (const qid of submitted) markSubmitted(qid);
    }} catch (e) {{}}
}}

function saveToStorage() {{
    localStorage.setItem(STORAGE_KEY, JSON.stringify({{
        votes:    votes,
        submitted: Array.from(submitted),
    }}));
}}

// ── Vote ─────────────────────────────────────────────────────────────────────
function vote(btn, qid, criterion, value) {{
    const card = btn.closest('.review-card');
    card.querySelectorAll('.vote-btn').forEach(b => {{
        const onclick = b.getAttribute('onclick') || '';
        if (onclick.includes(qid) && onclick.includes(criterion)) {{
            b.classList.toggle('selected',
                (value && b.classList.contains('pass')) ||
                (!value && b.classList.contains('fail')));
        }}
    }});
    btn.classList.add('selected');

    if (!votes[qid]) votes[qid] = {{}};
    votes[qid][criterion] = value;

    const verdictEl = document.getElementById('verdict-' + qid + '-' + criterion);
    if (verdictEl) {{
        verdictEl.textContent = value ? '✅ PASS' : '❌ FAIL';
        verdictEl.className = 'verdict-display ' + (value ? 'verdict-pass' : 'verdict-fail');
    }}

    const progEl = document.getElementById('progress-' + qid);
    const total = 7;
    const voted = Object.keys(votes[qid] || {{}}).length;
    if (progEl) progEl.textContent = voted + '/' + total;

    saveToStorage();
}}

function restoreButton(qid, criterion, value) {{
    const card = document.getElementById('card-' + qid);
    if (!card) return;
    card.querySelectorAll('.vote-btn').forEach(btn => {{
        const onclick = btn.getAttribute('onclick') || '';
        if (onclick.includes(qid) && onclick.includes(criterion)) {{
            if ((value && btn.classList.contains('pass')) ||
                (!value && btn.classList.contains('fail'))) {{
                btn.classList.add('selected');
            }}
        }}
    }});
    const verdictEl = document.getElementById('verdict-' + qid + '-' + criterion);
    if (verdictEl) {{
        verdictEl.textContent = value ? '✅ PASS' : '❌ FAIL';
        verdictEl.className = 'verdict-display ' + (value ? 'verdict-pass' : 'verdict-fail');
    }}
}}

// ── Submit ────────────────────────────────────────────────────────────────────
function submitVerdict(qid) {{
    const card    = document.getElementById('card-' + qid);
    const statusEl = document.getElementById('status-' + qid);
    const total = 7;
    const voted = Object.keys(votes[qid] || {{}}).length;

    if (voted < total) {{
        statusEl.textContent = 'Cần vote đủ 7 tiêu chí (hiện: ' + voted + '/' + total + ')';
        statusEl.className = 'submit-status in-progress';
        return;
    }}
    submitted.add(qid);
    card.classList.add('submitted');
    statusEl.textContent = '✅ Submitted';
    statusEl.className = 'submit-status submitted';
    saveToStorage();
    updateOverallProgress();
}}

function markSubmitted(qid) {{
    const card     = document.getElementById('card-' + qid);
    const statusEl = document.getElementById('status-' + qid);
    if (card) card.classList.add('submitted');
    if (statusEl) {{
        statusEl.textContent = '✅ Submitted';
        statusEl.className = 'submit-status submitted';
    }}
}}

function updateOverallProgress() {{
    const total = document.querySelectorAll('.review-card').length;
    const done  = submitted.size;
    const el    = document.getElementById('progress-overall');
    if (el) el.textContent = done + '/' + total + ' submitted';
}}

// ── Export JSON ───────────────────────────────────────────────────────────────
function exportJSON() {{
    const ann  = annotator || document.getElementById('annotator-input').value.trim() || 'anonymous';
    const now  = new Date().toISOString();
    // Collect notes
    const verdictEntries = {{}};
    for (const qid of Object.keys(votes)) {{
        verdictEntries[qid] = votes[qid];
        const noteEl = document.getElementById('notes-' + qid);
        if (noteEl && noteEl.value.trim()) verdictEntries[qid]['_notes'] = noteEl.value.trim();
    }}
    const output = {{
        annotator:        ann,
        timestamp:        now,
        total_annotated: Object.keys(votes).length,
        verdicts:         verdictEntries,
    }};
    const blob  = new Blob([JSON.stringify(output, null, 2)], {{type: 'application/json'}});
    const url   = URL.createObjectURL(blob);
    const a     = document.createElement('a');
    a.href      = url;
    a.download  = 'eval_review_annotations.json';
    a.click();
    URL.revokeObjectURL(url);
    alert('✅ Đã xuất ' + Object.keys(votes).length + ' câu → eval_review_annotations.json');
}}

// ── Bulk actions ─────────────────────────────────────────────────────────────
function markAllSubmitted() {{
    document.querySelectorAll('.review-card').forEach(card => {{
        const qid = card.id.replace('card-', '');
        if (!votes[qid] || Object.keys(votes[qid]).length < 7) return;
        if (!submitted.has(qid)) {{
            submitted.add(qid);
            card.classList.add('submitted');
            const statusEl = document.getElementById('status-' + qid);
            if (statusEl) {{
                statusEl.textContent = '✅ Submitted';
                statusEl.className = 'submit-status submitted';
            }}
        }}
    }});
    saveToStorage();
    updateOverallProgress();
}};

function markAllPass() {{
    const CRITERIA = [
        'format_pass','language_pass','grammar_pass','relevance_pass',
        'answerability_pass','correct_set_pass','overall_valid'
    ];
    document.querySelectorAll('.review-card').forEach(card => {{
        const qid = card.id.replace('card-', '');
        CRITERIA.forEach(c => {{
            if (!votes[qid]) votes[qid] = {{}};
            if (!(c in votes[qid])) votes[qid][c] = true;
            const btn = card.querySelector('.vote-btn.pass[onclick*="' + qid + '"][onclick*="' + c + '"]');
            if (btn) btn.classList.add('selected');
            const verdictEl = document.getElementById('verdict-' + qid + '-' + c);
            if (verdictEl) {{
                verdictEl.textContent = '✅ PASS';
                verdictEl.className = 'verdict-display verdict-pass';
            }}
        }});
        const progEl = document.getElementById('progress-' + qid);
        if (progEl) progEl.textContent = '7/7';
    }});
    saveToStorage();
}};

function clearAll() {{
    if (!confirm('Xóa tất cả votes? Hành động này không thể hoàn tác.')) return;
    votes    = {{}};
    submitted = new Set();
    document.querySelectorAll('.vote-btn').forEach(b => b.classList.remove('selected'));
    document.querySelectorAll('.verdict-display').forEach(el => {{
        el.textContent = '—';
        el.className   = 'verdict-display verdict-pending';
    }});
    document.querySelectorAll('.review-card').forEach(c => c.classList.remove('submitted'));
    document.querySelectorAll('.submit-status').forEach(el => {{
        el.textContent = '';
        el.className   = 'submit-status';
    }});
    document.querySelectorAll('[id^="progress-"]').forEach(el => {{
        const qid = el.id.replace('progress-', '');
        if (qid) el.textContent = '0/7';
    }});
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem('mcq_notes_v3');
    updateOverallProgress();
}};

// ── Tabs ─────────────────────────────────────────────────────────────────────
function showTab(name) {{
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    const btn = event && event.target || null;
    if (btn) btn.classList.add('active');
    document.querySelectorAll('.review-card').forEach(card => {{
        card.style.display = (name === 'all' || card.getAttribute('data-chapter') === name)
            ? '' : 'none';
    }});
}};
</script>
</body>
</html>"""


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate HTML human review interface for MCQ evaluation"
    )
    parser.add_argument(
        "--exp", default=Config.EXP_NAME,
        help="Experiment name (e.g. exp_04). Default: $EXP_NAME or Config.EXP_NAME"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output HTML path (default: output/{EXP}/review/eval_review.html)"
    )
    args = parser.parse_args()

    exp_name = args.exp or Config.EXP_NAME
    if not exp_name:
        print("❌ --exp or EXP_NAME environment variable is required.")
        sys.exit(1)

    # Override Config paths for this experiment
    import os as _os
    _os.environ["EXP_NAME"] = exp_name
    Config.EXP_NAME = exp_name
    Config.OUTPUT_DIR = Config.PROJECT_ROOT / "output" / Config.EXP_NAME
    Config.EVAL_OUTPUT = Config.OUTPUT_DIR / "07_eval"
    Config.EVAL_IWF_OUTPUT = Config.OUTPUT_DIR / "08_eval_iwf"

    print(f"📂 Experiment: {exp_name}")
    print(f"   Questions: {Config.EVAL_IWF_OUTPUT / 'final_accepted_questions.jsonl'}")
    print(f"   Evaluated: {Config.EVAL_OUTPUT / 'evaluated_questions.jsonl'}")

    questions = load_gen_mcqs()
    llm_evals  = load_evaluated()

    # Build IWF lookup from accepted questions (they carry distractor_evaluation)
    llm_iwf_map: dict[str, dict] = {
        q.get("question_id", f"q{i}"): q
        for i, q in enumerate(questions)
    }

    missing = sum(1 for q in questions if q.get("question_id", "") not in llm_evals)
    print(f"   Loaded {len(questions)} questions, {len(llm_evals)} with LLM eval "
          f"({missing} missing)")

    html = build_html(questions, llm_evals, llm_iwf_map, exp_name)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = (
            Config.PROJECT_ROOT
            / "output" / exp_name
            / "review"
            / "eval_review.html"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"\n✅ Saved: {out_path}")
    print(f"   Open in browser → vote → Submit → Export JSON")
    print(f"   Then run:")
    print(f"   python src/eval/eval_metrics.py --exp \"{exp_name}\" "
          f"--human-json output/{exp_name}/review/eval_review_annotations.json")


if __name__ == "__main__":
    main()
