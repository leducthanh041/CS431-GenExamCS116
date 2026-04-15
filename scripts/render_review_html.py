"""
render_review_html.py — Human Review Interface for MCQ Evaluation
================================================================
Exports final_accepted_questions.jsonl → interactive HTML with:
  - 6 checklist criteria per question (✅ Pass / ❌ Fail buttons)
  - "Show LLM Judge Verdict" to compare with Gemma's evaluation
  - localStorage persistence + JSON export

Usage:
  python scripts/render_review_html.py [--output <path>]

Output:
  output/exp_01_baseline/review/eval_review.html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import Config, load_jsonl


def load_gen_mcqs() -> list[dict]:
    path = Config.EVAL_IWF_OUTPUT / "final_accepted_questions.jsonl"
    if not path.exists():
        path = Config.GEN_COT_OUTPUT / "all_final_mcqs.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No questions found at {path}")
    return load_jsonl(path)


def load_evaluated() -> list[dict]:
    path = Config.EVAL_OUTPUT / "evaluated_questions.jsonl"
    if path.exists():
        return {q.get("question_id", f"q{i}"): q
                for i, q in enumerate(load_jsonl(path))}
    return {}


def difficulty_badge(d: str) -> str:
    colors = {"G1": "#27ae60", "G2": "#f39c12", "G3": "#e74c3c"}
    return f'<span style="background:{colors.get(d,"#7f8c8d")};color:#fff;padding:2px 10px;border-radius:12px;font-size:0.75em;font-weight:700">{d}</span>'


def eval_criterion_row(criterion: str, label: str, llm_val: bool | None, qid: str) -> str:
    """
    Build one review row for a criterion:
      - Shows criterion name
      - Hidden LLM verdict (shown on click)
      - Pass/Fail buttons (highlighted if already selected)
    """
    llm_icon  = "✅" if llm_val else "❌" if llm_val is not None else "—"
    llm_class = "llm-pass" if llm_val else "llm-fail" if llm_val is not None else "llm-unknown"
    return f"""
<tr data-criterion="{criterion}" class="criterion-row">
  <td style="padding:6px 10px;width:35%">
    <strong>{label}</strong>
    <div style="font-size:0.75em;color:#888;margin-top:2px">{criterion}</div>
  </td>
  <td style="padding:6px 10px;text-align:center">
    <span class="llm-badge {llm_class}" id="llm-{qid}-{criterion}">{llm_icon}</span>
    <span class="llm-label" style="font-size:0.75em;color:#555">(LLM)</span>
  </td>
  <td style="padding:6px 10px;text-align:center">
    <button class="vote-btn pass" onclick="vote(this, '{qid}', '{criterion}', true)"
            title="Pass">✅ Pass</button>
    <button class="vote-btn fail" onclick="vote(this, '{qid}', '{criterion}', false)"
            title="Fail">❌ Fail</button>
  </td>
  <td style="padding:6px 10px;width:25%;text-align:center">
    <span class="verdict-display" id="verdict-{qid}-{criterion}">—</span>
  </td>
</tr>"""


def build_review_card(q: dict, idx: int, llm_eval: dict | None, chapter: str = "") -> str:
    qid    = q.get("question_id", f"q{idx+1}")
    qtext  = q.get("question_text", "")
    qtype  = q.get("question_type", "single")
    opts   = q.get("options", {})
    correct = q.get("correct_answers", [])
    topic   = q.get("topic", q.get("_meta", {}).get("topic_name", ""))
    diff    = q.get("difficulty_label", q.get("_meta", {}).get("difficulty", "G2"))
    eval_   = llm_eval.get("evaluation", {}) if llm_eval else {}
    ch      = chapter or "OTHER"

    qtype_label = (
        '[Nhiều đáp án đúng]' if qtype == 'multiple_correct'
        else '[Một đáp án đúng]'
    )

    # ── Options ──────────────────────────────────────────────────
    options_html = ""
    for letter in ["A", "B", "C", "D"]:
        text = opts.get(letter, "")
        is_correct = letter in correct
        bg = "#d4edda" if is_correct else "#f8f9fa"
        border = "#27ae60" if is_correct else "#dee2e6"
        marker = " ✅" if is_correct else ""
        options_html += f"""
        <div style="background:{bg};border-left:4px solid {border};padding:7px 12px;margin:3px 0;border-radius:4px">
            <strong>{letter}.</strong> {text}{marker}
        </div>"""

    # ── Criterion rows ─────────────────────────────────────────────
    criteria = [
        ("format_pass",        "1. Format"),
        ("language_pass",      "2. Ngôn ngữ"),
        ("grammar_pass",       "3. Ngữ pháp"),
        ("relevance_pass",     "4. Relevance"),
        ("answerability_pass", "5. Trả lời được"),
        ("correct_set_pass",   "6. Đáp án đúng"),
    ]

    criterion_rows = ""
    for c, label in criteria:
        criterion_rows += eval_criterion_row(c, label, eval_.get(c), qid)

    # ── Overall ──────────────────────────────────────────────────
    llm_overall = eval_.get("overall_valid")
    overall_icon = "✅" if llm_overall else "❌" if llm_overall is not None else "—"
    llm_class = "llm-pass" if llm_overall else "llm-fail" if llm_overall is not None else "llm-unknown"

    quality = eval_.get("quality_score", "?")
    fail_reasons = eval_.get("fail_reasons", [])

    return f"""
<div class="review-card" id="card-{qid}" data-chapter="{ch}" style="background:#fff;border:2px solid #ddd;border-radius:10px;padding:18px;margin-bottom:20px;box-shadow:0 2px 6px rgba(0,0,0,.06)">
    <!-- Header -->
    <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;margin-bottom:12px">
        <div>
            <span style="font-size:0.75em;color:#888">#{idx+1}</span>
            <strong style="font-size:1em;color:#222;margin-left:6px">{qid}</strong>
        </div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
            {difficulty_badge(diff)}
            <span style="background:#2980b9;color:#fff;padding:2px 8px;border-radius:8px;font-size:0.75em">{topic}</span>
        </div>
    </div>

    <!-- Question text -->
    <p style="font-size:1em;font-weight:600;line-height:1.5;margin:0 0 12px;color:#2c3e50">
        {qtext}
    </p>

    <!-- Options (reveal on click) -->
    <details>
        <summary style="cursor:pointer;font-size:0.85em;color:#2980b9;font-weight:600;margin-bottom:8px">
            👁 Show Options + Correct Answers
        </summary>
        <div style="margin-bottom:10px">{options_html}</div>
    </details>

    <!-- Review Table -->
    <table style="width:100%;border-collapse:collapse;margin-top:12px">
        <tr style="background:#34495e;color:#fff;font-size:0.8em">
            <th style="padding:6px 10px;text-align:left">Tiêu chí</th>
            <th style="padding:6px 10px;text-align:center">LLM Judge</th>
            <th style="padding:6px 10px;text-align:center">Human Vote</th>
            <th style="padding:6px 10px;text-align:center">Verdict</th>
        </tr>
        {criterion_rows}
        <!-- Overall -->
        <tr style="background:#f0f0f0;font-weight:600">
          <td style="padding:8px 10px">✅ Overall Valid</td>
          <td style="padding:8px 10px;text-align:center">
            <span class="llm-badge {llm_class}">{overall_icon}</span>
          </td>
          <td style="padding:8px 10px;text-align:center">
            <button class="vote-btn pass" onclick="vote(this, '{qid}', 'overall_valid', true)">✅ Pass</button>
            <button class="vote-btn fail" onclick="vote(this, '{qid}', 'overall_valid', false)">❌ Fail</button>
          </td>
          <td style="padding:8px 10px;text-align:center">
            <span class="verdict-display" id="verdict-{qid}-overall_valid">—</span>
          </td>
        </tr>
    </table>

    <!-- Quality score + fail reasons -->
    <div style="margin-top:10px;font-size:0.8em;color:#555">
        <strong>LLM Quality Score:</strong> {quality}
        &nbsp;|&nbsp;
        <strong>Fail reasons:</strong> {', '.join(fail_reasons) or '—'}
    </div>

    <!-- Notes textarea -->
    <div style="margin-top:10px">
        <label style="font-size:0.8em;color:#555">
            <strong>Notes (optional):</strong>
        </label><br>
        <textarea
            id="notes-{qid}"
            placeholder="Ghi chú của người review..."
            rows="2"
            style="width:100%;padding:6px;border-radius:6px;border:1px solid #ccc;font-size:0.85em;resize:vertical"
            onchange="saveNotes()"
        ></textarea>
    </div>

    <!-- Submit per question -->
    <div style="margin-top:10px;display:flex;gap:8px;align-items:center">
        <span class="submit-status" id="status-{qid}" style="font-size:0.8em;color:#888"></span>
        <button onclick="submitVerdict('{qid}')"
                style="background:#2980b9;color:#fff;border:none;padding:6px 16px;border-radius:6px;cursor:pointer;font-size:0.85em">
            Submit this question
        </button>
        <span style="font-size:0.75em;color:#aaa">Progress: <span id="progress-{qid}">0/7</span>/7</span>
    </div>
</div>"""


def build_html(questions: list[dict], llm_evals: dict) -> str:
    # Build chapter lookup: question_id → chapter
    chapter_of: dict[str, str] = {}
    for q in questions:
        meta = q.get("_meta", {})
        topic_id = meta.get("topic_id", "unknown")
        ch = topic_id.split("_")[0].upper() if "_" in topic_id else "OTHER"
        chapter_of[q.get("question_id", "")] = ch

    # Build all cards once (no duplication — only the "All" tab)
    all_cards = "\n".join(
        build_review_card(q, i, llm_evals.get(q.get("question_id", f"q{i}")),
                           chapter_of.get(q.get("question_id", ""), "OTHER"))
        for i, q in enumerate(questions)
    )

    # Grouped for tab buttons
    grouped: dict[str, list] = {}
    for q in questions:
        meta = q.get("_meta", {})
        topic_id = meta.get("topic_id", "unknown")
        ch = topic_id.split("_")[0].upper() if "_" in topic_id else "OTHER"
        grouped.setdefault(ch, []).append(q)

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Human Review — CS431MCQGen ({len(questions)} câu)</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    max-width: 900px;
    margin: 0 auto;
    padding: 20px 16px;
    background: #f0f2f5;
    color: #333;
  }}
  h1 {{ text-align: center; color: #2c3e50; margin-bottom: 4px; }}
  .instructions {{
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 20px;
    font-size: 0.9em;
    line-height: 1.6;
  }}
  .instructions strong {{ color: #856404; }}
  .tab-bar {{
    display: flex;
    gap: 4px;
    margin-bottom: 16px;
    border-bottom: 2px solid #ddd;
  }}
  .tab-btn {{
    padding: 8px 16px;
    border: none;
    background: #e0e0e0;
    cursor: pointer;
    border-radius: 6px 6px 0 0;
    font-size: 0.85em;
  }}
  .tab-btn.active {{ background: #2980b9; color: #fff; font-weight: 600; }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}

  /* Vote buttons */
  .vote-btn {{
    padding: 4px 10px;
    border-radius: 6px;
    border: 2px solid transparent;
    cursor: pointer;
    font-size: 0.82em;
    margin: 0 2px;
    transition: all 0.15s;
  }}
  .vote-btn.pass {{
    background: #e8f5e9;
    border-color: #a5d6a7;
    color: #2e7d32;
  }}
  .vote-btn.pass.selected {{
    background: #4caf50;
    border-color: #388e3c;
    color: #fff;
  }}
  .vote-btn.fail {{
    background: #ffebee;
    border-color: #ef9a9a;
    color: #c62828;
  }}
  .vote-btn.fail.selected {{
    background: #f44336;
    border-color: #d32f2f;
    color: #fff;
  }}

  /* LLM badge */
  .llm-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 0.9em;
  }}
  .llm-pass  {{ background: #d4edda; color: #155724; }}
  .llm-fail  {{ background: #f8d7da; color: #721c24; }}
  .llm-unknown {{ background: #e9ecef; color: #6c757d; }}

  /* Verdict display */
  .verdict-display {{
    font-weight: 700;
    font-size: 0.9em;
  }}
  .verdict-pass {{ color: #4caf50; }}
  .verdict-fail {{ color: #f44336; }}
  .verdict-pending {{ color: #888; }}

  /* Criterion rows */
  .criterion-row:hover {{ background: #f8f9fa; }}
  .criterion-row.submitted {{ opacity: 0.6; }}

  /* Submit status */
  .submit-status.submitted {{ color: #4caf50; }}
  .submit-status.in-progress {{ color: #ff9800; }}

  /* Bottom toolbar */
  #toolbar {{
    position: sticky;
    bottom: 0;
    background: #2c3e50;
    color: #fff;
    padding: 10px 16px;
    border-radius: 10px 10px 0 0;
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
    z-index: 100;
    box-shadow: 0 -2px 10px rgba(0,0,0,.2);
  }}
  #toolbar button {{
    padding: 8px 16px;
    border-radius: 6px;
    border: none;
    cursor: pointer;
    font-size: 0.85em;
    font-weight: 600;
  }}
  #toolbar .btn-primary {{ background: #2980b9; color: #fff; }}
  #toolbar .btn-success {{ background: #27ae60; color: #fff; }}
  #toolbar .btn-warning {{ background: #f39c12; color: #fff; }}
  #toolbar .btn-danger  {{ background: #e74c3c; color: #fff; }}
  #progress-overall {{
    background: #fff;
    color: #2c3e50;
    padding: 4px 12px;
    border-radius: 12px;
    font-weight: 700;
  }}
  #annotator-input {{
    padding: 6px 10px;
    border-radius: 6px;
    border: none;
    font-size: 0.85em;
    width: 140px;
  }}
</style>
</head>
<body>

<h1>📝 Human Review Interface — CS431MCQGen</h1>

<div class="instructions">
  <strong>Hướng dẫn:</strong> Mỗi câu hỏi có 6 tiêu chí + 1 overall. Với mỗi tiêu chí, nhấn
  <strong>✅ Pass</strong> (đạt) hoặc <strong>❌ Fail</strong> (không đạt).
  Nhấn <strong>"Show LLM Judge"</strong> để xem đánh giá của Gemma-3-12b-it.
  Khi đã vote đủ 7 tiêu chí cho 1 câu, nhấn <strong>"Submit this question"</strong>.
  Cuối cùng nhấn <strong>"Export JSON"</strong> để tải file annotations.
</div>

<!-- Tabs -->
<div class="tab-bar">
  <button class="tab-btn active" onclick="showTab('all')">Tất cả</button>
  {''.join(f'<button class="tab-btn" onclick="showTab(\'{ch}\')">{ch}</button>' for ch in sorted(grouped.keys()))}
</div>

<!-- All questions in one container, filtered by data-chapter -->
<div id="tab-all" class="tab-content active">
  {all_cards}
</div>

<!-- Sticky toolbar -->
<div id="toolbar">
  <span style="font-size:0.8em">Annotator:</span>
  <input id="annotator-input" type="text" placeholder="Tên của bạn" value="">
  <span id="progress-overall">0/{len(questions)} submitted</span>
  <button class="btn-success" onclick="exportJSON()">📥 Export JSON</button>
  <button class="btn-warning" onclick="markAllReviewed()">Mark all as reviewed (auto-pass)</button>
  <button class="btn-danger"  onclick="clearAll()">🗑️ Clear all</button>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────
const STORAGE_KEY = 'mcq_review_votes';
let votes = {{}};
let submittedQuestions = new Set();
let annotator = '';

// ── Init ────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {{
    loadFromStorage();
    updateOverallProgress();
    document.getElementById('annotator-input').value = annotator || localStorage.getItem('mcq_annotator') || '';
    document.getElementById('annotator-input').addEventListener('change', () => {{
        annotator = document.getElementById('annotator-input').value.trim();
        localStorage.setItem('mcq_annotator', annotator);
    }});
}});

function loadFromStorage() {{
    try {{
        const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}');
        votes = saved.votes || {{}};
        submittedQuestions = new Set(saved.submitted || []);
        // Restore button states
        for (const [qid, crits] of Object.entries(votes)) {{
            for (const [criterion, value] of Object.entries(crits)) {{
                restoreButton(qid, criterion, value);
            }}
        }}
        // Update submit statuses
        for (const qid of submittedQuestions) {{
            markSubmitted(qid);
        }}
    }} catch (e) {{}}
}}

function saveToStorage() {{
    localStorage.setItem(STORAGE_KEY, JSON.stringify({{
        votes,
        submitted: Array.from(submittedQuestions),
    }}));
}}

// ── Vote ────────────────────────────────────────────────────────────────
function vote(btn, qid, criterion, value) {{
    const card = btn.closest('.review-card');

    // Clear sibling buttons for this criterion in this card
    card.querySelectorAll('.vote-btn').forEach(b => {{
        const onclick = b.getAttribute('onclick') || '';
        if (onclick.includes(qid) && onclick.includes(criterion)) {{
            if ((value && b.classList.contains('pass')) ||
                (!value && b.classList.contains('fail'))) {{
                b.classList.add('selected');
            }} else {{
                b.classList.remove('selected');
            }}
        }}
    }});
    btn.classList.add('selected');

    // Store vote
    if (!votes[qid]) votes[qid] = {{}};
    votes[qid][criterion] = value;

    // Update verdict display
    const verdictEl = document.getElementById('verdict-' + qid + '-' + criterion);
    if (verdictEl) {{
        verdictEl.textContent = value ? '✅ PASS' : '❌ FAIL';
        verdictEl.className = 'verdict-display ' + (value ? 'verdict-pass' : 'verdict-fail');
    }}

    // Update progress counter for this question
    const total = 7;  // 6 criteria + overall
    const voted = Object.keys(votes[qid] || {{}}).length;
    const progEl = document.getElementById('progress-' + qid);
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

// ── Submit individual question ────────────────────────────────────────
function submitVerdict(qid) {{
    const card = document.getElementById('card-' + qid);
    const statusEl = document.getElementById('status-' + qid);
    const total = 7;
    const voted = Object.keys(votes[qid] || {{}}).length;

    if (voted < total) {{
        statusEl.textContent = 'Cần vote đủ 7 tiêu chí (hiện: ' + voted + '/' + total + ')';
        statusEl.className = 'submit-status in-progress';
        return;
    }}
    submittedQuestions.add(qid);
    card.classList.add('submitted');
    statusEl.textContent = '✅ Submitted';
    statusEl.className = 'submit-status submitted';
    saveToStorage();
    updateOverallProgress();
}}

function markSubmitted(qid) {{
    const card = document.getElementById('card-' + qid);
    const statusEl = document.getElementById('status-' + qid);
    if (card) card.classList.add('submitted');
    if (statusEl) {{ statusEl.textContent = '✅ Submitted'; statusEl.className = 'submit-status submitted'; }}
}}

// ── Progress ───────────────────────────────────────────────────────────
function updateOverallProgress() {{
    const total = document.querySelectorAll('.review-card').length;
    const done = submittedQuestions.size;
    const el = document.getElementById('progress-overall');
    if (el) el.textContent = done + '/' + total + ' submitted';
}}

// ── Export JSON ────────────────────────────────────────────────────────
function exportJSON() {{
    const ann = annotator || document.getElementById('annotator-input').value.trim() || 'anonymous';
    const verdictEntries = {{}};
    for (const [qid, crits] of Object.entries(votes)) {{
        verdictEntries[qid] = crits;
    }}
    const output = {{
        annotator: ann,
        timestamp: new Date().toISOString(),
        total_annotated: Object.keys(votes).length,
        verdicts: verdictEntries,
    }};
    const blob = new Blob([JSON.stringify(output, null, 2)], {{type: 'application/json'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'eval_review_annotations.json';
    a.click();
    URL.revokeObjectURL(url);
}}

// ── Bulk actions ─────────────────────────────────────────────────────
function markAllReviewed() {{
    document.querySelectorAll('.review-card').forEach(card => {{
        const qid = card.id.replace('card-', '');
        // Auto-vote Pass for all missing criteria
        ['format_pass','language_pass','grammar_pass','relevance_pass',
         'answerability_pass','correct_set_pass','overall_valid'].forEach(c => {{
            if (!votes[qid] || !(c in votes[qid])) {{
                if (!votes[qid]) votes[qid] = {{}};
                votes[qid][c] = true;
                const btn = card.querySelector('.vote-btn.pass[onclick*="' + qid + '"][onclick*="' + c + '"]');
                if (btn) {{ btn.classList.add('selected'); }}
                const verdictEl = document.getElementById('verdict-' + qid + '-' + c);
                if (verdictEl) {{
                    verdictEl.textContent = '✅ PASS';
                    verdictEl.className = 'verdict-display verdict-pass';
                }}
            }}
        }});
        if (!submittedQuestions.has(qid)) {{
            submittedQuestions.add(qid);
            card.classList.add('submitted');
            const statusEl = document.getElementById('status-' + qid);
            if (statusEl) {{ statusEl.textContent = '✅ Submitted'; statusEl.className = 'submit-status submitted'; }}
        }}
    }});
    saveToStorage();
    updateOverallProgress();
}}

function clearAll() {{
    if (!confirm('Clear all votes? This cannot be undone.')) return;
    votes = {{}};
    submittedQuestions = new Set();
    document.querySelectorAll('.vote-btn').forEach(b => b.classList.remove('selected'));
    document.querySelectorAll('.verdict-display').forEach(el => {{
        el.textContent = '—';
        el.className = 'verdict-display verdict-pending';
    }});
    document.querySelectorAll('.review-card').forEach(c => c.classList.remove('submitted'));
    document.querySelectorAll('.submit-status').forEach(el => {{
        el.textContent = '';
        el.className = 'submit-status';
    }});
    document.querySelectorAll('[id^="progress-"]').forEach(el => {{
        const qid = el.id.replace('progress-', '');
        if (qid) el.textContent = '0/7';
    }});
    localStorage.removeItem(STORAGE_KEY);
    updateOverallProgress();
}}

// ── Tabs ──────────────────────────────────────────────────────────────
function showTab(name) {{
    // Update button active states
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    const clickedBtn = event ? event.target : document.querySelector('.tab-btn[onclick*="showTab"]');
    if (clickedBtn) clickedBtn.classList.add('active');

    // Filter cards: show all, or only cards with matching data-chapter
    document.querySelectorAll('.review-card').forEach(card => {{
        card.style.display = (name === 'all' || card.getAttribute('data-chapter') === name) ? '' : 'none';
    }});
}}

function saveNotes() {{ /* Notes saved inline per textarea */ }}
</script>
</body>
</html>"""


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML path (default: review/eval_review.html)")
    args = parser.parse_args()

    print("Loading questions...")
    questions = load_gen_mcqs()
    llm_evals = load_evaluated()
    print(f"Loaded {len(questions)} questions, {len(llm_evals)} with LLM evaluations")

    html = build_html(questions, llm_evals)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Config.PROJECT_ROOT / "output" / Config.EXP_NAME / "review" / "eval_review.html"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"✅ Saved: {out_path}")
    print("   Open in browser → review questions → Export JSON → compute Cohen's κ")


if __name__ == "__main__":
    main()
