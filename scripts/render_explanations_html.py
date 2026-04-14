#!/usr/bin/env python3
"""
render_explanations_html.py — Render explanations.jsonl → browsable HTML
=====================================================================
Usage:
    python ./scripts/render_explanations_html.py --exp exp_03_test_15q
    python ./scripts/render_explanations_html.py --input /datastore/uittogether/LuuTru/Thanhld/CS431MCQGen/output/exp_03_test_15q/09_explain/explanations.jsonl --output out.html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# ── Path setup ───────────────────────────────────────────────────────────────
import sys as _sys
from pathlib import Path as _Path
_pdir = str(_Path(__file__).resolve().parent.parent)
if _pdir not in _sys.path:
    _sys.path.insert(0, _pdir)

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))
from common import Config, load_jsonl


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_ts(seconds: int) -> str:
    if seconds >= 3600:
        hh = seconds // 3600
        mm = (seconds % 3600) // 60
        ss = seconds % 60
        return f"{hh}:{mm:02d}:{ss:02d}"
    else:
        mm = seconds // 60
        ss = seconds % 60
        return f"{mm}:{ss:02d}"


def _ts_to_seconds(ts) -> int:
    """Parse '1:39' or '1:39:45' → total seconds."""
    if not ts:
        return 0
    ts_str = str(ts).strip()
    parts = ts_str.split(":")
    try:
        if len(parts) == 3:          # HH:MM:SS
            return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
        elif len(parts) == 2:        # MM:SS
            return int(parts[0])*60 + int(parts[1])
        else:                        # plain seconds
            return int(float(ts_str))
    except (ValueError, TypeError):
        return 0


# ── Render sources ───────────────────────────────────────────────────────────

def render_sources(sources: list[dict]) -> str:
    """Render citation list as HTML."""
    if not sources:
        return '<p class="no-sources">Không có trích dẫn</p>'

    html_parts = ['<div class="sources">']
    for s in sources:
        stype = s.get("type", "")
        if stype == "video":
            url = s.get("url", "")
            ts  = s.get("timestamp", "") or s.get("youtube_ts_start", "") or ""
            ts_sec = _ts_to_seconds(ts)

            if ts_sec > 0:
                # Normalize youtu.be → youtube.com/watch?v=...&t=NNN
                if "youtu.be/" in url:
                    vid = url.split("youtu.be/")[1].split("?")[0].split("&")[0]
                    yt_full = f"https://www.youtube.com/watch?v={vid}&t={ts_sec}"
                elif "youtube.com/watch" in url:
                    base = url.split("&t=")[0].split("?t=")[0].split("?")[0]
                    yt_full = f"{base}&t={ts_sec}"
                else:
                    yt_full = f"{url}&t={ts_sec}"
            else:
                yt_full = url

            desc = s.get("description", "")
            html_parts.append(
                f'<div class="source source-video">'
                f'<span class="source-icon">▶️</span>'
                f'<span class="source-desc">{desc}</span>'
                f'<a href="{yt_full}" target="_blank" class="source-link">'
                f'{yt_full[:90]}{"..." if len(yt_full)>90 else ""}</a>'
                f'</div>'
            )
        elif stype == "slide":
            file = s.get("file", "")
            page = s.get("page", "")
            desc = s.get("description", "")
            html_parts.append(
                f'<div class="source source-slide">'
                f'<span class="source-icon">📄</span>'
                f'<span class="source-desc">{desc or file}</span>'
                f'</div>'
            )
        elif stype == "web":
            url = s.get("url", "")
            title = s.get("title", s.get("description", ""))
            html_parts.append(
                f'<div class="source source-web">'
                f'<span class="source-icon">🌐</span>'
                f'<a href="{url}" target="_blank" class="source-link">{title or url}</a>'
                f'</div>'
            )
    html_parts.append('</div>')
    return "\n".join(html_parts)


def render_distractor_explanations(de: dict) -> str:
    """Render distractor explanations as HTML."""
    if not de:
        return '<p class="no-distractor-exp">Chưa có giải thích cho các đáp án sai</p>'

    parts = ['<div class="distractor-list">']
    for letter, text in de.items():
        if not text:
            continue
        parts.append(
            f'<div class="distractor-item">'
            f'<span class="distractor-letter">{letter}:</span>'
            f'<span class="distractor-text">{text}</span>'
            f'</div>'
        )
    parts.append('</div>')
    return "\n".join(parts)


# ── Render single question ────────────────────────────────────────────────────

def render_question(exp_item: dict) -> str:
    """Render a single explained question as HTML."""
    qid        = exp_item.get("question_id", "?")
    qtype      = exp_item.get("question_type", "single_correct")
    diff       = exp_item.get("difficulty_label", "G2")
    topic      = exp_item.get("topic", exp_item.get("_meta", {}).get("topic_id", "?"))
    meta       = exp_item.get("_meta", {})
    chapter    = meta.get("chapter_id", "")

    qtext      = exp_item.get("question_text", "")
    options    = exp_item.get("options", {})
    correct    = exp_item.get("correct_answers", [])

    exp_data   = exp_item.get("explanation", {})
    rationale  = exp_data.get("correct_answer_rationale", "")
    de         = exp_data.get("distractor_explanations", {})
    kctx       = exp_data.get("knowledge_context", {})
    sources    = exp_data.get("sources", exp_data.get("sources_used", []))

    # Build options HTML
    option_letters = ["A", "B", "C", "D"]
    options_html_parts = []
    for letter in option_letters:
        text = options.get(letter, "")
        if not text:
            continue
        is_correct = letter in correct
        badge = '<span class="opt-correct-badge">✓</span>' if is_correct else ""
        cls = "option correct" if is_correct else "option"
        options_html_parts.append(
            f'<div class="{cls}">'
            f'<span class="opt-letter">{letter}.</span>'
            f'<span class="opt-text">{text}</span>'
            f'{badge}'
            f'</div>'
        )

    # Difficulty badge color
    diff_colors = {"G1": "#2ecc71", "G2": "#f39c12", "G3": "#e74c3c"}
    diff_color  = diff_colors.get(diff, "#888")
    type_badge  = "1 đáp án" if qtype == "single_correct" else "nhiều đáp án"

    # Knowledge context
    kctx_html = ""
    if kctx:
        kctx_html = (
            f'<div class="knowledge-context">'
            f'<div class="kc-section-title">📚 Kiến thức bổ sung</div>'
            f'<div class="kc-scope"><strong>Phạm vi:</strong> {kctx.get("topic_scope","")}</div>'
            f'<div class="kc-prereq"><strong>Prerequisites:</strong> '
            f'{", ".join(kctx.get("prerequisites",[]))}</div>'
            f'<div class="kc-advanced"><strong>Nâng cao:</strong> {kctx.get("advanced_knowledge","")}</div>'
            f'<div class="kc-value"><strong>Giá trị học tập:</strong> {kctx.get("learning_value","")}</div>'
            f'</div>'
        )

    # Question motivation
    motivation = exp_data.get("question_motivation", "")
    motivation_html = ""
    if motivation:
        motivation_html = (
            f'<div class="motivation-block">'
            f'<div class="section-title">💡 Tại sao ra câu hỏi này?</div>'
            f'<div class="motivation-text">{motivation}</div>'
            f'</div>'
        )

    # Render
    return f"""
<div class="question-card" id="{qid}">
    <div class="question-header">
        <span class="qid">{qid}</span>
        <span class="chapter-badge">{chapter}</span>
        <span class="topic-badge">{topic}</span>
        <span class="diff-badge" style="background:{diff_color}">{diff}</span>
        <span class="type-badge">{type_badge}</span>
    </div>

    <div class="question-stem">
        <p>{qtext}</p>
    </div>

    <div class="options-block">
        {"".join(options_html_parts)}
    </div>

    <div class="explanation-block">
        {motivation_html}

        <div class="section-title">📝 Giải thích đáp án đúng</div>
        <div class="rationale">
            {rationale or '<em class="no-rationale">Chưa có giải thích</em>'}
        </div>

        <div class="section-title">❌ Giải thích các đáp án sai</div>
        {render_distractor_explanations(de)}

        {kctx_html}

        <div class="section-title">🔗 Trích dẫn nguồn</div>
        {render_sources(sources)}
    </div>
</div>
"""


# ── Build full HTML ──────────────────────────────────────────────────────────

def build_html(items: list[dict], exp_name: str) -> str:
    """Build full HTML page from list of explained questions."""
    n = len(items)

    # Stats
    g1 = sum(1 for x in items if x.get("difficulty_label") == "G1")
    g2 = sum(1 for x in items if x.get("difficulty_label") == "G2")
    g3 = sum(1 for x in items if x.get("difficulty_label") == "G3")
    single = sum(1 for x in items if x.get("question_type") == "single_correct")
    multi  = sum(1 for x in items if x.get("question_type") == "multiple_correct")

    questions_html = "\n".join(render_question(item) for item in items)

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CS431MCQGen — Explanations ({exp_name})</title>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        font-family: 'Segoe UI', 'DejaVu Sans', Arial, sans-serif;
        background: #f0f2f5;
        color: #1a1a2e;
        font-size: 14px;
        line-height: 1.6;
    }}
    .page-header {{
        background: linear-gradient(135deg, #2c3e50, #3498db);
        color: white;
        padding: 24px 32px;
        position: sticky;
        top: 0;
        z-index: 100;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }}
    .page-header h1 {{
        font-size: 22px;
        margin-bottom: 8px;
    }}
    .stats-bar {{
        display: flex;
        gap: 20px;
        font-size: 13px;
        opacity: 0.95;
    }}
    .stats-bar span {{
        background: rgba(255,255,255,0.15);
        padding: 2px 10px;
        border-radius: 12px;
    }}
    .toc {{
        background: #fff;
        padding: 16px 32px;
        border-bottom: 1px solid #e0e0e0;
        max-height: 200px;
        overflow-y: auto;
    }}
    .toc-title {{ font-weight: 600; color: #2c3e50; margin-bottom: 8px; }}
    .toc-items {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .toc-item {{
        font-size: 12px;
        background: #eef2f7;
        color: #3498db;
        padding: 2px 8px;
        border-radius: 4px;
        cursor: pointer;
        text-decoration: none;
    }}
    .toc-item:hover {{ background: #3498db; color: white; }}
    .main-content {{
        max-width: 860px;
        margin: 24px auto;
        padding: 0 16px;
    }}
    .question-card {{
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin-bottom: 28px;
        overflow: hidden;
    }}
    .question-header {{
        background: #f8f9fa;
        padding: 12px 20px;
        border-bottom: 1px solid #e9ecef;
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }}
    .qid {{ font-weight: 700; color: #3498db; font-size: 13px; }}
    .chapter-badge, .topic-badge {{
        font-size: 11px;
        background: #eef2f7;
        color: #555;
        padding: 2px 8px;
        border-radius: 4px;
    }}
    .diff-badge {{
        font-size: 11px;
        font-weight: 700;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
    }}
    .type-badge {{
        font-size: 11px;
        background: #e8f5e9;
        color: #388e3c;
        padding: 2px 8px;
        border-radius: 4px;
    }}
    .question-stem {{
        padding: 18px 20px;
        border-bottom: 1px solid #f0f0f0;
    }}
    .question-stem p {{
        font-size: 15px;
        font-weight: 500;
        color: #1a1a2e;
    }}
    .options-block {{
        padding: 12px 20px;
        border-bottom: 1px solid #f0f0f0;
    }}
    .option {{
        padding: 7px 12px;
        margin: 4px 0;
        border-radius: 6px;
        border: 1px solid #e0e0e0;
        background: #fafafa;
        display: flex;
        align-items: flex-start;
        gap: 8px;
    }}
    .option.correct {{
        border-color: #27ae60;
        background: #eafaf1;
    }}
    .opt-letter {{ font-weight: 700; color: #333; min-width: 20px; }}
    .opt-correct-badge {{
        margin-left: auto;
        color: #27ae60;
        font-weight: 900;
        font-size: 16px;
    }}
    .explanation-block {{
        padding: 16px 20px;
        background: #fffdf0;
        border-top: 1px solid #f5f0d0;
    }}
    .section-title {{
        font-weight: 700;
        font-size: 13px;
        color: #2c3e50;
        margin: 12px 0 6px;
        border-left: 4px solid #3498db;
        padding-left: 8px;
    }}
    .section-title:first-child {{ margin-top: 0; }}
    .rationale {{
        background: #fff;
        border: 1px solid #e8f4f8;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 13px;
        line-height: 1.7;
        color: #333;
    }}
    .distractor-list {{ margin-top: 4px; }}
    .distractor-item {{
        padding: 6px 10px;
        margin: 3px 0;
        background: #fff;
        border: 1px solid #fde8e8;
        border-radius: 5px;
        font-size: 13px;
        display: flex;
        gap: 8px;
    }}
    .distractor-letter {{ font-weight: 700; color: #e74c3c; min-width: 20px; }}
    .distractor-text {{ color: #555; line-height: 1.6; }}
    .knowledge-context {{
        background: #f0f7ff;
        border: 1px solid #d0e8ff;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 4px;
        font-size: 13px;
    }}
    .kc-section-title {{ font-weight: 700; color: #1565c0; margin-bottom: 6px; }}
    .kc-scope, .kc-prereq, .kc-advanced, .kc-value {{
        margin: 3px 0;
        color: #333;
    }}
    .sources {{ margin-top: 4px; }}
    .source {{
        padding: 6px 10px;
        margin: 3px 0;
        border-radius: 5px;
        font-size: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .source-video {{ background: #fff3e0; border: 1px solid #ffe0b2; }}
    .source-slide {{ background: #e3f2fd; border: 1px solid #bbdefb; }}
    .source-web {{ background: #e8f5e9; border: 1px solid #c8e6c9; }}
    .source-icon {{ font-size: 14px; }}
    .source-desc {{ flex: 1; color: #333; }}
    .motivation-block {{
        background: #fff8e1;
        border: 1px solid #ffe082;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 12px;
    }}
    .motivation-text {{
        font-size: 13px;
        color: #333;
        line-height: 1.7;
    }}
    .source-link {{
        color: #1565c0;
        text-decoration: none;
        font-size: 11px;
        word-break: break-all;
    }}
    .source-link:hover {{ text-decoration: underline; }}
    .no-sources, .no-rationale, .no-distractor-exp {{
        color: #999;
        font-style: italic;
        font-size: 12px;
    }}
    .filter-bar {{
        background: white;
        padding: 12px 32px;
        border-bottom: 1px solid #e0e0e0;
        position: sticky;
        top: 72px;
        z-index: 90;
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        align-items: center;
    }}
    .filter-label {{ font-size: 13px; font-weight: 600; color: #555; }}
    .filter-btn {{
        font-size: 12px;
        padding: 4px 12px;
        border-radius: 16px;
        border: 1px solid #ccc;
        background: white;
        cursor: pointer;
        transition: all 0.2s;
    }}
    .filter-btn.active {{
        background: #3498db;
        color: white;
        border-color: #3498db;
    }}
    .filter-btn:hover {{ border-color: #3498db; }}
    @media (max-width: 600px) {{
        .page-header, .main-content, .toc, .filter-bar {{ padding-left: 12px; padding-right: 12px; }}
        .stats-bar {{ gap: 8px; font-size: 12px; }}
    }}
</style>
<script>
function filterByType(type) {{
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    if (type === 'all') {{
        document.querySelectorAll('.question-card').forEach(c => c.style.display = '');
        document.querySelectorAll('.filter-btn').forEach(b => {{
            if (b.dataset.type === 'all') b.classList.add('active');
        }});
    }} else {{
        document.querySelectorAll('.question-card').forEach(c => {{
            c.style.display = c.dataset.type === type ? '' : 'none';
        }});
        document.querySelectorAll('.filter-btn').forEach(b => {{
            if (b.dataset.type === type) b.classList.add('active');
        }});
    }}
}}

function filterByDiff(diff) {{
    document.querySelectorAll('.question-card').forEach(c => {{
        c.style.display = c.dataset.diff === diff ? '' : 'none';
    }});
}}

function showAll() {{
    document.querySelectorAll('.question-card').forEach(c => c.style.display = '');
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    document.querySelector('.filter-btn[data-type="all"]').classList.add('active');
}}

function scrollToQid(qid) {{
    document.getElementById(qid)?.scrollIntoView({{behavior:'smooth', block:'start'}});
}}

// Auto-number options
document.addEventListener('DOMContentLoaded', () => {{
    document.querySelectorAll('.question-card').forEach((card, i) => {{
        card.dataset.type = card.querySelector('.type-badge')?.textContent.includes('nhiều') ? 'multiple_correct' : 'single_correct';
        card.dataset.diff = card.querySelector('.diff-badge')?.textContent || 'G2';
    }});
}});
</script>
</head>
<body>

<div class="page-header">
    <h1>📝 CS431MCQGen — Giải thích câu hỏi</h1>
    <div class="stats-bar">
        <span>📊 {n} câu hỏi</span>
        <span>✅ G1: {g1}  |  🟡 G2: {g2}  |  ❌ G3: {g3}</span>
        <span>1 đáp án: {single}  |  Nhiều đáp án: {multi}</span>
        <span>Experiment: <b>{exp_name}</b></span>
    </div>
</div>

<div class="toc">
    <div class="toc-title">📑 Mục lục</div>
    <div class="toc-items">
        {''.join(f'<a class="toc-item" onclick="scrollToQid(\'{x.get("question_id","")}\')">{x.get("question_id","")}</a>' for x in items)}
    </div>
</div>

<div class="filter-bar">
    <span class="filter-label">Lọc theo loại:</span>
    <button class="filter-btn active" data-type="all" onclick="showAll()">Tất cả</button>
    <button class="filter-btn" data-type="single_correct" onclick="filterByType('single_correct')">1 đáp án</button>
    <button class="filter-btn" data-type="multiple_correct" onclick="filterByType('multiple_correct')">Nhiều đáp án</button>
    <span class="filter-label" style="margin-left:16px">Lọc theo độ khó:</span>
    <button class="filter-btn" onclick="filterByDiff('G1')">G1</button>
    <button class="filter-btn" onclick="filterByDiff('G2')">G2</button>
    <button class="filter-btn" onclick="filterByDiff('G3')">G3</button>
    <button class="filter-btn" onclick="showAll()">Hiện tất cả</button>
</div>

<div class="main-content">
{questions_html}
</div>

</body>
</html>
"""


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Render explanations.jsonl → HTML")
    parser.add_argument("--exp", default=None,
                        help="Experiment name (auto-detects path)")
    parser.add_argument("--input", "-i", default=None,
                        help="Path to explanations.jsonl (overrides --exp)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output HTML path")
    args = parser.parse_args()

    # Resolve input file
    if args.input:
        input_path = Path(args.input)
    elif args.exp:
        input_path = (Config.PROJECT_ROOT / "output" / args.exp
                      / "09_explain" / "explanations.jsonl")
    else:
        print("❌ Must specify --exp or --input")
        sys.exit(1)

    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        sys.exit(1)

    # Resolve output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / "explanations.html"

    # Load data
    items = load_jsonl(input_path)
    print(f"📋 Loaded {len(items)} explained questions from {input_path}")
    print(f"📝 Rendering → {output_path}")

    # Build HTML
    exp_name = args.exp or input_path.parent.parent.name
    html = build_html(items, exp_name)

    # Write
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ HTML saved: {output_path}")


if __name__ == "__main__":
    main()
