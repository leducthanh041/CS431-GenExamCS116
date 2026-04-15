"""
prompt_parser.py — Parse user free-text prompt → generation config
===================================================================
Dùng cho web deployment (Streamlit): user nhập câu prompt tự do,
hệ thống parse để xác định:
  1. Chapters/topics cần tập trung
  2. Số câu hỏi (target range)
  3. Các yêu cầu đặc biệt khác

Cách parse:
  - Dùng LLM (GPT-4o) để parse free-text → structured config
  - Fallback: keyword matching đơn giản nếu không có API key

 Ví dụ prompts:
  "Tôi muốn ôn tập chương 7b và chương 8 về classification và CNN"
  → focus_chapters: ["ch07b", "ch08"]
  → topic_weights: {"ch07b": 2.0, "ch08": 2.0}

  "Cho tôi 50 câu hỏi tập trung vào ensemble models và hyperparameter tuning"
  → target_range: [40, 55]
  → focus_chapters: ["ch09", "ch10"]
  → topic_weights: {"ch09": 1.5, "ch10": 2.0}

  "Tôi cần 20 câu hỏi G3 (khó) về deep learning"
  → target_range: [18, 25]
  → focus_chapters: ["ch08"]
  → difficulty_override: "G3"
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# NOTE: Must add parent dir (src/) to path so 'from common' works
import sys as _sys
from pathlib import Path as _Path
_pdir = str(_Path(__file__).resolve().parent.parent)  # .../CS431MCQGen/src/
if _pdir not in _sys.path:
    _sys.path.insert(0, _pdir)

from common import Config  # for Config.PROJECT_ROOT reference if needed

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Known chapter / topic names (for keyword matching) ─────────────────────

CHAPTER_ALIASES: dict[str, list[str]] = {
    "ch02": ["ch02", "popular libraries", "thư viện phổ biến", "numpy", "pandas", "matplotlib"],
    "ch03": ["ch03", "pipeline", "eda", "exploratory data"],
    "ch04": ["ch04", "tiền xử lý", "data preprocessing", "missing data", "outlier", "feature"],
    "ch05": ["ch05", "đánh giá mô hình", "eval model", "metrics", "classification metrics", "regression metrics"],
    "ch06": ["ch06", "unsupervised", "clustering", "dimensionality reduction", "pca"],
    "ch07a": ["ch07a", "regression", "linear regression", "regularization", "ridge", "lasso"],
    "ch07b": ["ch07b", "classification", "logistic regression", "decision tree", "svm", "supervised"],
    "ch08": ["ch08", "deep learning", "cnn", "neural network", "neural networks"],
    "ch09": ["ch09", "parameter tuning", "hyperparameter", "grid search", "random search", "bayesian"],
    "ch10": ["ch10", "ensemble", "bagging", "boosting", "random forest", "xgboost"],
    "ch11": ["ch11", "deployment", "model serving", "api", "monitoring"],
}

DIFFICULTY_KEYWORDS: dict[str, list[str]] = {
    "G1": ["g1", "dễ", "nhớ", "nhận biết", "remember", "easy", "basic", "cơ bản"],
    "G2": ["g2", "vừa", "trung bình", "áp dụng", "apply", "medium", "intermediate", "tính toán"],
    "G3": ["g3", "khó", "nâng cao", "đánh giá", "evaluate", "create", "difficult", "hard"],
}

QUESTION_COUNT_PATTERNS = [
    r"(\d+)\s*(?:-\s*(\d+))?\s*câu",
    r"(\d+)\s*(?:-\s*(\d+))?\s*question",
    r"gen(?:erate)?\s*(\d+)",
    r"tạo\s*(\d+)",
    r"sinh\s*(\d+)",
]


# ── Keyword-based fallback parser ─────────────────────────────────────────────

def parse_by_keywords(user_prompt: str) -> dict[str, Any]:
    """
    Parse user prompt using keyword matching (no LLM needed).
    Fast fallback when no API key is available.
    """
    prompt_lower = user_prompt.lower()

    # ── Detect chapters ──────────────────────────────────────────────────────
    found_chapters: list[str] = []
    for ch_id, aliases in CHAPTER_ALIASES.items():
        for alias in aliases:
            if alias in prompt_lower:
                if ch_id not in found_chapters:
                    found_chapters.append(ch_id)
                break

    # ── Detect difficulty ────────────────────────────────────────────────────
    difficulty_override = None
    for diff, keywords in DIFFICULTY_KEYWORDS.items():
        for kw in keywords:
            if kw in prompt_lower:
                difficulty_override = diff
                break
        if difficulty_override:
            break

    # ── Detect question count ────────────────────────────────────────────────
    target_min = 25
    target_max = 35
    for pattern in QUESTION_COUNT_PATTERNS:
        m = re.search(pattern, prompt_lower)
        if m:
            first = int(m.group(1))
            if m.lastindex and m.group(2):
                second = int(m.group(2))
                target_min = min(first, second)
                target_max = max(first, second)
            else:
                # Single number: use as upper bound, generate ~20% more
                target_max = max(first, 30)
                target_min = max(1, int(first * 0.7))
            break

    # ── Detect question type ───────────────────────────────────────────────
    has_multi = any(kw in prompt_lower for kw in [
        "nhiều đáp án", "multiple correct", "nhiều đáp án đúng", "multi-answer"
    ])
    single_ratio = 0.6 if has_multi else 0.8

    return {
        "focus_chapters": found_chapters,
        "focus_topics": [],
        "target_range": [target_min, target_max],
        "difficulty_override": difficulty_override,
        "single_correct_ratio": single_ratio,
        "topic_weights": {ch: 2.0 for ch in found_chapters} if found_chapters else {},
        "parse_method": "keyword",
    }


# ── LLM-based parser (for web deployment) ────────────────────────────────────

LLM_PARSE_PROMPT = """Bạn là một trợ lý parse prompt người dùng thành cấu hình sinh câu hỏi trắc nghiệm.

[NHIỆM VỤ]
Parse câu prompt của người dùng dưới đây và trả về cấu hình JSON.

[PARSING RULES]

1. **focus_chapters**: Các chapters cần tập trung (từ danh sách: ch02, ch03, ch04, ch05, ch06, ch07a, ch07b, ch08, ch09, ch10, ch11)
   - Map tên tiếng Việt/tiếng Anh về chapter ID:
     * "ch07b" / "classification" / "logistic regression" / "decision tree" / "svm" → ch07b
     * "ch08" / "deep learning" / "cnn" / "neural network" → ch08
     * "ch04" / "feature" / "missing data" / "tiền xử lý" → ch04
     * "ch10" / "ensemble" / "bagging" / "boosting" / "random forest" → ch10
     * "ch09" / "hyperparameter" / "grid search" / "bayesian" → ch09
     * "ch05" / "metrics" / "đánh giá" → ch05
     * "ch06" / "clustering" / "unsupervised" / "pca" → ch06
     * "ch07a" / "regression" / "linear regression" / "regularization" → ch07a
     * "ch02" / "popular libraries" / "numpy" / "pandas" → ch02
     * "ch03" / "pipeline" / "eda" → ch03
     * "ch11" / "deployment" / "api" / "monitoring" → ch11

2. **target_range**: Số câu hỏi [min, max]
   - "20 câu" → [18, 25]
   - "30-50 câu" → [30, 50]
   - Mặc định: [25, 35] nếu không nói rõ

3. **difficulty_override**: Mức khó (G1=dễ, G2=vừa, G3=khó)
   - "khó" / "nâng cao" / "G3" → G3
   - "vừa" / "G2" → G2
   - "dễ" / "G1" → G1
   - Không nói → null

4. **topic_weights**: Trọng số per chapter (1.0=default, 2.0=gấp đôi)
   - Chapters được nhắc đến → weight = 2.0
   - Others → 1.0

5. **single_correct_ratio**: Tỉ lệ single answer
   - Mặc định 0.8 (80% single, 20% multiple)
   - Nếu prompt nói "nhiều đáp án" → giảm xuống 0.6

[PROMPT CẦN PARSE]
{user_prompt}

[OUTPUT FORMAT — JSON ONLY]
{{
  "focus_chapters": ["ch07b", "ch08"],
  "focus_topics": [],
  "target_range": [25, 35],
  "difficulty_override": null,
  "single_correct_ratio": 0.8,
  "topic_weights": {{"ch07b": 2.0, "ch08": 2.0}},
  "parse_method": "llm"
}}
"""


def parse_with_llm(
    user_prompt: str,
    api_key: str | None = None,
) -> dict[str, Any] | None:
    """
    Parse using GPT-4o (requires API key).
    Returns None if parsing fails or no API key.
    """
    import os
    api_key = api_key or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a JSON parser. Only output valid JSON."},
                {"role": "user", "content": LLM_PARSE_PROMPT.format(user_prompt=user_prompt)},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        raw = response.choices[0].message.content.strip()

        # Extract JSON
        for match in re.finditer(r'\{', raw):
            try:
                parsed = json.loads(raw[match.start():])
                # Validate required fields
                if "focus_chapters" in parsed:
                    parsed["parse_method"] = "llm"
                    return parsed
            except json.JSONDecodeError:
                continue

        return None
    except Exception:
        return None


def parse_user_prompt(
    user_prompt: str,
    use_llm: bool = True,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Main entry point: parse user free-text prompt → generation config.

    Args:
        user_prompt: Free-text prompt from user (e.g. "Tôi muốn ôn chương 7b và ch08")
        use_llm: Try LLM parse first, fallback to keyword matching
        api_key: OpenAI API key (reads OPENAI_API_KEY env var by default)

    Returns:
        Dict with: focus_chapters, focus_topics, target_range,
                   difficulty_override, single_correct_ratio, topic_weights, parse_method
    """
    if use_llm:
        llm_result = parse_with_llm(user_prompt, api_key)
        if llm_result:
            print(f"  ✅ Parsed with LLM: {llm_result.get('focus_chapters', [])}")
            return llm_result

    # Fallback: keyword matching
    result = parse_by_keywords(user_prompt)
    print(f"  ✅ Parsed with keywords: {result['focus_chapters']} | method=keyword")
    return result


def merge_with_base_config(
    parsed: dict[str, Any],
    base_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Merge parsed prompt into base generation config.
    Returns a complete generation config dict ready for pipeline.
    """
    if base_config is None:
        try:
            from gen.prompt_config import load_generation_config
            base_config = load_generation_config()
        except Exception:
            base_config = {}

    gen = base_config.get("generation", {}).copy()

    # Override with parsed values
    if parsed.get("focus_chapters"):
        gen["focus_chapters"] = parsed["focus_chapters"]
    if parsed.get("focus_topics"):
        gen["focus_topics"] = parsed["focus_topics"]
    if parsed.get("target_range"):
        gen["target_range"] = parsed["target_range"]
    if parsed.get("single_correct_ratio"):
        gen["single_correct_ratio"] = parsed["single_correct_ratio"]
    if parsed.get("topic_weights"):
        # Merge with existing weights
        existing = gen.get("topic_weights", {})
        for ch, w in parsed["topic_weights"].items():
            existing[ch] = w
        gen["topic_weights"] = existing

    result = dict(base_config)
    result["generation"] = gen
    result["_parsed_prompt"] = parsed
    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parse user prompt into generation config")
    parser.add_argument("--prompt", "-p", required=True, help="User prompt to parse")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM, use keyword only")
    args = parser.parse_args()

    parsed = parse_user_prompt(args.prompt, use_llm=not args.no_llm)
    print("\n=== Parsed Config ===")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))

    merged = merge_with_base_config(parsed)
    print("\n=== Merged Generation Config ===")
    print(json.dumps(merged.get("generation", {}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
