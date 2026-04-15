"""
eval_metrics.py — Quantitative Evaluation Metrics for MCQ Generation
=====================================================================
Implements 3 core metrics + human judgment comparison.

Core metrics (automatic, no reference set required):
  1. Topic Coverage                (vs topic_list.json)
  2. LLM Judge Pass Rate           (from 07_eval + 08_eval_iwf output)
  3. Bloom Distribution KL Div.   (vs target G1→L1/L2, G2→L3+L4, G3→L5+L6)

Human Judgment module:
  4. Human vs LLM Agreement       (per-question JSON annotation → agreement metrics)

Supported human annotation file formats:
  A. render_review_html.py export (verdicts{} per question_id):
     {
       "annotator": "Nguyen Van A",
       "timestamp": "2026-04-15T...",
       "total_annotated": 10,
       "verdicts": {
         "<question_id>": { "format_pass": true, "overall_valid": true, ... }
       }
     }
  B. Standalone annotation JSON (questions{} with criteria{}):
     {
       "annotator": "...",
       "date": "2026-04-15",
       "questions": {
         "<question_id>": {
           "overall_judgment": "accept",
           "criteria": { ... },
           "distractor_quality": { ... },
           "notes": "..."
         }
       }
     }

Dependencies:
  pip install numpy scipy scikit-learn
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import Config, load_jsonl

# ── Override EXP_NAME from environment (set by pipeline script) ───────────────
_exp_name = os.environ.get("EXP_NAME", "")
if _exp_name:
    Config.EXP_NAME = _exp_name
    Config.OUTPUT_DIR = Config.PROJECT_ROOT / "output" / Config.EXP_NAME
    Config.EVAL_OUTPUT = Config.OUTPUT_DIR / "07_eval"
    Config.EVAL_IWF_OUTPUT = Config.OUTPUT_DIR / "08_eval_iwf"
    Config.EXPLAIN_OUTPUT = Config.OUTPUT_DIR / "09_explain"
    print(f"[eval_metrics] EXP_NAME overridden: {Config.EXP_NAME}")


# ==============================================================================
# 1. Topic Coverage
# ==============================================================================

def compute_topic_coverage(
    gen_mcqs: list[dict],
    topic_list: list[dict],
) -> dict[str, Any]:
    """
    Compute topic coverage: what % of topics in topic_list.json
    have at least one generated MCQ.
    """
    gen_topics: set[str] = set()
    for q in gen_mcqs:
        t = (q.get("topic")
             or q.get("_meta", {}).get("topic_name", "")
             or "").strip()
        if t:
            gen_topics.add(t)

    ref_topics: set[str] = set()
    for ch in topic_list:
        for t in ch.get("topics", []):
            ref_topics.add(t["topic_name"])

    covered = gen_topics & ref_topics
    missing = ref_topics - gen_topics

    chapter_coverage: dict[str, dict[str, Any]] = {}
    for ch in topic_list:
        ch_name = ch.get("chapter_name", ch.get("chapter_id", "?"))
        ch_topics = {t["topic_name"] for t in ch.get("topics", [])}
        ch_covered = ch_topics & gen_topics
        chapter_coverage[ch_name] = {
            "covered": sorted(ch_covered),
            "missing": sorted(ch_topics - gen_topics),
            "ratio": len(ch_covered) / len(ch_topics) if ch_topics else 0.0,
        }

    return {
        "coverage_ratio":   len(covered) / len(ref_topics) if ref_topics else 0.0,
        "topics_covered":   sorted(covered),
        "topics_missing":   sorted(missing),
        "num_covered":      len(covered),
        "num_total":        len(ref_topics),
        "extra_topics":     sorted(gen_topics - ref_topics),
        "chapter_coverage": chapter_coverage,
    }


# ==============================================================================
# 2. LLM Judge Pass Rate
# ==============================================================================

def compute_judge_pass_rate(
    evaluated_file: Path,
    iwf_file: Path,
) -> dict[str, Any]:
    """
    Aggregate pass/reject rates from pipeline evaluation outputs:
      - 07_eval: evaluated_questions.jsonl  (8-criteria Gemma-3-12b evaluation)
      - 08_eval_iwf: final_accepted_questions.jsonl  (IWF distractor filter)

    Returns per-criterion pass rates, overall rates, and quality score stats.
    """
    if not evaluated_file.exists():
        return {"error": f"File not found: {evaluated_file}"}
    if not iwf_file.exists():
        return {"error": f"File not found: {iwf_file}"}

    evaluated    = load_jsonl(evaluated_file)
    iwf_accepted = load_jsonl(iwf_file)

    total_evaluated = len(evaluated)
    total_accepted  = len(iwf_accepted)
    total_rejected   = total_evaluated - total_accepted
    final_pass_rate  = total_accepted / total_evaluated if total_evaluated else 0.0

    # ── Per-criterion pass rate (8 criteria) ─────────────────────────────────
    all_criteria = [
        "format_pass", "language_pass", "grammar_pass",
        "relevance_pass", "answerability_pass", "correct_set_pass",
        "no_four_correct_pass", "answer_not_in_stem_pass",
    ]
    criterion_rates: dict[str, float] = {}
    for c in all_criteria:
        passed = sum(
            1 for q in evaluated
            if q.get("evaluation", {}).get(c, False)
        )
        criterion_rates[c] = passed / total_evaluated if total_evaluated else 0.0

    # ── IWF pass rate ────────────────────────────────────────────────────────
    iwf_passed = sum(
        1 for q in iwf_accepted
        if q.get("distractor_evaluation", {}).get("overall_distractor_quality_pass", False)
    )
    iwf_total = len(iwf_accepted)
    iwf_pass_rate = iwf_passed / iwf_total if iwf_total else 0.0

    # ── IWF per-type pass rate ─────────────────────────────────────────────
    iwf_types = [
        "plausible_distractor", "vague_terms", "grammar_clue",
        "absolute_terms", "distractor_length", "k_type_combination",
    ]
    iwf_type_rates: dict[str, float] = {}
    for iwf_type in iwf_types:
        passed = sum(
            1 for q in iwf_accepted
            if q.get("distractor_evaluation", {}).get(iwf_type, True) is True
        )
        iwf_type_rates[iwf_type] = passed / iwf_total if iwf_total else 0.0

    # ── Quality score stats ─────────────────────────────────────────────────
    scores = [
        q.get("evaluation", {}).get("quality_score")
        for q in evaluated
        if q.get("evaluation", {}).get("quality_score") is not None
    ]
    score_stats: dict[str, Any] = {}
    if scores:
        score_stats = {
            "mean":   round(float(np.mean(scores)), 4),
            "std":    round(float(np.std(scores)),  4),
            "median": round(float(np.median(scores)), 4),
            "min":    round(float(np.min(scores)),  4),
            "max":    round(float(np.max(scores)),  4),
        }

    # ── Per-difficulty pass rate ──────────────────────────────────────────
    diff_rates: dict[str, dict[str, float]] = {}
    for diff in ["G1", "G2", "G3"]:
        diff_qs        = [q for q in evaluated
                          if q.get("difficulty_label", q.get("_meta", {}).get("difficulty", "")) == diff]
        diff_accepted = [q for q in iwf_accepted
                         if q.get("difficulty_label", q.get("_meta", {}).get("difficulty", "")) == diff]
        n_eval = len(diff_qs)
        n_acc  = len(diff_accepted)
        diff_rates[diff] = {
            "evaluated": n_eval,
            "accepted":   n_acc,
            "pass_rate":  n_acc / n_eval if n_eval else 0.0,
        }

    return {
        "final_pass_rate":       final_pass_rate,
        "total_evaluated":      total_evaluated,
        "total_accepted":        total_accepted,
        "total_rejected":        total_rejected,
        "criterion_pass_rates":  criterion_rates,
        "iwf_pass_rate":         iwf_pass_rate,
        "iwf_passed":            iwf_passed,
        "iwf_total":             iwf_total,
        "iwf_type_pass_rates":   iwf_type_rates,
        "quality_score_stats":   score_stats,
        "per_difficulty_rates":  diff_rates,
    }


# ==============================================================================
# 3. Bloom Distribution KL Divergence
# ==============================================================================

BLOOM_KEYWORDS: dict[int, list[str]] = {
    1: ["nhớ", "định nghĩa", "liệt kê", "trình bày", "nêu", "cho biết",
        "thuộc tính", "công thức", "hàm", "lệnh", " cú pháp", "là gì"],
    2: ["giải thích", "ví dụ", "so sánh", "phân biệt", "tổng hợp",
        "mô tả", "trình bày", "tại sao", "như thế nào", "hoạt động"],
    3: ["áp dụng", "sử dụng", "thực hiện", "tính toán", "viết code",
        "chạy", "kết quả", "đầu ra", "đầu vào", "giá trị"],
    4: ["phân tích", "tại sao", "lỗi", "sai", "đúng", "so sánh",
        "đánh giá", "hiệu suất", "độ phức tạp", "cách cải thiện"],
    5: ["đánh giá", "lựa chọn", "tốt nhất", "hiệu quả nhất", "nên",
        "không nên", "ưu nhược điểm", "so sánh và chọn"],
    6: ["thiết kế", "xây dựng", "cải tiến", "đề xuất", "tạo ra",
        "lập trình", "phát triển", "viết chương trình"],
}

BLOOM_TARGET: dict[int, float] = {
    1: 0.20,
    2: 0.20,
    3: 0.20,
    4: 0.20,
    5: 0.10,
    6: 0.10,
}
BLOOM_NAMES = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]


def _classify_bloom(text: str) -> int:
    """Classify question stem into Bloom level (1-6) via keyword matching."""
    text_lower = text.lower()
    scores: dict[int, float] = {i: 0 for i in range(1, 7)}
    for level, keywords in BLOOM_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[level] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 3  # fallback → L3


def compute_bloom_kl_divergence(gen_mcqs: list[dict]) -> dict[str, Any]:
    """
    Classify each question into Bloom level via keyword matching.
    Compute KL(actual ‖ target) where target encodes G1→L1+L2, G2→L3+L4, G3→L5+L6.
    """
    try:
        from scipy.stats import entropy as _entropy
    except ImportError:
        def _entropy(p, q):
            return sum(
                pi * (np.log(pi) - np.log(qi + 1e-9))
                for pi, qi in zip(p, q) if pi > 0
            )

    bloom_counts: dict[int, int] = {i: 0 for i in range(1, 7)}
    per_question: list[dict[str, Any]] = []

    for q in gen_mcqs:
        stem = q.get("question_text", "")
        diff = (
            q.get("difficulty_label")
            or q.get("_meta", {}).get("difficulty", "G2")
        )
        level = _classify_bloom(stem)
        bloom_counts[level] += 1
        per_question.append({
            "question_id": q.get("question_id", "unknown"),
            "difficulty":  diff,
            "bloom_level": level,
            "bloom_name":  BLOOM_NAMES[level - 1],
        })

    n = len(gen_mcqs) or 1
    actual = np.array([bloom_counts[i] + 1e-9 for i in range(1, 7)])
    actual = actual / actual.sum()
    target = np.array([BLOOM_TARGET[i] for i in range(1, 7)])
    kl_div = float(_entropy(actual, target))

    # Per-difficulty distribution
    diff_bloom: dict[str, dict[int, float]] = {}
    for diff in ["G1", "G2", "G3"]:
        diff_qs = [pq for pq in per_question if pq["difficulty"] == diff]
        cnt: dict[int, int] = {i: 0 for i in range(1, 7)}
        for pq in diff_qs:
            cnt[pq["bloom_level"]] += 1
        tot = len(diff_qs) or 1
        diff_bloom[diff] = {i: cnt[i] / tot for i in range(1, 7)}

    return {
        "per_question":         per_question,
        "bloom_counts":         bloom_counts,
        "actual_distribution":  actual.tolist(),
        "target_distribution": target.tolist(),
        "kl_divergence":       round(kl_div, 6),
        "per_difficulty_bloom": diff_bloom,
        "note": "Bloom levels classified via keyword matching. "
                "Target: G1→{L1,L2}, G2→{L3,L4}, G3→{L5+L6}",
    }


# ==============================================================================
# 4. Human Judgment — Agreement vs LLM Judge
# ==============================================================================
#
# Supports TWO annotation file formats:
#   A. render_review_html.py export  (verdicts{} per question_id)
#   B. Standalone annotation JSON    (questions{} with per-criterion structure)
#
# Agreement computed:
#   1. Overall accept/reject: Cohen's κ + confusion matrix + P/R/F1
#   2. Per-criterion (6–8 criteria): Cohen's κ + agreement rate
#   3. IWF overall (if available): Cohen's κ + confusion matrix
#   4. IWF per-type (if available): pass rate per IWF flaw type
# ==============================================================================

# Criteria from HTML export (render_review_html.py)
HTML_EVAL_CRITERIA = [
    "format_pass", "language_pass", "grammar_pass",
    "relevance_pass", "answerability_pass", "correct_set_pass",
    "overall_valid",
]

# Criteria for standalone annotation format
ANN_CRITERIA = [
    "format_pass", "language_pass", "grammar_pass",
    "relevance_pass", "answerability_pass", "correct_set_pass",
    "no_four_correct_pass", "answer_not_in_stem_pass",
]

IWF_TYPES = [
    "plausible_distractor", "vague_terms", "grammar_clue",
    "absolute_terms", "distractor_length", "k_type_combination",
]

_JUDGMENT_TRUE  = {True, "true", "accept", "pass"}
_JUDGMENT_FALSE = {False, "false", "reject", "fail"}


def _parse_bool(val: Any) -> bool | None:
    """Parse annotation value → Python bool. None if indeterminate."""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        low = val.lower().strip()
        if low in _JUDGMENT_TRUE:
            return True
        if low in _JUDGMENT_FALSE:
            return False
    return None


def _interpret_kappa(k: float) -> str:
    if k < 0:     return "Poor"
    if k < 0.20:  return "Slight"
    if k < 0.40:  return "Fair"
    if k < 0.60:  return "Moderate"
    if k < 0.80:  return "Substantial"
    return "Almost Perfect"


def _cm_stats(h_binary: list[int], l_binary: list[int]) -> dict[str, Any]:
    """Build confusion matrix + Cohen's κ + P/R/F1 from two binary lists."""
    n = len(h_binary)
    tp = sum(1 for h, l in zip(h_binary, l_binary) if h == 1 and l == 1)
    tn = sum(1 for h, l in zip(h_binary, l_binary) if h == 0 and l == 0)
    fp = sum(1 for h, l in zip(h_binary, l_binary) if h == 0 and l == 1)
    fn = sum(1 for h, l in zip(h_binary, l_binary) if h == 1 and l == 0)
    agree = tp + tn
    p_o = agree / n if n else 0.0

    p_e_num = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn))
    p_e_den = n * n
    p_e = p_e_num / p_e_den if p_e_den else 0.0

    accuracy   = agree / n if n else 0.0
    precision  = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Manual kappa (fallback if sklearn unavailable)
    kappa = (p_o - p_e) / (1 - p_e) if abs(1 - p_e) > 1e-9 else 0.0

    return {
        "n":              n,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "agreement_rate": round(p_o, 4),
        "p_o":            round(p_o, 4),
        "p_e":            round(p_e, 4),
        "kappa":          round(kappa, 4),
        "kappa_interp":  _interpret_kappa(kappa),
        "accuracy":      round(accuracy, 4),
        "precision":      round(precision, 4),
        "recall":         round(recall, 4),
        "f1_score":       round(f1, 4),
    }


# ── Format detection ─────────────────────────────────────────────────────────

def _detect_format(annotations: dict) -> str:
    """Detect annotation format: 'html_export', 'standalone', or 'unknown'."""
    if "verdicts" in annotations:
        return "html_export"
    if "questions" in annotations:
        return "standalone"
    return "unknown"


# ── HTML export format handlers ──────────────────────────────────────────────

def _compute_html_overall(matched: dict[str, dict]) -> dict[str, Any]:
    """
    Compute overall accept/reject agreement from HTML export verdicts.

    Human overall:
      - Explicit overall_valid vote if present
      - Otherwise majority vote across the 6 criteria
    LLM overall: overall_valid from Step 07 eval
    """
    h_overall: list[int] = []
    l_overall: list[int] = []

    for qid, m in matched.items():
        votes    = m["votes"]
        llm_eval = m["llm_eval"]

        hv = _parse_bool(votes.get("overall_valid"))
        if hv is None:
            cvs = [_parse_bool(votes.get(c)) for c in HTML_EVAL_CRITERIA[:-1]]
            cvs = [v for v in cvs if v is not None]
            hv = (sum(cvs) >= len(cvs) / 2) if cvs else None

        lv = _parse_bool(llm_eval.get("overall_valid"))

        if hv is not None and lv is not None:
            h_overall.append(int(hv))
            l_overall.append(int(lv))

    return _cm_stats(h_overall, l_overall) if h_overall else {}


def _compute_html_per_criterion(matched: dict[str, dict]) -> dict[str, Any]:
    """Per-criterion κ + agreement from HTML export."""
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        return {}

    per_criterion: dict[str, Any] = {}
    for criterion in HTML_EVAL_CRITERIA:
        h_vals: list[int] = []
        l_vals: list[int] = []
        for qid, m in matched.items():
            hv = _parse_bool(m["votes"].get(criterion))
            lv = _parse_bool(m["llm_eval"].get(criterion))
            if hv is not None and lv is not None:
                h_vals.append(int(hv))
                l_vals.append(int(lv))
        if h_vals:
            try:
                kappa = round(float(cohen_kappa_score(h_vals, l_vals)), 4)
            except Exception:
                kappa = None
            agree_rate = round(
                sum(1 for h, l in zip(h_vals, l_vals) if h == l) / len(h_vals), 4
            )
            per_criterion[criterion] = {
                "n":               len(h_vals),
                "agreement_rate": agree_rate,
                "cohens_kappa":   kappa,
                "kappa_interp":   _interpret_kappa(kappa) if kappa is not None else "N/A",
            }
    return per_criterion


def _build_html_per_question_detail(matched: dict[str, dict]) -> list[dict[str, Any]]:
    """Build per-question detail list from HTML export format."""
    rows: list[dict[str, Any]] = []
    for qid, m in matched.items():
        votes    = m["votes"]
        llm_eval = m["llm_eval"]

        hv_overall = _parse_bool(votes.get("overall_valid"))
        if hv_overall is None:
            cvs = [_parse_bool(votes.get(c)) for c in HTML_EVAL_CRITERIA[:-1]]
            cvs = [v for v in cvs if v is not None]
            hv_overall = (sum(cvs) >= len(cvs) / 2) if cvs else None

        lv_overall = _parse_bool(llm_eval.get("overall_valid"))
        overall_match = (
            (hv_overall == lv_overall)
            if (hv_overall is not None and lv_overall is not None) else None
        )

        criterion_detail: dict[str, Any] = {}
        criterion_mismatches: list[str] = []
        for c in HTML_EVAL_CRITERIA:
            hv = _parse_bool(votes.get(c))
            lv = _parse_bool(llm_eval.get(c))
            match = (hv == lv) if (hv is not None and lv is not None) else None
            criterion_detail[c] = {"human": hv, "llm": lv, "match": match}
            if match is False:
                criterion_mismatches.append(c)

        has_disagreement = (overall_match is False or bool(criterion_mismatches))

        rows.append({
            "question_id":          qid,
            "overall_match":       overall_match,
            "human_overall":       hv_overall,
            "llm_overall":         lv_overall,
            "criteria":            criterion_detail,
            "criteria_mismatches": criterion_mismatches,
            "iwf_mismatches":      [],
            "has_disagreement":    has_disagreement,
        })
    return rows


# ── Standalone annotation format handlers ───────────────────────────────────

def _compute_standalone_overall(matched: dict[str, dict]) -> dict[str, Any]:
    """Overall accept/reject from standalone annotation (overall_judgment field)."""
    h_overall: list[int] = []
    l_overall: list[int] = []

    for qid, m in matched.items():
        hv = _parse_bool(m["ann"].get("overall_judgment"))
        # LLM: IWF verdict if available, else Step07 overall_valid
        llm_iwf_v = m["llm_iwf"].get("overall_distractor_quality_pass") if m["llm_iwf"] else None
        lv = _parse_bool(llm_iwf_v)
        if lv is None:
            lv = _parse_bool(m["llm_eval"].get("overall_valid"))

        if hv is not None and lv is not None:
            h_overall.append(int(hv))
            l_overall.append(int(lv))

    return _cm_stats(h_overall, l_overall) if h_overall else {}


def _compute_standalone_per_criterion(matched: dict[str, dict]) -> dict[str, Any]:
    """Per-criterion agreement from standalone annotation."""
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        return {}

    per_criterion: dict[str, Any] = {}
    for criterion in ANN_CRITERIA:
        h_vals: list[int] = []
        l_vals: list[int] = []
        for qid, m in matched.items():
            criteria_data = m["ann"].get("criteria", {}) if isinstance(m["ann"], dict) else {}
            hv = _parse_bool(criteria_data.get(criterion))
            lv = _parse_bool(m["llm_eval"].get(criterion))
            if hv is not None and lv is not None:
                h_vals.append(int(hv))
                l_vals.append(int(lv))
        if h_vals:
            try:
                kappa = round(float(cohen_kappa_score(h_vals, l_vals)), 4)
            except Exception:
                kappa = None
            agree_rate = round(
                sum(1 for h, l in zip(h_vals, l_vals) if h == l) / len(h_vals), 4
            )
            per_criterion[criterion] = {
                "n":               len(h_vals),
                "agreement_rate": agree_rate,
                "cohens_kappa":   kappa,
                "kappa_interp":   _interpret_kappa(kappa) if kappa is not None else "N/A",
            }
    return per_criterion


def _compute_standalone_iwf(matched: dict[str, dict]) -> tuple[dict[str, Any], dict[str, Any]]:
    """IWF overall + per-type agreement from standalone annotation."""
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        return {}, {}

    # IWF overall
    h_iwf: list[int] = []
    l_iwf: list[int] = []
    for qid, m in matched.items():
        dq = m["ann"].get("distractor_quality", {}) if isinstance(m["ann"], dict) else {}
        hv = _parse_bool(dq.get("iwf_overall"))
        lv = _parse_bool(m["llm_iwf"].get("overall_distractor_quality_pass"))
        if hv is not None and lv is not None:
            h_iwf.append(int(hv))
            l_iwf.append(int(lv))
    iwf_overall_stats = _cm_stats(h_iwf, l_iwf) if h_iwf else {}

    # IWF per-type
    iwf_per_type: dict[str, Any] = {}
    for iwf_type in IWF_TYPES:
        h_vals: list[int] = []
        l_vals: list[int] = []
        for qid, m in matched.items():
            dq = m["ann"].get("distractor_quality", {}) if isinstance(m["ann"], dict) else {}
            hv = _parse_bool(dq.get(iwf_type))
            lv = _parse_bool(m["llm_iwf"].get(iwf_type))
            if hv is not None and lv is not None:
                h_vals.append(int(hv))
                l_vals.append(int(lv))
        if h_vals:
            agree = round(
                sum(1 for h, l in zip(h_vals, l_vals) if h == l) / len(h_vals), 4
            )
            tp = sum(1 for h, l in zip(h_vals, l_vals) if h == 1 and l == 1)
            iwf_per_type[iwf_type] = {
                "n":                len(h_vals),
                "agreement_rate":   agree,
                "human_pass_rate":  round(sum(h_vals) / len(h_vals), 4),
                "llm_pass_rate":    round(sum(l_vals) / len(l_vals), 4),
                "both_pass_count":  tp,
            }

    return iwf_overall_stats, iwf_per_type


def _build_standalone_per_question_detail(matched: dict[str, dict]) -> list[dict[str, Any]]:
    """Build per-question detail list from standalone annotation format."""
    rows: list[dict[str, Any]] = []
    for qid, m in matched.items():
        ann       = m["ann"]
        llm_ev    = m["llm_eval"]
        llm_iwf_q = m["llm_iwf"]

        # Overall
        hv_overall = _parse_bool(ann.get("overall_judgment"))
        lv_overall = _parse_bool(
            llm_iwf_q.get("overall_distractor_quality_pass", False)
            if llm_iwf_q else llm_ev.get("overall_valid")
        )
        overall_match = (
            (hv_overall == lv_overall)
            if (hv_overall is not None and lv_overall is not None) else None
        )

        # Per criterion
        criteria_data = ann.get("criteria", {}) if isinstance(ann, dict) else {}
        criterion_detail: dict[str, Any] = {}
        criterion_mismatches: list[str] = []
        for c in ANN_CRITERIA:
            hv = _parse_bool(criteria_data.get(c))
            lv = _parse_bool(llm_ev.get(c))
            match = (hv == lv) if (hv is not None and lv is not None) else None
            criterion_detail[c] = {"human": hv, "llm": lv, "match": match}
            if match is False:
                criterion_mismatches.append(c)

        # IWF detail
        dq_data = ann.get("distractor_quality", {}) if isinstance(ann, dict) else {}
        iwf_detail: dict[str, Any] = {}
        iwf_mismatches: list[str] = []
        for iwf_type in IWF_TYPES:
            hv = _parse_bool(dq_data.get(iwf_type))
            lv = _parse_bool(llm_iwf_q.get(iwf_type))
            match = (hv == lv) if (hv is not None and lv is not None) else None
            iwf_detail[iwf_type] = {"human": hv, "llm": lv, "match": match}
            if match is False:
                iwf_mismatches.append(iwf_type)

        has_disagreement = (
            overall_match is False
            or bool(criterion_mismatches)
            or bool(iwf_mismatches)
        )

        rows.append({
            "question_id":          qid,
            "overall_match":       overall_match,
            "human_overall":       hv_overall,
            "llm_overall":         lv_overall,
            "criteria":            criterion_detail,
            "iwf_detail":          iwf_detail,
            "criteria_mismatches": criterion_mismatches,
            "iwf_mismatches":      iwf_mismatches,
            "notes":               ann.get("notes", "") if isinstance(ann, dict) else "",
            "has_disagreement":   has_disagreement,
        })
    return rows


# ── Main human judgment entry point ──────────────────────────────────────────

def compute_human_judgment(
    human_annotation_file: Path,
    evaluated_file: Path,
    iwf_file: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """
    Compare human judgments against LLM judge decisions.
    Supports two annotation file formats:
      A. HTML export from render_review_html.py  (verdicts{} per question)
      B. Standalone annotation JSON  (questions{} with criteria{})

    Args:
        human_annotation_file: JSON annotation from human reviewer
        evaluated_file:        07_eval/evaluated_questions.jsonl
        iwf_file:              08_eval_iwf/final_accepted_questions.jsonl
        output_path:           Where to save comparison JSON
                              (default: annotation_file.with_llm_comparison.json)

    Returns dict with sections:
      - meta: annotator, date, n_annotated, n_matched, format
      - overall: Cohen's κ + CM for accept/reject overall
      - per_criterion: per-criterion κ + agreement
      - iwf_overall: Cohen's κ + CM for distractor quality
      - iwf_per_type: pass rate per IWF flaw type
      - per_question: matched questions with human + LLM verdicts
      - disagreement_analysis: list of questions with mismatches
      - disagreement_summary: count of mismatches per criterion
    """
    # ── Load inputs ───────────────────────────────────────────────────────────
    if not human_annotation_file.exists():
        return {"error": f"Annotation file not found: {human_annotation_file}"}
    if not evaluated_file.exists():
        return {"error": f"Evaluated file not found: {evaluated_file}"}
    if not iwf_file.exists():
        return {"error": f"IWF file not found: {iwf_file}"}

    with open(human_annotation_file, encoding="utf-8") as f:
        annotations = json.load(f)

    evaluated = load_jsonl(evaluated_file)
    iwf_qs    = load_jsonl(iwf_file)

    # Build LLM lookup: question_id → eval fields
    llm_eval: dict[str, dict[str, Any]] = {
        q.get("question_id", f"q{i}"): q.get("evaluation", {})
        for i, q in enumerate(evaluated)
    }
    llm_iwf: dict[str, dict[str, Any]] = {
        q.get("question_id", f"q{i}"): q.get("distractor_evaluation", {})
        for i, q in enumerate(iwf_qs)
    }

    # ── Detect format ─────────────────────────────────────────────────────────
    fmt = _detect_format(annotations)
    annotator          = annotations.get("annotator", "unknown")
    timestamp_or_date = annotations.get("timestamp", "") or annotations.get("date", "")

    if fmt == "html_export":
        verdicts = annotations.get("verdicts", {})
        matched = {
            qid: {"votes": votes, "llm_eval": llm_eval.get(qid, {}),
                  "llm_iwf": llm_iwf.get(qid, {})}
            for qid, votes in verdicts.items()
        }
        n_annotated         = len(verdicts)
        n_unmatched          = sum(1 for qid in verdicts
                                   if not llm_eval.get(qid) and not llm_iwf.get(qid))
        per_question_detail = _build_html_per_question_detail(matched)
        overall_stats       = _compute_html_overall(matched)
        per_criterion       = _compute_html_per_criterion(matched)
        iwf_overall         = {}
        iwf_per_type        = {}

    elif fmt == "standalone":
        questions = annotations.get("questions", {})
        matched = {
            qid: {"ann": ann, "llm_eval": llm_eval.get(qid, {}),
                  "llm_iwf": llm_iwf.get(qid, {})}
            for qid, ann in questions.items()
            if llm_eval.get(qid) or llm_iwf.get(qid)
        }
        n_annotated  = len(questions)
        n_unmatched  = len(questions) - len(matched)
        per_question_detail = _build_standalone_per_question_detail(matched)
        overall_stats       = _compute_standalone_overall(matched)
        per_criterion       = _compute_standalone_per_criterion(matched)
        iwf_overall, iwf_per_type = _compute_standalone_iwf(matched)

    else:
        return {
            "error": (
                f"Unknown annotation format. "
                f"Expected 'verdicts'{{}} (HTML export) or 'questions'{{}} (standalone). "
                f"Keys found: {list(annotations.keys())}"
            )
        }

    # ── Meta ────────────────────────────────────────────────────────────────
    meta: dict[str, Any] = {
        "annotator":    annotator,
        "date":         timestamp_or_date,
        "format":       fmt,
        "n_annotated":  n_annotated,
        "n_matched":    len(matched),
        "n_unmatched":  n_unmatched,
    }

    # ── Disagreement analysis ───────────────────────────────────────────────
    disagreements      = [d for d in per_question_detail if d.get("has_disagreement")]
    disagreement_summary: dict[str, int] = {}
    for d in per_question_detail:
        for cm in d.get("criteria_mismatches", []):
            disagreement_summary[cm] = disagreement_summary.get(cm, 0) + 1
        for im in d.get("iwf_mismatches", []):
            key = f"iwf_{im}"
            disagreement_summary[key] = disagreement_summary.get(key, 0) + 1

    # ── Assemble result ─────────────────────────────────────────────────────
    result: dict[str, Any] = {
        "meta":                  meta,
        "overall":               overall_stats,
        "per_criterion":        per_criterion,
        "iwf_overall":          iwf_overall,
        "iwf_per_type":        iwf_per_type,
        "per_question":          per_question_detail,
        "disagreement_analysis": disagreements,
        "disagreement_summary":  disagreement_summary,
    }

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = (
        output_path
        or human_annotation_file.with_name(
            human_annotation_file.stem + "_with_llm_comparison.json"
        )
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Human judgment saved → {out_path}")
    return result


# ==============================================================================
# CLI
# ==============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run MCQGen evaluation metrics (core + human judgment)"
    )
    parser.add_argument(
        "--exp", required=True,
        help="Experiment name (e.g. exp_03_test_15q)"
    )
    parser.add_argument(
        "--human-json",
        help=(
            "Path to human annotation JSON. "
            "Supports render_review_html.py export (verdicts{}) "
            "or standalone annotation (questions{})"
        )
    )
    parser.add_argument(
        "--output",
        help="Save JSON output to this path (default: output/{EXP}/metrics_report.json)"
    )
    args = parser.parse_args()

    exp_dir       = Config.PROJECT_ROOT / "output" / args.exp
    accepted_file = exp_dir / "08_eval_iwf" / "final_accepted_questions.jsonl"
    eval_file     = exp_dir / "07_eval" / "evaluated_questions.jsonl"
    topic_file    = Config.TOPIC_LIST_FILE

    gen_mcqs  = load_jsonl(accepted_file) if accepted_file.exists() else []
    evaluated = load_jsonl(eval_file)        if eval_file.exists()        else []
    try:
        with open(topic_file, encoding="utf-8") as f:
            topic_list = json.load(f)
    except Exception:
        topic_list = []

    print("\n" + "=" * 60)
    print(f"  MCQGen Metrics — {args.exp}")
    print("=" * 60)

    results: dict[str, Any] = {}

    # ── 1. Topic Coverage ─────────────────────────────────────────────────
    print("\n[1/3] Topic Coverage...")
    tc = compute_topic_coverage(gen_mcqs, topic_list)
    results["topic_coverage"] = tc
    print(f"  {tc['num_covered']}/{tc['num_total']} topics "
          f"({tc['coverage_ratio']*100:.1f}%)")

    # ── 2. Judge Pass Rate ─────────────────────────────────────────────────
    print("\n[2/3] Judge Pass Rate...")
    if evaluated and accepted_file.exists():
        pr = compute_judge_pass_rate(eval_file, accepted_file)
        results["judge_pass_rate"] = pr
        if "error" not in pr:
            print(f"  Accepted: {pr['total_accepted']}/{pr['total_evaluated']} "
                  f"({pr['final_pass_rate']*100:.1f}%)")
            if pr.get("quality_score_stats"):
                qs = pr["quality_score_stats"]
                print(f"  Quality score: mean={qs['mean']}  std={qs['std']}  "
                      f"median={qs['median']}")
            if pr.get("iwf_pass_rate") is not None:
                print(f"  IWF pass rate: {pr['iwf_passed']}/{pr['iwf_total']} "
                      f"({pr['iwf_pass_rate']*100:.1f}%)")
        else:
            print(f"  ⚠️  {pr['error']}")
    else:
        print("  ⚠️  Evaluation files not found — skipping judge pass rate")

    # ── 3. Bloom KL Divergence ─────────────────────────────────────────────
    print("\n[3/3] Bloom KL Divergence...")
    if gen_mcqs:
        bloom = compute_bloom_kl_divergence(gen_mcqs)
        results["bloom_kl_divergence"] = bloom
        counts = bloom["bloom_counts"]
        print(f"  KL = {bloom['kl_divergence']:.4f}")
        print(f"  L1={counts[1]} L2={counts[2]} L3={counts[3]} "
              f"L4={counts[4]} L5={counts[5]} L6={counts[6]}")
    else:
        print("  ⚠️  No accepted questions — skipping Bloom metric")

    # ── 4. Human Judgment ───────────────────────────────────────────────────
    if args.human_json:
        print(f"\n[4] Human Judgment...")
        hj = compute_human_judgment(
            human_annotation_file=Path(args.human_json),
            evaluated_file=eval_file,
            iwf_file=accepted_file,
        )
        results["human_judgment"] = hj
        if "error" not in hj:
            meta = hj.get("meta", {})
            fmt_label = {
                "html_export": "render_review_html.py export",
                "standalone":  "Standalone annotation JSON",
            }.get(meta.get("format", ""), meta.get("format", "unknown"))
            print(f"  Format: {fmt_label} | "
                  f"annotated={meta.get('n_annotated', 0)}  "
                  f"matched={meta.get('n_matched', 0)}")
            overall = hj.get("overall", {})
            if overall:
                print(f"  Overall κ={overall.get('kappa', 'N/A')} "
                      f"({overall.get('kappa_interp', '')})")
                print(f"  Agreement: {overall.get('agreement_rate', 0)*100:.1f}%  "
                      f"F1={overall.get('f1_score', 0):.4f}")
                cm = overall.get("TP", 0), overall.get("TN", 0), \
                     overall.get("FP", 0), overall.get("FN", 0)
                print(f"  CM: TP={cm[0]} TN={cm[1]} FP={cm[2]} FN={cm[3]}")
            iwf_o = hj.get("iwf_overall", {})
            if iwf_o:
                print(f"  IWF κ={iwf_o.get('kappa', 'N/A')} "
                      f"({iwf_o.get('kappa_interp', '')})")
            if hj.get("disagreement_summary"):
                print(f"  Top disagreements:")
                for k, v in sorted(hj["disagreement_summary"].items(),
                                   key=lambda x: -x[1])[:5]:
                    print(f"    {k}: {v}")
        else:
            print(f"  ⚠️  {hj['error']}")

    print("\n" + "=" * 60)

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = (
        Path(args.output) if args.output
        else exp_dir / "metrics_report.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Metrics saved → {out_path}")

    md_path = out_path.with_suffix(".md")
    md_report = _generate_markdown(results, args.exp)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"✅ Markdown report → {md_path}")


# ==============================================================================
# Markdown report generator
# ==============================================================================

def _generate_markdown(results: dict, exp_name: str) -> str:
    """Generate self-contained markdown report."""
    import datetime

    lines: list[str] = [
        f"# MCQGen Pipeline — Metrics Report\n",
        f"**Experiment:** `{exp_name}`  \n",
        f"**Generated:** {datetime.date.today().strftime('%Y-%m-%d')}  \n",
        "---\n",
    ]

    # ── 1. Topic Coverage ──────────────────────────────────────────────────
    tc = results.get("topic_coverage", {})
    cov_pct = tc.get("coverage_ratio", 0) * 100
    lines += [
        "## 1. Topic Coverage\n",
        f"| Metric | Value |\n",
        f"|--------|-------|\n",
        f"| Topics covered | {tc.get('num_covered', 0)} / {tc.get('num_total', 0)} |\n",
        f"| Coverage rate | **{cov_pct:.1f}%** |\n",
        "\n",
        "> **≥ 80%**: Tốt — phần lớn topics được đánh giá.  \n",
        "> **< 60%**: Cần xem lại `topic_list.json` và generation config.\n",
    ]
    if tc.get("topics_missing"):
        lines += [
            "\n**Topics bị thiếu:**\n",
            *(f"- {t}\n" for t in tc["topics_missing"]),
        ]

    # ── 2. Judge Pass Rate ────────────────────────────────────────────────
    pr = results.get("judge_pass_rate", {})
    if "error" not in pr and pr:
        pass_pct = pr.get("final_pass_rate", 0) * 100
        lines += [
            "\n## 2. LLM Judge Pass Rate\n",
            f"| Metric | Value |\n",
            f"|--------|-------|\n",
            f"| Evaluated | {pr.get('total_evaluated', 0)} |\n",
            f"| **Accepted** | **{pr.get('total_accepted', 0)} ({pass_pct:.1f}%)** |\n",
            f"| Rejected | {pr.get('total_rejected', 0)} |\n",
            f"| IWF pass | {pr.get('iwf_passed', 0)} / {pr.get('iwf_total', 0)} "
            f"({(pr.get('iwf_pass_rate', 0) or 0)*100:.1f}%) |\n",
            "\n",
            "> **≥ 70%**: Pipeline chất lượng tốt.  \n",
            "> **50–70%**: Trung bình — distractors yếu hoặc poor alignment.  \n",
            "> **< 50%**: Pipeline cần tuning.\n",
        ]

        qs = pr.get("quality_score_stats", {})
        if qs:
            lines += [
                f"\n**Quality Score:** mean={qs.get('mean','N/A')}  "
                f"std={qs.get('std','N/A')}  "
                f"median={qs.get('median','N/A')}  "
                f"range=[{qs.get('min','?')}, {qs.get('max','?')}]\n",
            ]

        cr = pr.get("criterion_pass_rates", {})
        if cr:
            lines += [
                "\n**Per-Criterion Pass Rates:**\n",
                f"| Criterion | Pass Rate |\n",
                f"|-----------|----------|\n",
            ]
            for c, rate in cr.items():
                label = c.replace("_pass", "").replace("_", " ").title()
                lines.append(f"| {label} | {rate*100:.1f}% |\n")

        iwf_tr = pr.get("iwf_type_pass_rates", {})
        if iwf_tr:
            lines += [
                "\n**IWF Distractor Quality Pass Rates:**\n",
                f"| IWF Type | Pass Rate |\n",
                f"|----------|----------|\n",
            ]
            iwf_labels = {
                "plausible_distractor": "Plausible Distractor",
                "vague_terms":          "Vague Terms",
                "grammar_clue":         "Grammar Clue",
                "absolute_terms":       "Absolute Terms",
                "distractor_length":     "Distractor Length",
                "k_type_combination":    "K-Type Combination",
            }
            for k, v in iwf_tr.items():
                lines.append(f"| {iwf_labels.get(k, k)} | {v*100:.1f}% |\n")

        dr = pr.get("per_difficulty_rates", {})
        if dr:
            lines += [
                "\n**Per-Difficulty Pass Rates:**\n",
                f"| Difficulty | Evaluated | Accepted | Pass Rate |\n",
                f"|------------|----------|---------|----------|\n",
            ]
            for diff, stats in dr.items():
                lines.append(
                    f"| {diff} | {stats['evaluated']} | "
                    f"{stats['accepted']} | {stats['pass_rate']*100:.1f}% |\n"
                )
    else:
        err = pr.get("error", "Unknown error")
        lines += [f"\n## 2. LLM Judge Pass Rate\n> Not available: {err}\n"]

    # ── 3. Bloom KL Divergence ────────────────────────────────────────────
    bloom = results.get("bloom_kl_divergence", {})
    if bloom:
        counts  = bloom.get("bloom_counts", {})
        target  = bloom.get("target_distribution", [])
        kl      = bloom.get("kl_divergence", 0)
        n       = sum(counts.values()) or 1

        lines += [
            "\n## 3. Bloom Taxonomy Distribution\n",
            f"| Level | Name | Count | Actual % | Target % |\n",
            f"|-------|------|-------|----------|----------|\n",
        ]
        for i in range(6):
            lvl  = i + 1
            name = BLOOM_NAMES[i]
            cnt  = counts.get(lvl, 0)
            act  = cnt / n * 100
            tgt  = (target[i] * 100) if target else 0
            lines.append(
                f"| L{lvl} | {name} | {cnt} | {act:.1f}% | {tgt:.1f}% |\n"
            )

        lines += [
            f"\n**KL Divergence:** {kl:.4f}  \n",
            "> **< 0.2**: Tốt — phân bố gần target.  \n",
            "> **0.2–0.5**: Trung bình.  \n",
            "> **> 0.5**: Cao — phân bố không cân bằng.\n",
        ]

        pdb = bloom.get("per_difficulty_bloom", {})
        if pdb:
            lines += [
                "\n**Bloom Levels by Difficulty Label:**\n",
                f"| Difficulty | L1 | L2 | L3 | L4 | L5 | L6 |\n",
                f"|------------|----|----|----|----|----|----|\n",
            ]
            for diff in ["G1", "G2", "G3"]:
                vals = pdb.get(diff, {})
                row = [f"{vals.get(i, 0)*100:.0f}%" for i in range(1, 7)]
                lines.append(f"| {diff} | {' | '.join(row)} |\n")

    # ── 4. Human Judgment ─────────────────────────────────────────────────
    hj = results.get("human_judgment", {})
    if hj and "error" not in hj:
        meta = hj.get("meta", {})
        fmt_label = {
            "html_export": "render_review_html.py export (`verdicts{}`)",
            "standalone":  "Standalone annotation JSON (`questions{}`)",
        }.get(meta.get("format", ""), meta.get("format", "unknown"))

        lines += [
            "\n## 4. Human Judgment vs LLM Judge\n",
            f"| Metric | Value |\n",
            f"|--------|-------|\n",
            f"| Annotator | {meta.get('annotator', 'N/A')} |\n",
            f"| Date | {meta.get('date', 'N/A')} |\n",
            f"| Format | {fmt_label} |\n",
            f"| Annotated | {meta.get('n_annotated', 0)} |\n",
            f"| Matched | {meta.get('n_matched', 0)} |\n",
            f"| Unmatched | {meta.get('n_unmatched', 0)} |\n",
        ]

        overall = hj.get("overall", {})
        if overall:
            lines += [
                "\n### Overall Accept/Reject Agreement\n",
                f"| Metric | Value |\n",
                f"|--------|-------|\n",
                f"| Cohen's κ | **{overall.get('kappa', 'N/A')}** "
                f"({overall.get('kappa_interp', '')}) |\n",
                f"| Agreement rate | {overall.get('agreement_rate', 0)*100:.1f}% |\n",
                f"| Accuracy | {overall.get('accuracy', 0)*100:.1f}% |\n",
                f"| Precision | {overall.get('precision', 0)*100:.1f}% |\n",
                f"| Recall | {overall.get('recall', 0)*100:.1f}% |\n",
                f"| F1 Score | **{overall.get('f1_score', 0)*100:.1f}%** |\n",
                f"| P_e (chance) | {overall.get('p_e', 'N/A')} |\n",
                "\n**Confusion Matrix:**\n",
                f"| | LLM Accept | LLM Reject |\n",
                f"|---|---|---|\n",
                f"| Human Accept | {overall.get('TP', 0)} | {overall.get('FN', 0)} |\n",
                f"| Human Reject | {overall.get('FP', 0)} | {overall.get('TN', 0)} |\n",
            ]

        iwf_o = hj.get("iwf_overall", {})
        if iwf_o:
            lines += [
                "\n### IWF Distractor Quality Agreement\n",
                f"| Metric | Value |\n",
                f"|--------|-------|\n",
                f"| Cohen's κ | **{iwf_o.get('kappa', 'N/A')}** "
                f"({iwf_o.get('kappa_interp', '')}) |\n",
                f"| Agreement rate | {iwf_o.get('agreement_rate', 0)*100:.1f}% |\n",
                f"| F1 Score | {iwf_o.get('f1_score', 0)*100:.1f}% |\n",
            ]

        pc = hj.get("per_criterion", {})
        if pc:
            display_criteria = (
                HTML_EVAL_CRITERIA
                if meta.get("format") == "html_export"
                else ANN_CRITERIA
            )
            lines += [
                "\n### Per-Criterion Agreement\n",
                f"| Criterion | N | Agreement | Cohen's κ | Interpretation |\n",
                f"|-----------|---|-----------|------------|---------------|\n",
            ]
            for c in display_criteria:
                s = pc.get(c, {})
                label = c.replace("_pass", "").replace("_", " ").title()
                lines.append(
                    f"| {label} | {s.get('n','?')} | "
                    f"{s.get('agreement_rate',0)*100:.1f}% | "
                    f"{s.get('cohens_kappa', 'N/A')} | "
                    f"{s.get('kappa_interp', 'N/A')} |\n"
                )

        iwf_pt = hj.get("iwf_per_type", {})
        if iwf_pt:
            lines += [
                "\n### IWF Per-Type Pass Rates\n",
                f"| IWF Type | N | Human Pass | LLM Pass | Agreement |\n",
                f"|----------|---|-----------|----------|------------|\n",
            ]
            iwf_labels = {
                "plausible_distractor": "Plausible",
                "vague_terms":         "Vague Terms",
                "grammar_clue":         "Grammar Clue",
                "absolute_terms":       "Absolute Terms",
                "distractor_length":    "Length",
                "k_type_combination":   "K-Type",
            }
            for k in IWF_TYPES:
                s = iwf_pt.get(k, {})
                lines.append(
                    f"| {iwf_labels.get(k, k)} | {s.get('n','?')} | "
                    f"{s.get('human_pass_rate',0)*100:.1f}% | "
                    f"{s.get('llm_pass_rate',0)*100:.1f}% | "
                    f"{s.get('agreement_rate',0)*100:.1f}% |\n"
                )

        ds = hj.get("disagreement_summary", {})
        if ds:
            lines += [
                "\n### Disagreement Summary (top mismatches)\n",
                f"| Criterion / IWF | Mismatches |\n",
                f"|-----------------|------------|\n",
            ]
            for k, v in sorted(ds.items(), key=lambda x: -x[1])[:8]:
                lines.append(f"| {k} | {v} |\n")

        disagreements = hj.get("disagreement_analysis", [])
        if disagreements:
            lines += [
                f"\n### Disagreement Detail ({len(disagreements)} questions)\n",
            ]
            for d in disagreements[:10]:
                flag = "✅ Match" if d.get("overall_match") else "❌ Mismatch"
                lines += [
                    f"- **`{d['question_id']}`**  \n",
                    f"  - Overall: {flag}  \n",
                ]
                if d.get("criteria_mismatches"):
                    lines.append(
                        f"  - Criteria mismatches: {', '.join(d['criteria_mismatches'])}  \n"
                    )
                if d.get("iwf_mismatches"):
                    lines.append(
                        f"  - IWF mismatches: {', '.join(d['iwf_mismatches'])}  \n"
                    )
                if d.get("notes"):
                    lines.append(f"  - Notes: {d['notes']}  \n")
    else:
        hj_err = (hj or {}).get("error", "Human judgment not computed.")
        lines += [
            "\n## 4. Human Judgment vs LLM Judge\n",
            f"> Not available: {hj_err}\n",
            "> Run with `--human-json path/to/annotation.json` to compute.\n",
            "\n",
            "> Supported formats:\n",
            ">   A. **HTML export** → `render_review_html.py` → `verdicts{}`  \n",
            ">   B. **Standalone JSON** → `questions{}` with per-criterion structure\n",
        ]

    # ── Summary table ─────────────────────────────────────────────────────
    def _cov_status() -> str:
        v = results.get("topic_coverage", {}).get("coverage_ratio", 0) * 100
        return "✅" if v >= 80 else "⚠️" if v >= 60 else "❌"

    def _pass_status() -> str:
        v = results.get("judge_pass_rate", {}).get("final_pass_rate", 0) * 100
        return "✅" if v >= 70 else "⚠️" if v >= 50 else "❌"

    def _bloom_status() -> str:
        v = results.get("bloom_kl_divergence", {}).get("kl_divergence", 999)
        return "✅" if v < 0.2 else "⚠️" if v < 0.5 else "❌"

    def _hj_status() -> tuple[str, str]:
        hj2 = results.get("human_judgment", {})
        if "error" in hj2 or not hj2:
            return "N/A", "⚠️"
        k = hj2.get("overall", {}).get("kappa", 0)
        return (
            f"{k:.4f}" if k else "N/A",
            "✅" if k >= 0.6 else "⚠️" if k >= 0.4 else "❌",
        )

    hj_val, hj_flag = _hj_status()

    lines += [
        "\n---\n",
        "## Summary\n",
        f"| Metric | Target | Result | Status |\n",
        f"|--------|-------|--------|--------|\n",
        f"| Topic Coverage | ≥ 80% | "
        f"{results.get('topic_coverage',{}).get('coverage_ratio',0)*100:.1f}% | {_cov_status()} |\n",
        f"| LLM Judge Pass Rate | ≥ 70% | "
        f"{results.get('judge_pass_rate',{}).get('final_pass_rate',0)*100:.1f}% | {_pass_status()} |\n",
        f"| Bloom KL Divergence | ≤ 0.2 | "
        f"{results.get('bloom_kl_divergence',{}).get('kl_divergence', 'N/A')} | {_bloom_status()} |\n",
        f"| Human vs LLM κ | ≥ 0.6 | {hj_val} | {hj_flag} |\n",
        "\n---\n",
        "*Generated by `eval_metrics.py` — MCQGen Pipeline*\n",
    ]
    return "".join(lines)


if __name__ == "__main__":
    main()
