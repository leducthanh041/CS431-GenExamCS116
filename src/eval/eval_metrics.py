"""
eval_metrics.py — Quantitative Evaluation Metrics for MCQ Generation
=====================================================================
Implements 4 evaluation metrics for the MCQ generation pipeline.

Core metrics (automatic, no reference set required):
  1. Topic Coverage                (vs topic_list.json)
  2. Answer Ratio                  (single-correct vs multiple-correct)
  3. Diversity Openings            (entropy-based diversity of opening signatures)
  4. Fleiss's Kappa                (agreement among 4 human reviewers on overall_valid)

Dependencies:
  pip install numpy
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from collections import Counter
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


DEFAULT_HUMAN_REVIEW_FILENAMES = [
    "Nguyen_review.json",
    "Phuong_review.json",
    "Thanh_review.json",
    "Thanhhn_review.json",
]

OPENING_SIGNATURE_TOKENS = 3
_TOKEN_EDGE_RE = re.compile(r"(^[^\w]+|[^\w]+$)", flags=re.UNICODE)


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
# 2. Answer Ratio
# ==============================================================================

def compute_answer_ratio(gen_mcqs: list[dict]) -> dict[str, Any]:
    """Compute the single-correct vs multiple-correct answer ratio."""
    single = sum(1 for q in gen_mcqs if q.get("question_type") == "single_correct")
    multi = sum(1 for q in gen_mcqs if q.get("question_type") == "multiple_correct")
    total = len(gen_mcqs)
    return {
        "total": total,
        "single_correct": single,
        "multiple_correct": multi,
        "single_pct": round(single / total * 100, 1) if total else 0.0,
        "multiple_pct": round(multi / total * 100, 1) if total else 0.0,
    }


# ==============================================================================
# 3. Diversity Openings
# ==============================================================================

def _normalize_opening_text(text: str) -> str:
    """Normalize Vietnamese question text for robust prefix matching."""
    text = unicodedata.normalize("NFKC", text or "").lower().strip()
    tokens: list[str] = []
    for raw_token in text.split():
        token = _TOKEN_EDGE_RE.sub("", raw_token)
        if token:
            tokens.append(token)
    return " ".join(tokens)


def _extract_opening_signature(text: str, num_tokens: int = OPENING_SIGNATURE_TOKENS) -> str:
    """Return a short normalized opening signature from the first N tokens."""
    tokens = _normalize_opening_text(text).split()
    return " ".join(tokens[:num_tokens])


def compute_diversity_openings(gen_mcqs: list[dict]) -> dict[str, Any]:
    """
    Compute opening diversity with an entropy-based score over opening signatures.

    We define an opening signature as the first few normalized tokens of the
    question stem. Diversity is then measured as the normalized Shannon entropy
    of the signature distribution:

      H = -sum_i p_i log2 p_i
      H_norm = H / log2(K)

    where p_i is the empirical frequency of opening signature i and K is the
    number of distinct signatures.

    Supporting diagnostics:
      - top opening signatures (first 3 tokens)
      - effective number of openings (2^H)
      - repetition rate of opening signatures

    This is an entropy-based evaluation metric rather than a training loss.
    It avoids a brittle manually curated list of weak openings while still
    directly targeting concentration vs spread in how questions begin.
    """
    opening_counter: Counter[str] = Counter()
    per_question: list[dict[str, Any]] = []

    for q in gen_mcqs:
        question_id = q.get("question_id", "unknown")
        question_text = q.get("question_text", "") or ""
        signature = _extract_opening_signature(question_text)

        if signature:
            opening_counter[signature] += 1

        per_question.append({
            "question_id": question_id,
            "opening_signature": signature,
        })

    total = len(gen_mcqs)
    repeated_count = sum(max(count - 1, 0) for count in opening_counter.values())

    entropy = 0.0
    normalized_entropy = 0.0
    effective_num_openings = 0.0
    if opening_counter:
        counts = np.array(list(opening_counter.values()), dtype=float)
        probs = counts / counts.sum()
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        effective_num_openings = float(2 ** entropy)
        max_entropy = float(np.log2(len(probs))) if len(probs) > 1 else 0.0
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
        elif total:
            normalized_entropy = 0.0

    top_openings = [
        {
            "opening": opening,
            "count": count,
            "pct": round(count / total * 100, 1) if total else 0.0,
        }
        for opening, count in opening_counter.most_common(10)
    ]

    return {
        "total": total,
        "opening_diversity_pct": round(normalized_entropy * 100, 1) if total else 0.0,
        "unique_opening_signatures": len(opening_counter),
        "opening_entropy": round(entropy, 4),
        "normalized_opening_entropy": round(normalized_entropy, 4),
        "effective_num_openings": round(effective_num_openings, 4),
        "repeated_opening_count": repeated_count,
        "repetition_rate_pct": round(repeated_count / total * 100, 1) if total else 0.0,
        "top_openings": top_openings,
        "per_question": per_question,
        "note": (
            "Primary score = opening_diversity_pct = normalized Shannon entropy "
            "over opening-signature frequencies. Higher scores mean opening forms "
            "are distributed more evenly instead of concentrating on a few repeated templates."
        ),
    }


# ==============================================================================
# Legacy metric retained for backward compatibility
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
# Legacy metric: Bloom Distribution KL Divergence
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
# Legacy metric: Human Judgment — Agreement vs LLM Judge
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


def resolve_human_review_files(spec: str | None = None) -> list[Path]:
    """
    Resolve the review files used for Fleiss's kappa.

    Supported inputs:
      - None: use the 4 default review files next to this script
      - comma-separated list of JSON paths
      - a directory containing *_review.json files
      - a single file path inside a review directory
    """
    review_dir = Path(__file__).resolve().parent
    default_paths = [review_dir / name for name in DEFAULT_HUMAN_REVIEW_FILENAMES]

    if not spec:
        return default_paths

    parts = [Path(part.strip()).expanduser() for part in spec.split(",") if part.strip()]
    if len(parts) > 1:
        return parts

    target = parts[0]
    if target.is_dir():
        discovered = sorted(target.glob("*_review.json"))
        return discovered or default_paths

    sibling_reviews = sorted(target.parent.glob("*_review.json"))
    if sibling_reviews:
        ordered_default = [path for path in default_paths if path in sibling_reviews]
        extras = [path for path in sibling_reviews if path not in ordered_default]
        return ordered_default + extras

    return [target]


def _extract_reviewer_votes(annotations: dict[str, Any]) -> dict[str, dict[str, bool | None]]:
    """Normalize supported review JSON formats into {question_id: {criterion: bool}}."""
    fmt = _detect_format(annotations)
    normalized: dict[str, dict[str, bool | None]] = {}

    if fmt == "html_export":
        for qid, verdicts in annotations.get("verdicts", {}).items():
            normalized[qid] = {
                criterion: _parse_bool(verdicts.get(criterion))
                for criterion in HTML_EVAL_CRITERIA
            }
        return normalized

    if fmt == "standalone":
        for qid, question in annotations.get("questions", {}).items():
            criteria = question.get("criteria", {}) if isinstance(question, dict) else {}
            row = {
                criterion: _parse_bool(criteria.get(criterion))
                for criterion in ANN_CRITERIA
            }
            row["overall_valid"] = _parse_bool(question.get("overall_judgment"))
            if row["overall_valid"] is None:
                criterion_votes = [vote for vote in row.values() if vote is not None]
                row["overall_valid"] = (
                    sum(criterion_votes) >= len(criterion_votes) / 2
                    if criterion_votes else None
                )
            normalized[qid] = row
        return normalized

    return normalized


def _compute_fleiss_kappa(vote_rows: list[list[bool]]) -> dict[str, Any]:
    """Compute Fleiss's kappa, raw agreement, and Gwet's AC1 for binary ratings."""
    if not vote_rows:
        return {"error": "No fully annotated items available for Fleiss's kappa."}

    n_items = len(vote_rows)
    n_raters = len(vote_rows[0])
    matrix = np.zeros((n_items, 2), dtype=float)
    split_patterns: Counter[str] = Counter()
    unanimous_items = 0

    for idx, votes in enumerate(vote_rows):
        true_count = sum(1 for vote in votes if vote is True)
        false_count = n_raters - true_count
        matrix[idx, 0] = false_count
        matrix[idx, 1] = true_count
        split_patterns[f"{true_count} accept / {false_count} reject"] += 1
        if true_count in {0, n_raters}:
            unanimous_items += 1

    agreement_by_item = ((matrix ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    mean_item_agreement = float(np.mean(agreement_by_item))
    category_marginals = matrix.sum(axis=0) / (n_items * n_raters)
    chance_agreement = float(np.sum(category_marginals ** 2))
    accept_rate = float(category_marginals[1])
    reject_rate = float(category_marginals[0])

    # For binary nominal ratings, a standard AC1 implementation uses the pooled
    # category proportions and defines the chance-agreement term as 2π(1-π).
    gwet_chance_agreement = float(2 * accept_rate * reject_rate)
    if abs(1 - gwet_chance_agreement) > 1e-12:
        gwet_ac1 = float(
            (mean_item_agreement - gwet_chance_agreement) / (1 - gwet_chance_agreement)
        )
    else:
        gwet_ac1 = None

    if abs(1 - chance_agreement) > 1e-12:
        kappa = float((mean_item_agreement - chance_agreement) / (1 - chance_agreement))
        if abs(kappa) < 0.05 and mean_item_agreement >= 0.90:
            kappa_interp = "Skew-sensitive / near-zero under label imbalance"
            note = (
                "Raw agreement is high, but Fleiss's kappa is near zero because "
                "almost all ratings fall into the same category; this is a known "
                "class-imbalance effect of chance-corrected agreement coefficients."
            )
        else:
            kappa_interp = _interpret_kappa(kappa)
            note = ""
    else:
        kappa = None
        if np.isclose(mean_item_agreement, 1.0):
            kappa_interp = "N/A (all reviewers gave the same label to every item)"
            note = (
                "Fleiss's kappa is undefined here because all ratings collapse into "
                "a single category, although raw agreement is perfect."
            )
        else:
            kappa_interp = "N/A"
            note = "Fleiss's kappa is undefined because chance agreement is 1.0."

    return {
        "n_items": n_items,
        "n_raters": n_raters,
        "raw_agreement": round(mean_item_agreement, 4),
        "raw_agreement_pct": round(mean_item_agreement * 100, 1),
        "fleiss_kappa": round(kappa, 4) if kappa is not None else None,
        "kappa_interp": kappa_interp,
        "mean_item_agreement": round(mean_item_agreement, 4),
        "mean_item_agreement_pct": round(mean_item_agreement * 100, 1),
        "gwet_ac1": round(gwet_ac1, 4) if gwet_ac1 is not None else None,
        "gwet_chance_agreement": round(gwet_chance_agreement, 4),
        "unanimous_items": unanimous_items,
        "non_unanimous_items": n_items - unanimous_items,
        "unanimous_pct": round(unanimous_items / n_items * 100, 1) if n_items else 0.0,
        "category_marginals": {
            "reject": round(reject_rate, 4),
            "accept": round(accept_rate, 4),
        },
        "chance_agreement": round(chance_agreement, 4),
        "split_patterns": dict(split_patterns),
        "note": note,
    }


def compute_human_fleiss_kappa(review_files: list[Path]) -> dict[str, Any]:
    """Compute Fleiss's kappa across reviewers using only overall_valid."""
    if len(review_files) < 2:
        return {
            "error": (
                "Fleiss's kappa requires at least 2 review files. "
                f"Received: {len(review_files)}"
            )
        }

    reviewer_payloads: list[dict[str, Any]] = []
    missing_files = [str(path) for path in review_files if not path.exists()]
    if missing_files:
        return {"error": f"Review file(s) not found: {missing_files}"}

    for path in review_files:
        with open(path, encoding="utf-8") as f:
            annotations = json.load(f)
        reviewer_payloads.append({
            "annotator": annotations.get("annotator", path.stem),
            "path": str(path),
            "votes": _extract_reviewer_votes(annotations),
        })

    reviewer_question_sets = [set(payload["votes"]) for payload in reviewer_payloads]
    shared_questions = sorted(set.intersection(*reviewer_question_sets)) if reviewer_question_sets else []
    union_questions = sorted(set.union(*reviewer_question_sets)) if reviewer_question_sets else []

    if not shared_questions:
        return {"error": "No shared question_ids across the provided review files."}

    vote_rows: list[list[bool]] = []
    overall_question_votes: list[dict[str, Any]] = []
    for qid in shared_questions:
        votes = [
            payload["votes"].get(qid, {}).get("overall_valid")
            for payload in reviewer_payloads
        ]
        if any(vote is None for vote in votes):
            continue
        vote_rows.append([bool(vote) for vote in votes])
        accept_votes = sum(1 for vote in votes if vote is True)
        reject_votes = len(votes) - accept_votes
        overall_question_votes.append({
            "question_id": qid,
            "accept_votes": accept_votes,
            "reject_votes": reject_votes,
            "unanimous": accept_votes in {0, len(votes)},
        })

    overall_stats = _compute_fleiss_kappa(vote_rows)
    overall_stats["questions_used"] = len(overall_question_votes)
    return {
        "meta": {
            "reviewers": [payload["annotator"] for payload in reviewer_payloads],
            "review_files": [payload["path"] for payload in reviewer_payloads],
            "n_reviewers": len(reviewer_payloads),
            "n_questions_shared": len(shared_questions),
            "n_questions_union": len(union_questions),
        },
        "overall": overall_stats,
        "overall_question_votes": overall_question_votes,
    }


def _interpret_kappa(k: float | None) -> str:
    """Interpret Cohen's κ. None/NaN → N/A."""
    import math
    if k is None or (isinstance(k, float) and math.isnan(k)):
        return "N/A (all votes same class)"
    if k < 0:     return "Poor"
    if k < 0.20:  return "Slight"
    if k < 0.40:  return "Fair"
    if k < 0.60:  return "Moderate"
    if k < 0.80:  return "Substantial"
    return "Almost Perfect"


def _cm_stats(h_binary: list[int], l_binary: list[int]) -> dict[str, Any]:
    """Build confusion matrix + Cohen's κ + P/R/F1 from two binary lists.

    Handles degenerate case: when all votes are the same class (e.g. 100% pass),
    sklearn returns NaN → returns "N/A" with note.
    """
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

    # Manual kappa — handles degenerate case (all votes same class → p_e=1 → undefined)
    if abs(1 - p_e) > 1e-9:
        kappa_val = round((p_o - p_e) / (1 - p_e), 4)
        # κ ≈ 0 with high agreement = skewed-distribution artifact (not a problem).
        # When p_e → 1 (nearly all "pass"), κ = (p_o - p_e)/(1-p_e) ≈ 0 even when
        # agreement is 94.4% because there are too few negatives to measure variation.
        if abs(kappa_val) < 0.05 and p_o >= 0.90:
            kappa_interp = "N/A (κ ≈ 0 — skewed distribution; agreement is high)"
        else:
            kappa_interp = _interpret_kappa(kappa_val)
    else:
        kappa_val = None  # p_e ≈ 1, all votes same class → undefined
        kappa_interp = "N/A (all votes same class — κ undefined)"

    return {
        "n":              n,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "agreement_rate": round(p_o, 4),
        "p_o":            round(p_o, 4),
        "p_e":            round(p_e, 4),
        "kappa":          kappa_val,
        "kappa_interp":  kappa_interp,
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
                # labels=[0,1] required — prevents degenerate NaN when all votes are same class
                raw_kappa = cohen_kappa_score(h_vals, l_vals, labels=[0, 1])
                # sklearn still returns NaN when all votes identical and p_e≈1
                kappa = None if (raw_kappa != raw_kappa) else round(float(raw_kappa), 4)
                kappa_interp = _interpret_kappa(kappa)
            except Exception:
                kappa = None
                kappa_interp = "N/A"
            agree_rate = round(
                sum(1 for h, l in zip(h_vals, l_vals) if h == l) / len(h_vals), 4
            )
            per_criterion[criterion] = {
                "n":               len(h_vals),
                "agreement_rate": agree_rate,
                "cohens_kappa":   kappa,
                "kappa_interp":   kappa_interp,
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
                kappa = round(
                    float(cohen_kappa_score(h_vals, l_vals, labels=[0, 1])), 4
                )
                kappa_interp = _interpret_kappa(kappa)
            except Exception:
                kappa = None
                kappa_interp = "N/A"
            agree_rate = round(
                sum(1 for h, l in zip(h_vals, l_vals) if h == l) / len(h_vals), 4
            )
            per_criterion[criterion] = {
                "n":               len(h_vals),
                "agreement_rate": agree_rate,
                "cohens_kappa":   kappa,
                "kappa_interp":   kappa_interp,
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


# ── Legacy single-review entry point ─────────────────────────────────────────

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
        description="Run MCQGen evaluation metrics"
    )
    parser.add_argument(
        "--exp", required=True,
        help="Experiment name (e.g. exp_03_test_15q)"
    )
    parser.add_argument(
        "--human-json",
        dest="human_json",
        help=(
            "Optional review-file spec for Fleiss's kappa: comma-separated JSON paths, "
            "a directory containing *_review.json, or any path inside that directory. "
            "If omitted, the 4 default review files next to eval_metrics.py are used."
        )
    )
    parser.add_argument(
        "--human-csv",
        dest="human_json",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output",
        help="Save JSON output to this path (default: output/{EXP}/metrics_report.json)"
    )
    args = parser.parse_args()

    exp_dir       = Config.PROJECT_ROOT / "output" / args.exp
    accepted_file = exp_dir / "08_eval_iwf" / "final_accepted_questions.jsonl"
    topic_file    = Config.TOPIC_LIST_FILE

    gen_mcqs = load_jsonl(accepted_file) if accepted_file.exists() else []
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
    print("\n[1/4] Topic Coverage...")
    tc = compute_topic_coverage(gen_mcqs, topic_list)
    results["topic_coverage"] = tc
    print(f"  {tc['num_covered']}/{tc['num_total']} topics "
          f"({tc['coverage_ratio']*100:.1f}%)")

    # ── 2. Answer Ratio ───────────────────────────────────────────────────
    print("\n[2/4] Answer Ratio...")
    ar = compute_answer_ratio(gen_mcqs)
    results["answer_ratio"] = ar
    print(f"  Single: {ar['single_correct']}/{ar['total']} ({ar['single_pct']}%)  "
          f"Multiple: {ar['multiple_correct']}/{ar['total']} ({ar['multiple_pct']}%)")

    # ── 3. Diversity Openings ──────────────────────────────────────────────
    print("\n[3/4] Diversity Openings...")
    div = compute_diversity_openings(gen_mcqs)
    results["diversity_openings"] = div
    print(f"  Opening diversity: {div['opening_diversity_pct']}%")
    print(f"  Distinct opening signatures: {div['unique_opening_signatures']}  "
          f"| Effective openings: {div['effective_num_openings']}")
    print(f"  Normalized entropy: {div['normalized_opening_entropy']}  "
          f"| Repetition rate: {div['repetition_rate_pct']}%")

    # ── 4. Human Review Agreement ──────────────────────────────────────────
    print("\n[4/4] Human Review Agreement (Fleiss's κ)...")
    review_files = resolve_human_review_files(args.human_json)
    hk = compute_human_fleiss_kappa(review_files)
    results["human_fleiss_kappa"] = hk
    if "error" not in hk:
        meta = hk.get("meta", {})
        overall = hk.get("overall", {})
        print(f"  Reviewers: {meta.get('n_reviewers', 0)}  "
              f"| Shared questions: {meta.get('n_questions_shared', 0)}")
        print(f"  Raw agreement: {overall.get('raw_agreement_pct', 0):.1f}%")
        print(f"  Gwet's AC1: {overall.get('gwet_ac1', 'N/A')}  "
              f"| AC1 chance: {overall.get('gwet_chance_agreement', 'N/A')}")
        print(f"  Overall Fleiss's κ: {overall.get('fleiss_kappa', 'N/A')} "
              f"({overall.get('kappa_interp', 'N/A')})")
        print(f"  Mean item agreement: {overall.get('mean_item_agreement_pct', 0):.1f}%  "
              f"| Unanimous items: {overall.get('unanimous_items', 0)}/"
              f"{overall.get('n_items', 0)}")
        if overall.get("note"):
            print(f"  Note: {overall['note']}")
    else:
        print(f"  ⚠️  {hk['error']}")

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

    # ── 2. Answer Ratio ─────────────────────────────────────────────────────
    ar = results.get("answer_ratio", {})
    if ar:
        sp = ar.get("single_pct", 0)
        mp = ar.get("multiple_pct", 0)
        lines += [
            "\n## 2. Answer Ratio (Single vs Multiple Correct)\n",
            f"| Type | Count | Percentage |\n",
            f"|------|-------|------------|\n",
            f"| Single correct (1 đáp án) | {ar.get('single_correct', 0)} | {sp}% |\n",
            f"| Multiple correct (>1 đáp án) | {ar.get('multiple_correct', 0)} | {mp}% |\n",
            f"| **Total** | **{ar.get('total', 0)}** | 100% |\n",
            "\n",
            "> Target trong pipeline: **80% single, 20% multiple**.  \n",
            f"> Single: {sp}% (target: 80%)  |  Multiple: {mp}% (target: 20%)  \n",
            "> " + (
                "✅ Tỉ lệ phù hợp với target."
                if abs(sp - 80) <= 15 else
                "⚠️  Tỉ lệ lệch khỏi target."
            ) + "\n",
        ]

    # ── 3. Diversity Openings ──────────────────────────────────────────────
    div = results.get("diversity_openings", {})
    if div:
        tot = div.get("total", 0)
        lines += [
            "\n## 3. Diversity Openings (Đa dạng cách đặt câu hỏi)\n",
            f"| Metric | Value |\n",
            f"|--------|-------|\n",
            f"| Opening diversity score | {div.get('opening_diversity_pct', 0)}% |\n",
            f"| Unique opening signatures | {div.get('unique_opening_signatures', 0)} |\n",
            f"| Effective number of openings | {div.get('effective_num_openings', 0)} |\n",
            f"| Normalized opening entropy | {div.get('normalized_opening_entropy', 0)} |\n",
            f"| Repetition rate | {div.get('repetition_rate_pct', 0)}% |\n",
            "\n",
            "✅ **Primary score:** `opening_diversity_pct` = normalized Shannon entropy trên phân phối opening signatures.  \n",
            "Opening signature được lấy từ vài token đầu sau khi normalize text, nên metric đo mức độ các cách mở đầu bị dồn vào một vài template hay được phân tán đều hơn.\n",
            "\n",
            "> " + (
                "✅ Tốt — opening forms được phân bố khá đều."
                if div.get("opening_diversity_pct", 0) >= 80 else
                "⚠️  Cần cải thiện — opening forms còn tập trung vào ít template."
            ) + "\n",
        ]

        top_openings = div.get("top_openings", [])
        if top_openings:
            lines += [
                "\n**Top opening signatures:**\n",
                f"| Opening | Count | Percentage |\n",
                f"|---------|-------|------------|\n",
            ]
            for row in top_openings:
                lines.append(
                    f"| {row['opening']} | {row['count']} | {row['pct']}% |\n"
                )

    # ── 4. Human Review Agreement ──────────────────────────────────────────
    hk = results.get("human_fleiss_kappa", {})
    if hk and "error" not in hk:
        meta = hk.get("meta", {})
        overall = hk.get("overall", {})
        lines += [
            "\n## 4. Human Review Agreement (Fleiss's Kappa on overall_valid)\n",
            f"| Metric | Value |\n",
            f"|--------|-------|\n",
            f"| Reviewers | {meta.get('n_reviewers', 0)} |\n",
            f"| Reviewer names | {', '.join(meta.get('reviewers', [])) or 'N/A'} |\n",
            f"| Shared questions | {meta.get('n_questions_shared', 0)} |\n",
            f"| Question union | {meta.get('n_questions_union', 0)} |\n",
            f"| Raw agreement | **{overall.get('raw_agreement_pct', 0):.1f}%** |\n",
            f"| Gwet's AC1 | **{overall.get('gwet_ac1', 'N/A')}** |\n",
            f"| AC1 chance term | {overall.get('gwet_chance_agreement', 'N/A')} |\n",
            f"| Overall Fleiss's κ | **{overall.get('fleiss_kappa', 'N/A')}** ({overall.get('kappa_interp', 'N/A')}) |\n",
            f"| Mean item agreement | {overall.get('mean_item_agreement_pct', 0):.1f}% |\n",
            f"| Unanimous items | {overall.get('unanimous_items', 0)} / {overall.get('n_items', 0)} |\n",
            f"| Chance agreement | {overall.get('chance_agreement', 'N/A')} |\n",
        ]
        if overall.get("note"):
            lines += [f"\n> {overall['note']}\n"]
        lines += [
            "\n> `Raw agreement` là tỷ lệ đồng thuận quan sát trực tiếp, không hiệu chỉnh chance.  \n",
            "> `Gwet's AC1` cũng là chance-corrected agreement nhưng ổn định hơn Fleiss/Cohen kappa khi nhãn rất lệch.\n",
        ]
    else:
        hk_err = (hk or {}).get("error", "Human review agreement not computed.")
        lines += [
            "\n## 4. Human Review Agreement (Fleiss's Kappa on overall_valid)\n",
            f"> Not available: {hk_err}\n",
            "> By default, `eval_metrics.py` looks for the 4 `*_review.json` files next to itself.\n",
        ]

    # ── Summary table ─────────────────────────────────────────────────────
    def _cov_status() -> str:
        v = results.get("topic_coverage", {}).get("coverage_ratio", 0) * 100
        return "✅" if v >= 80 else "⚠️" if v >= 60 else "❌"

    def _ar_status() -> str:
        sp = results.get("answer_ratio", {}).get("single_pct", 0)
        return "✅" if abs(sp - 80) <= 15 else "⚠️"

    def _div_status() -> str:
        score = results.get("diversity_openings", {}).get("opening_diversity_pct", 0)
        return "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"

    def _hk_status() -> tuple[str, str]:
        hk2 = results.get("human_fleiss_kappa", {})
        if "error" in hk2 or not hk2:
            return "N/A", "⚠️"
        overall = hk2.get("overall", {})
        k = overall.get("fleiss_kappa")
        if k is None:
            agreement_pct = overall.get("mean_item_agreement_pct", 0.0)
            status = "✅" if agreement_pct >= 90 else "⚠️"
            return f"N/A ({agreement_pct:.1f}% agreement)", status
        return (
            f"{k:.4f}",
            "✅" if k >= 0.6 else "⚠️" if k >= 0.4 else "❌",
        )

    def _ac1_status() -> tuple[str, str]:
        hk2 = results.get("human_fleiss_kappa", {})
        if "error" in hk2 or not hk2:
            return "N/A", "⚠️"
        ac1 = hk2.get("overall", {}).get("gwet_ac1")
        if ac1 is None:
            return "N/A", "⚠️"
        return (
            f"{ac1:.4f}",
            "✅" if ac1 >= 0.6 else "⚠️" if ac1 >= 0.4 else "❌",
        )

    def _raw_agreement_status() -> tuple[str, str]:
        hk2 = results.get("human_fleiss_kappa", {})
        if "error" in hk2 or not hk2:
            return "N/A", "⚠️"
        raw = hk2.get("overall", {}).get("raw_agreement_pct")
        if raw is None:
            return "N/A", "⚠️"
        return (
            f"{raw:.1f}%",
            "✅" if raw >= 90 else "⚠️" if raw >= 80 else "❌",
        )

    hk_val, hk_flag = _hk_status()
    ac1_val, ac1_flag = _ac1_status()
    raw_val, raw_flag = _raw_agreement_status()

    lines += [
        "\n---\n",
        "## Summary\n",
        f"| Metric | Target | Result | Status |\n",
        f"|--------|-------|--------|--------|\n",
        f"| Topic Coverage | ≥ 80% | "
        f"{results.get('topic_coverage',{}).get('coverage_ratio',0)*100:.1f}% | {_cov_status()} |\n",
        f"| Answer Ratio (Single) | ≈ 80% | "
        f"{results.get('answer_ratio',{}).get('single_pct',0):.1f}% | {_ar_status()} |\n",
        f"| Diversity Openings | ≥ 80 entropy score | "
        f"{results.get('diversity_openings',{}).get('opening_diversity_pct',0):.1f}% | {_div_status()} |\n",
        f"| Human Raw Agreement | ≥ 90% | {raw_val} | {raw_flag} |\n",
        f"| Human Gwet's AC1 | ≥ 0.6 | {ac1_val} | {ac1_flag} |\n",
        f"| Human Fleiss's κ | ≥ 0.6 | {hk_val} | {hk_flag} |\n",
        "\n---\n",
        "*Generated by `eval_metrics.py` — MCQGen Pipeline*\n",
    ]
    return "".join(lines)


if __name__ == "__main__":
    main()
