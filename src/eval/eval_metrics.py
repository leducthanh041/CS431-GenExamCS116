"""
eval_metrics.py — Quantitative Evaluation Metrics for MCQ Generation
=====================================================================
Implements 6 Tier-1 metrics + Cohen's κ for human vs LLM judge reliability.

Metrics:
  1. BLEU-1, BLEU-2, BLEU-4        (reference-based, sacrebleu)
  2. ROUGE-L F1                    (reference-based, rouge_score)
  3. BERTScore-F1                  (semantic similarity, multilingual)
  4. Topic Coverage                (vs topic_list.json)
  5. LLM Judge Pass Rate           (from 07_eval + 08_eval_iwf output)
  6. Bloom Distribution KL Div.   (vs target G1→L1/L2, G2→L3/L4, G3→L5/L6)

Human Review:
  - compute_inter_rater_kappa()    (Cohen's κ: human vs Gemma judge)

Dependencies:
  pip install sacrebleu rouge-score bert-score scikit-learn numpy
"""

from __future__ import annotations

import json
import sys
import re
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import Config, load_jsonl


# ==============================================================================
# 1. BLEU + ROUGE-L
# ==============================================================================

def compute_bleu_rouge(gen_mcqs: list[dict], ref_mcqs: list[dict]) -> dict[str, Any]:
    """
    Compute BLEU-1/2/4 and ROUGE-L between generated and reference MCQs.
    Match by topic name. Reference set: trusted_quiz items (10 questions).

    Returns per-question scores + aggregate mean/std.
    """
    try:
        from sacrebleu import sentence_bleu
        from rouge_score import rouge_scorer
    except ImportError:
        return {
            "error": "sacrebleu or rouge-score not installed. Run: pip install sacrebleu rouge-score",
            "per_question": [],
            "bleu1_mean": None, "bleu1_std": None,
            "bleu2_mean": None, "bleu2_std": None,
            "bleu4_mean": None, "bleu4_std": None,
            "rouge_l_mean": None, "rouge_l_std": None,
        }

    # Build topic → ref stems mapping
    ref_by_topic: dict[str, list[str]] = {}
    for r in ref_mcqs:
        t = r.get("topic", "").strip()
        if t:
            ref_by_topic.setdefault(t, []).append(r["question_text"])

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results: list[dict] = []

    for gen in gen_mcqs:
        topic = (gen.get("topic") or "").strip()
        refs = ref_by_topic.get(topic, [])
        if not refs:
            continue

        hyp = gen["question_text"]
        bleu_scores: dict[str, float] = {}
        rouge_scores: list[float] = []

        for ref in refs:
            for n in [1, 2, 4]:
                key = f"bleu{n}"
                if key not in bleu_scores or bleu_scores[key] == 0:
                    score = sentence_bleu(hyp, [ref], tokenize="13a", max_order=n).score / 100
                    bleu_scores[key] = max(bleu_scores.get(key, 0), score)
            rouge = scorer.score(ref, hyp)["rougeL"].fmeasure
            rouge_scores.append(rouge)

        results.append({
            "question_id": gen.get("question_id", "unknown"),
            "topic":        topic,
            "bleu1":        bleu_scores.get("bleu1"),
            "bleu2":        bleu_scores.get("bleu2"),
            "bleu4":        bleu_scores.get("bleu4"),
            "rouge_l":      max(rouge_scores) if rouge_scores else None,
        })

    if not results:
        return {
            "note": "No topic-matched reference questions found. BLEU/ROUGE not computed.",
            "per_question": [],
            "bleu1_mean": None, "bleu1_std": None,
            "bleu2_mean": None, "bleu2_std": None,
            "bleu4_mean": None, "bleu4_std": None,
            "rouge_l_mean": None, "rouge_l_std": None,
        }

    def _agg(key: str) -> tuple[float, float]:
        vals = [r[key] for r in results if r.get(key) is not None]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)

    b1_m, b1_s = _agg("bleu1")
    b2_m, b2_s = _agg("bleu2")
    b4_m, b4_s = _agg("bleu4")
    rl_m,  rl_s = _agg("rouge_l")

    return {
        "per_question": results,
        "bleu1_mean": b1_m,  "bleu1_std": b1_s,
        "bleu2_mean": b2_m,  "bleu2_std": b2_s,
        "bleu4_mean": b4_m,  "bleu4_std": b4_s,
        "rouge_l_mean": rl_m, "rouge_l_std": rl_s,
        "n_matched": len(results),
        "note": f"Matched {len(results)} generated questions with {len(ref_by_topic)} reference topics",
    }


# ==============================================================================
# 2. BERTScore
# ==============================================================================

def compute_bertscore(gen_mcqs: list[dict], ref_mcqs: list[dict]) -> dict[str, Any]:
    """
    Compute BERTScore precision/recall/F1 using multilingual BERT.
    Match by topic. Reference: trusted_quiz items.

    Returns aggregate P/R/F1 mean across matched questions.
    """
    try:
        from bert_score import get_bertscore
    except ImportError:
        return {
            "error": "bert-score not installed. Run: pip install bert-score",
            "bertscore_precision_mean": None,
            "bertscore_recall_mean":    None,
            "bertscore_f1_mean":         None,
        }

    # Build topic → ref stems mapping
    ref_by_topic: dict[str, list[str]] = {}
    for r in ref_mcqs:
        t = r.get("topic", "").strip()
        if t:
            ref_by_topic.setdefault(t, []).append(r["question_text"])

    hyps, refs, ids, topics = [], [], [], []
    for gen in gen_mcqs:
        topic = (gen.get("topic") or "").strip()
        matched_refs = ref_by_topic.get(topic, [])
        if matched_refs:
            hyps.append(gen["question_text"])
            refs.append(matched_refs[0])   # best-match ref
            ids.append(gen.get("question_id", "unknown"))
            topics.append(topic)

    if not hyps:
        return {
            "note": "No topic-matched reference questions found. BERTScore not computed.",
            "bertscore_precision_mean": None,
            "bertscore_recall_mean":    None,
            "bertscore_f1_mean":         None,
        }

    P, R, F1 = get_bertscore(
        hyps, refs,
        lang="multilingual",
        model_type="bert-base-multilingual-cased",
        rescale_with_baseline=True,
    )

    P_l = P.tolist()
    R_l = R.tolist()
    F1_l = F1.tolist()

    return {
        "per_question": [
            {"question_id": ids[i], "topic": topics[i],
             "precision": P_l[i], "recall": R_l[i], "f1": F1_l[i]}
            for i in range(len(ids))
        ],
        "bertscore_precision_mean": float(np.mean(P_l)),
        "bertscore_recall_mean":    float(np.mean(R_l)),
        "bertscore_f1_mean":         float(np.mean(F1_l)),
        "bertscore_precision_std":  float(np.std(P_l)),
        "bertscore_recall_std":     float(np.std(R_l)),
        "bertscore_f1_std":          float(np.std(F1_l)),
        "n_matched": len(hyps),
    }


# ==============================================================================
# 3. Topic Coverage
# ==============================================================================

def compute_topic_coverage(
    gen_mcqs: list[dict],
    topic_list: dict,
) -> dict[str, Any]:
    """
    Compute topic coverage: what % of topics in topic_list.json
    have at least one generated MCQ.
    """
    # Topics from generated MCQs
    gen_topics: set[str] = set()
    for q in gen_mcqs:
        t = (q.get("topic") or q.get("_meta", {}).get("topic_name", "") or "").strip()
        if t:
            gen_topics.add(t)

    # Topics from topic_list.json (list of chapters)
    ref_topics: set[str] = set()
    for ch in topic_list:  # topic_list is a list, not dict
        for t in ch.get("topics", []):
            ref_topics.add(t["topic_name"])

    covered = gen_topics & ref_topics
    missing = ref_topics - gen_topics

    # Also check per-chapter coverage
    chapter_coverage: dict[str, dict] = {}
    for ch in topic_list:  # list
        ch_name = ch.get("chapter_name", ch.get("chapter_id", "?"))
        ch_topics = {t["topic_name"] for t in ch.get("topics", [])}
        ch_covered = ch_topics & gen_topics
        chapter_coverage[ch_name] = {
            "covered": sorted(ch_covered),
            "missing": sorted(ch_topics - gen_topics),
            "ratio": len(ch_covered) / len(ch_topics) if ch_topics else 0,
        }

    return {
        "coverage_ratio":   len(covered) / len(ref_topics) if ref_topics else 0,
        "topics_covered":   sorted(covered),
        "topics_missing":   sorted(missing),
        "num_covered":      len(covered),
        "num_total":        len(ref_topics),
        "extra_topics":     sorted(gen_topics - ref_topics),  # topics in gen but not in list
        "chapter_coverage": chapter_coverage,
    }


# ==============================================================================
# 4. LLM Judge Pass Rate
# ==============================================================================

def compute_judge_pass_rate(
    evaluated_file: Path,
    iwf_file: Path,
) -> dict[str, Any]:
    """
    Aggregate pass/reject rates from the existing pipeline outputs:
    - 07_eval: evaluated_questions.jsonl (pass/fail on 6 criteria)
    - 08_eval_iwf: final_accepted_questions.jsonl (final accepted after IWF)

    Returns per-criterion pass rates and overall rates.
    """
    if not evaluated_file.exists():
        return {"error": f"File not found: {evaluated_file}"}
    if not iwf_file.exists():
        return {"error": f"File not found: {iwf_file}"}

    evaluated = load_jsonl(evaluated_file)
    iwf_accepted = load_jsonl(iwf_file)

    # Overall final pass rate
    total_evaluated = len(evaluated)
    total_accepted = len(iwf_accepted)
    total_rejected = total_evaluated - total_accepted
    final_pass_rate = total_accepted / total_evaluated if total_evaluated else 0

    # Per-criterion pass rate from 07_eval
    criteria = [
        "format_pass", "language_pass", "grammar_pass",
        "relevance_pass", "answerability_pass", "correct_set_pass",
    ]
    criterion_rates: dict[str, float] = {}
    for c in criteria:
        passed = sum(1 for q in evaluated if q.get("evaluation", {}).get(c, False))
        criterion_rates[c] = passed / total_evaluated if total_evaluated else 0

    # IWF pass rate
    iwf_passed = sum(
        1 for q in iwf_accepted
        if q.get("distractor_evaluation", {}).get("overall_distractor_quality_pass", False)
    )
    iwf_total = len(iwf_accepted)
    iwf_pass_rate = iwf_passed / iwf_total if iwf_total else 0

    # Quality score stats
    scores = [q.get("evaluation", {}).get("quality_score") for q in evaluated
              if q.get("evaluation", {}).get("quality_score") is not None]
    score_stats = {
        "quality_score_mean":   float(np.mean(scores)) if scores else None,
        "quality_score_std":   float(np.std(scores))  if scores else None,
        "quality_score_median": float(np.median(scores)) if scores else None,
        "quality_score_min":    float(np.min(scores))  if scores else None,
        "quality_score_max":    float(np.max(scores))  if scores else None,
    }

    return {
        "final_pass_rate":   final_pass_rate,
        "total_evaluated":   total_evaluated,
        "total_accepted":    total_accepted,
        "total_rejected":    total_rejected,
        "iwf_pass_rate":     iwf_pass_rate,
        "iwf_passed":        iwf_passed,
        "iwf_total":          iwf_total,
        "criterion_pass_rates": criterion_rates,
        "quality_score_stats":  score_stats,
    }


# ==============================================================================
# 5. Bloom Distribution KL Divergence
# ==============================================================================

# Bloom's Taxonomy Vietnamese keywords for classification
BLOOM_KEYWORDS: dict[int, list[str]] = {
    # Level 1 — Remember
    1: ["nhớ", "định nghĩa", "liệt kê", "trình bày", "nêu", "cho biết",
        "thuộc tính", "công thức", "hàm", "lệnh", " cú pháp", "là gì"],
    # Level 2 — Understand
    2: ["giải thích", "ví dụ", "so sánh", "phân biệt", "tổng hợp",
        "mô tả", "trình bày", "tại sao", "như thế nào", "hoạt động"],
    # Level 3 — Apply
    3: ["áp dụng", "sử dụng", "thực hiện", "tính toán", "viết code",
        "chạy", "kết quả", "đầu ra", "đầu vào", "giá trị"],
    # Level 4 — Analyze
    4: ["phân tích", "tại sao", "lỗi", "sai", "đúng", "so sánh",
        "đánh giá", "hiệu suất", "độ phức tạp", "cách cải thiện"],
    # Level 5 — Evaluate
    5: ["đánh giá", "lựa chọn", "tốt nhất", "hiệu quả nhất", "nên",
        "không nên", "ưu nhược điểm", "so sánh và chọn"],
    # Level 6 — Create
    6: ["thiết kế", "xây dựng", "cải tiến", "đề xuất", "tạo ra",
        "lập trình", "phát triển", "viết chương trình"],
}

# Target distribution: G1→L1+L2, G2→L3+L4, G3→L5+L6
BLOOM_TARGET: dict[int, float] = {
    1: 0.20,  # G1: 40% total
    2: 0.20,
    3: 0.20,  # G2: 40% total
    4: 0.20,
    5: 0.10,  # G3: 20% total
    6: 0.10,
}


def _classify_bloom(text: str) -> int:
    """Classify text into Bloom level (1-6) using keyword matching."""
    text_lower = text.lower()
    scores: dict[int, float] = {i: 0 for i in range(1, 7)}
    for level, keywords in BLOOM_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[level] += 1
    best = max(scores, key=scores.get)
    # Fallback to G2→L3-L4 mapping if no keywords found
    return best if scores[best] > 0 else 3


def compute_bloom_kl_divergence(gen_mcqs: list[dict]) -> dict[str, Any]:
    """
    Classify each question into Bloom level via keyword matching.
    Compute actual vs target distribution, then KL divergence.

    Target: G1→{L1,L2}, G2→{L3,L4}, G3→{L5,L6}
    """
    try:
        from scipy.stats import entropy
    except ImportError:
        # Fallback: manual KL
        def entropy_simple(p, q):
            return sum(pi * (np.log(pi) - np.log(qi + 1e-9))
                       for pi, qi in zip(p, q) if pi > 0)
        _entropy = entropy_simple
    else:
        _entropy = entropy

    bloom_counts: dict[int, int] = {i: 0 for i in range(1, 7)}
    per_question: list[dict] = []

    for q in gen_mcqs:
        stem = q.get("question_text", "")
        diff = q.get("difficulty_label", q.get("_meta", {}).get("difficulty", "G2"))
        level = _classify_bloom(stem)
        bloom_counts[level] += 1
        per_question.append({
            "question_id": q.get("question_id", "unknown"),
            "difficulty": diff,
            "bloom_level": level,
            "bloom_name": ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"][level - 1],
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
        cnt = {i: 0 for i in range(1, 7)}
        for pq in diff_qs:
            cnt[pq["bloom_level"]] += 1
        tot = len(diff_qs) or 1
        diff_bloom[diff] = {i: cnt[i] / tot for i in range(1, 7)}

    return {
        "per_question": per_question,
        "bloom_counts": bloom_counts,
        "actual_distribution": actual.tolist(),
        "target_distribution": target.tolist(),
        "kl_divergence":       kl_div,
        "per_difficulty_bloom": diff_bloom,
        "note": "Bloom levels classified via keyword matching. KL target: G1→{L1,L2}, G2→{L3,L4}, G3→{L5,L6}",
    }


# ==============================================================================
# 6. Cohen's κ — Human vs LLM Judge
# ==============================================================================

def compute_inter_rater_kappa(
    annotation_file: Path,
    llm_results_file: Path,
) -> dict[str, Any]:
    """
    Compute Cohen's κ between human annotations and Gemma-3-12b-it judge.

    annotation_file: JSON with structure {"verdicts": {qid: {criterion: bool}}}
    llm_results_file: evaluated_questions.jsonl

    Returns per-criterion κ + overall κ.
    """
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        return {"error": "scikit-learn not installed. Run: pip install scikit-learn"}

    if not annotation_file.exists():
        return {"error": f"Annotation file not found: {annotation_file}"}
    if not llm_results_file.exists():
        return {"error": f"LLM results file not found: {llm_results_file}"}

    with open(annotation_file, encoding="utf-8") as f:
        annotations = json.load(f)

    verdicts: dict[str, dict] = annotations.get("verdicts", {})
    if not verdicts:
        return {"error": "No verdicts found in annotation file"}

    annotator = annotations.get("annotator", "unknown")

    # Load LLM results → dict keyed by question_id
    llm_results_list = load_jsonl(llm_results_file)
    llm_by_id: dict[str, dict] = {q.get("question_id", f"q{i}"): q.get("evaluation", {})
                                  for i, q in enumerate(llm_results_list)}

    # Match question IDs
    criteria = [
        "format_pass", "language_pass", "grammar_pass",
        "relevance_pass", "answerability_pass", "correct_set_pass",
        "overall_valid",
    ]

    kappas: dict[str, Any] = {}
    per_q: list[dict] = []

    for qid, human_v in verdicts.items():
        if qid not in llm_by_id:
            continue
        llm_v = llm_by_id[qid]
        q_result: dict[str, Any] = {"question_id": qid}
        for c in criteria:
            h = human_v.get(c)
            g = llm_v.get(c)
            if h is not None and g is not None:
                q_result[f"human_{c}"] = h
                q_result[f"llm_{c}"]   = g
                q_result[f"agree_{c}"] = (h == g)
        per_q.append(q_result)

        for c in criteria:
            h = human_v.get(c)
            g = llm_v.get(c)
            if h is not None and g is not None:
                kappas.setdefault(c, {"human": [], "llm": []})
                kappas[c]["human"].append(int(h))
                kappas[c]["llm"].append(int(g))

    # Compute κ per criterion
    criterion_kappas: dict[str, float] = {}
    agreement_rates: dict[str, float] = {}

    for c in criteria:
        if c in kappas and kappas[c]["human"]:
            pairs = [(h, g) for h, g in zip(kappas[c]["human"], kappas[c]["llm"])]
            agree = sum(1 for h, g in pairs if h == g) / len(pairs)
            kappa = float(cohen_kappa_score(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
            ))
            criterion_kappas[c] = round(kappa, 4)
            agreement_rates[c] = round(agree, 4)

    # Overall κ (all criteria pooled)
    all_h, all_g = [], []
    for c in criteria:
        if c in kappas and kappas[c]["human"]:
            all_h.extend(kappas[c]["human"])
            all_g.extend(kappas[c]["llm"])

    overall_kappa = float(cohen_kappa_score(all_h, all_g)) if all_h else None
    overall_agreement = sum(1 for h, g in zip(all_h, all_g) if h == g) / len(all_h) if all_h else None

    # κ interpretation
    def interpret_kappa(k: float) -> str:
        if k < 0:    return "Poor"
        if k < 0.20: return "Slight"
        if k < 0.40: return "Fair"
        if k < 0.60: return "Moderate"
        if k < 0.80: return "Substantial"
        return "Almost Perfect"

    return {
        "annotator":        annotator,
        "n_questions":      len(per_q),
        "overall_kappa":    round(overall_kappa, 4) if overall_kappa is not None else None,
        "overall_agreement": round(overall_agreement, 4) if overall_agreement is not None else None,
        "overall_interpretation": interpret_kappa(overall_kappa) if overall_kappa is not None else None,
        "per_criterion": {
            c: {
                "kappa": criterion_kappas.get(c),
                "agreement": agreement_rates.get(c),
                "interpretation": interpret_kappa(criterion_kappas.get(c, -1)),
            }
            for c in criteria
        },
        "per_question": per_q,
    }
