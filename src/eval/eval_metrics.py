"""
eval_metrics.py — Quantitative Evaluation Metrics for MCQ Generation
=====================================================================
Implements 5 metrics for MCQ pipeline quality assessment.

Metrics (no reference set required):
  1. Topic Coverage                (vs topic_list.json)
  2. LLM Judge Pass Rate           (from 07_eval + 08_eval_iwf output)
  3. Bloom Distribution KL Div.   (vs target G1→L1/L2, G2→L3+L4, G3→L5+L6)
  4. Answer Ratio                  (single vs multiple correct)
  5. Diversity Openings            (avoid weak/banned question openings)

Human Review:
  - compute_human_agreement_from_csv()   (Cohen's κ: human vs LLM judge accept/reject)
  - compute_inter_rater_kappa()           (Cohen's κ: detailed per-criterion)

Dependencies:
  pip install scikit-learn numpy scipy
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


# ==============================================================================
# 7. Human Labels CSV — Simple Accept/Reject vs LLM Judge
# ==============================================================================

def compute_human_agreement_from_csv(
    human_csv: Path,
    llm_results_file: Path,
) -> dict[str, Any]:
    """
    Simple human vs LLM agreement using only accept/reject per question.

    human_csv format (2 columns, no header):
        question_id,human_label
        ch04_t01_q0,accept
        ch04_t01_q1,reject

    human_label values: accept | reject (case-insensitive)

    llm_results_file: evaluated_results.jsonl — uses "accepted" field

    Computes: Agreement rate, Confusion matrix, Cohen's Kappa,
              Accuracy, Precision, Recall, F1
    """
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        return {"error": "scikit-learn not installed. Run: pip install scikit-learn"}

    # Load human labels
    human_labels: dict[str, int] = {}  # qid → 1 (accept) / 0 (reject)
    with open(human_csv, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("question_id"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            qid = parts[0].strip()
            label = parts[1].strip().lower()
            human_labels[qid] = 1 if label == "accept" else 0

    # Load LLM results
    llm_results_list = load_jsonl(llm_results_file)
    llm_by_id: dict[str, bool] = {}
    for q in llm_results_list:
        qid = q.get("question_id", "")
        llm_by_id[qid] = q.get("accepted", False)

    # Match
    matched = [(qid, human_labels[qid], bool(llm_by_id.get(qid, False)))
               for qid in human_labels if qid in llm_by_id]

    if not matched:
        return {
            "error": (f"No matching question_ids found. "
                      f"Human CSV has {len(human_labels)} entries, "
                      f"LLM file has {len(llm_by_id)} entries. "
                      f"Check that question_ids match.")
        }

    h_binary = [m[1] for m in matched]
    l_binary = [m[2] for m in matched]
    n = len(matched)

    agree = sum(1 for h, l in zip(h_binary, l_binary) if h == l)
    tp = sum(1 for h, l in zip(h_binary, l_binary) if h == 1 and l == 1)
    tn = sum(1 for h, l in zip(h_binary, l_binary) if h == 0 and l == 0)
    fp = sum(1 for h, l in zip(h_binary, l_binary) if h == 0 and l == 1)
    fn = sum(1 for h, l in zip(h_binary, l_binary) if h == 1 and l == 0)

    po = agree / n
    pe = (((tp + fp) / n) * ((tp + fn) / n)
          + ((fn + tn) / n) * ((fp + tn) / n))
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else 0.0

    accuracy  = agree / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    def _ki(k: float) -> str:
        if k < 0:    return "Poor"
        if k < 0.20: return "Slight"
        if k < 0.40: return "Fair"
        if k < 0.60: return "Moderate"
        if k < 0.80: return "Substantial"
        return "Almost Perfect"

    return {
        "metric": "human_llm_agreement",
        "n_matched": n,
        "agreement_rate": f"{agree/n*100:.1f}%",
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "cohens_kappa": round(kappa, 4),
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1_score":  round(f1,        4),
        "kappa_interpretation": _ki(kappa),
    }


# ==============================================================================
# CLI
# ==============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run MCQGen evaluation metrics")
    parser.add_argument("--exp", required=True, help="Experiment name (e.g. exp_03_test_15q)")
    parser.add_argument("--human-csv", default=None,
                        help="CSV: question_id,human_label (accept/reject)")
    parser.add_argument("--output", default=None, help="Save JSON output to file")
    args = parser.parse_args()

    exp_dir = Config.PROJECT_ROOT / "output" / args.exp
    accepted_file = exp_dir / "08_eval_iwf" / "final_accepted_questions.jsonl"
    eval_file = exp_dir / "07_eval" / "evaluated_questions.jsonl"
    topic_file = Config.TOPIC_LIST_FILE

    gen_mcqs   = load_jsonl(accepted_file) if accepted_file.exists() else []
    evaluated  = load_jsonl(eval_file) if eval_file.exists() else []
    try:
        with open(topic_file, encoding="utf-8") as f:
            topic_list = json.load(f)
    except Exception:
        topic_list = []

    print("\n" + "=" * 60)
    print(f"  MCQGen Metrics — {args.exp}")
    print("=" * 60)

    results: dict[str, Any] = {}

    # 1. Topic Coverage
    print("\n[1/5] Topic Coverage...")
    tc = compute_topic_coverage(gen_mcqs, topic_list)
    results["topic_coverage"] = tc
    print(f"  {tc['num_covered']}/{tc['num_total']} topics covered "
          f"({tc['coverage_ratio']*100:.1f}%)")

    # 2. Judge Pass Rate
    print("\n[2/5] Judge Pass Rate...")
    if evaluated and gen_mcqs:
        pr = compute_judge_pass_rate(eval_file, accepted_file)
        results["judge_pass_rate"] = pr
        if "error" not in pr:
            print(f"  Accepted: {pr['total_accepted']}/{pr['total_evaluated']} "
                  f"({pr['final_pass_rate']*100:.1f}%)")

    # 3. Bloom KL Divergence
    print("\n[3/5] Bloom Distribution KL...")
    bloom = compute_bloom_kl_divergence(gen_mcqs)
    results["bloom_kl_divergence"] = bloom
    print(f"  KL: {bloom['kl_divergence']:.4f}  Counts: {bloom['bloom_counts']}")

    # 4. Answer Ratio (single vs multiple)
    print("\n[4/5] Answer Ratio...")
    single = sum(1 for q in gen_mcqs if q.get("question_type") == "single_correct")
    multi  = sum(1 for q in gen_mcqs if q.get("question_type") == "multiple_correct")
    total  = len(gen_mcqs)
    results["answer_ratio"] = {
        "total": total,
        "single_correct": single,
        "multiple_correct": multi,
        "single_pct": round(single/total*100, 1) if total else 0,
        "multiple_pct": round(multi/total*100, 1) if total else 0,
    }
    print(f"  Single: {single}/{total} ({results['answer_ratio']['single_pct']}%)  "
          f"Multiple: {multi}/{total} ({results['answer_ratio']['multiple_pct']}%)")

    # 5. Diversity Openings
    print("\n[5/5] Diversity Openings...")
    avoid = ["hãy xác định", "khi nào", "ở đâu", "đâu là",
             "trong quá trình", "cho biết", "trong các phương pháp"]
    weak = sum(1 for q in gen_mcqs
               if any(p in q.get("question_text", "").lower() for p in avoid))
    results["diversity_openings"] = {
        "total": total,
        "weak_count": weak,
        "weak_pct": round(weak/total*100, 1) if total else 0,
    }
    print(f"  Weak openings: {weak}/{total} ({results['diversity_openings']['weak_pct']}%)")

    # Human vs LLM (CSV)
    if args.human_csv:
        print(f"\n[6/6] Human vs LLM Agreement...")
        ha = compute_human_agreement_from_csv(Path(args.human_csv), eval_file)
        results["human_llm_agreement"] = ha
        if "error" not in ha:
            print(f"  κ={ha['cohens_kappa']} ({ha['kappa_interpretation']})  "
                  f"Agree={ha['agreement_rate']}  F1={ha['f1_score']}")
            cm = ha["confusion_matrix"]
            print(f"  TP={cm['TP']} TN={cm['TN']} FP={cm['FP']} FN={cm['FN']}")
        else:
            print(f"  Error: {ha['error']}")

    print("\n" + "=" * 60)

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved: {out_path}")

        md_path = out_path.with_suffix(".md")
        md_report = generate_metrics_markdown(results, args.exp)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_report)
        print(f"✅ Markdown report: {md_path}")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))


def generate_metrics_markdown(results: dict, exp_name: str) -> str:
    """
    Generate a markdown explanation report for the metrics.
    Explains WHY each metric matters and interprets the results.
    """
    import datetime

    lines = [
        f"# MCQGen Pipeline — Metrics Report\n",
        f"**Experiment:** `{exp_name}`\n",
        f"**Generated:** {datetime.date.today().strftime('%Y-%m-%d')}\n",
        "---\n",
    ]

    # ── 1. Topic Coverage ─────────────────────────────────────────────────
    tc = results.get("topic_coverage", {})
    lines += [
        "## 1. Topic Coverage\n",
        "| Metric | Value |\n",
        "|--------|-------|\n",
        f"| Topics covered | {tc.get('num_covered', 0)}/{tc.get('num_total', 0)} |\n",
        f"| Coverage rate | {tc.get('coverage_ratio', 0)*100:.1f}% |\n",
        "\n### Tại sao dùng metric này?\n",
        "Topic Coverage đo **độ rộng của chương trình giảng dạy** — tỉ lệ các mục tiêu học tập\n",
        "trong `topic_list.json` có ít nhất 1 câu hỏi generated.\n",
        "Coverage thấp có nghĩa một số phần của môn học **không có câu hỏi**, tạo ra khoảng trống\n",
        "trong đánh giá sinh viên.\n",
        "\n### Cách đọc kết quả\n",
        "> Coverage **≥ 80%**: Tốt — phần lớn các topics đều được đánh giá.\n",
        "> Coverage **< 60%**: Cần xem lại `topic_list.json` và generation config — có thể pipeline\n",
        "> đang bỏ qua topics do `num_questions` thấp hoặc retrieval kém.\n",
    ]
    if tc.get("topics_missing"):
        lines += [
            "\n### Topics bị thiếu\n",
            "> " + ", ".join(tc["topics_missing"]) + "\n",
        ]

    # ── 2. Judge Pass Rate ────────────────────────────────────────────────
    pr = results.get("judge_pass_rate", {})
    if "error" not in pr:
        lines += [
            "\n## 2. LLM Judge Pass Rate\n",
            "| Metric | Value |\n",
            "|--------|-------|\n",
            f"| Đã evaluate | {pr.get('total_evaluated', 0)} |\n",
            f"| **Accepted** | **{pr.get('total_accepted', 0)} ({pr.get('final_pass_rate',0)*100:.1f}%)** |\n",
            f"| Rejected | {pr.get('total_rejected', 0)} |\n",
            "\n### Tại sao dùng metric này?\n",
            "LLM judge (Gemma-3-12b-it) đánh giá mỗi câu hỏi qua **6 tiêu chí**:\n",
            "1. **Format** — cấu trúc MCQ đúng (4 options, field đáp án)\n",
            "2. **Language** — tiếng Việt tự nhiên, không lỗi ngữ pháp\n",
            "3. **Relevance** — câu hỏi phù hợp với topic context\n",
            "4. **Answerability** — câu hỏi đủ rõ ràng để trả lời\n",
            "5. **Correct answer quality** — đáp án đúng chính xác\n",
            "6. **Distractor quality** — các options sai có vẻ hợp lý nhưng sai rõ ràng\n",
            "\n### Cách đọc kết quả\n",
            "> Pass rate **≥ 70%**: Pipeline chất lượng tốt.\n",
            "> Pass rate **50–70%**: Trung bình — vấn đề thường gặp: distractors yếu hoặc\n",
            "> poor alignment với course material.\n",
            "> Pass rate **< 50%**: Pipeline cần tuning — có thể stem generation hoặc\n",
            "> distractor generation cần cải thiện.\n",
        ]
        if pr.get("iwf_total"):
            lines += [
                "\n### Distractor Quality (IWF Filter)\n",
                "| Metric | Value |\n",
                "|--------|-------|\n",
                f"| Passed IWF | {pr.get('iwf_passed',0)}/{pr.get('iwf_total',0)} ({pr.get('iwf_pass_rate',0)*100:.1f}%) |\n",
                "\nIWF (Item Writing Flaws) kiểm tra distractor quality:\n",
                "- **Not plausibly wrong** — distractors quá dễ loại bỏ\n",
                "- **Overlapping with correct** — distractors chồng lấn với đáp án đúng\n",
                "- **Grammatically inconsistent** — cách viết option không khớp với stem\n",
                "- **Too similar to each other** — distractors thiếu discrimination power\n",
            ]
        cr = pr.get("criterion_pass_rates", {})
        if cr:
            lines += [
                "\n### Per-Criterion Pass Rates\n",
                "| Criterion | Pass Rate |\n",
                "|-----------|----------|\n",
            ]
            for c, rate in cr.items():
                label = c.replace("_pass", "").replace("_", " ").title()
                lines.append(f"| {label} | {rate*100:.1f}% |\n")
    else:
        lines += [f"\n## 2. Judge Pass Rate\n> Not available: {pr.get('error')}\n"]

    # ── 3. Bloom Distribution ─────────────────────────────────────────────
    bloom = results.get("bloom_kl_divergence", {})
    if bloom:
        counts = bloom.get("bloom_counts", {})
        target = bloom.get("target_distribution", [])
        kl = bloom.get("kl_divergence", 0)
        names = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
        n = sum(counts.values()) or 1

        lines += [
            "\n## 3. Bloom Taxonomy Distribution\n",
            "| Level | Name | Count | Actual % | Target % |\n",
            "|-------|------|-------|----------|----------|\n",
        ]
        for i in range(6):
            level = i + 1
            name = names[i]
            cnt = counts.get(level, 0)
            act_pct = cnt / n * 100
            tgt_pct = (target[i] * 100) if target else 0
            lines.append(f"| L{level} | {name} | {cnt} | {act_pct:.1f}% | {tgt_pct:.1f}% |\n")

        lines += [
            f"\n**KL Divergence:** {kl:.4f}  (thấp hơn = gần target hơn)\n",
            "\n### Tại sao dùng Bloom Taxonomy?\n",
            "Bloom's Taxonomy phân loại câu hỏi theo **độ phức tạp nhận thức**:\n",
            "| Level | Cognitive Skill | Example Verb |\n",
            "|-------|----------------|-------------|\n",
            "| L1 Remember | Recall facts | nhớ, liệt kê, định nghĩa |\n",
            "| L2 Understand | Explain concepts | giải thích, so sánh, mô tả |\n",
            "| L3 Apply | Use knowledge | áp dụng, tính toán, sử dụng |\n",
            "| L4 Analyze | Draw connections | phân tích, so sánh, đánh giá |\n",
            "| L5 Evaluate | Justify decisions | đánh giá, lựa chọn, phê phán |\n",
            "| L6 Create | Produce new work | thiết kế, xây dựng, đề xuất |\n",
            "\n### Tại sao dùng KL Divergence?\n",
            "KL divergence đo **mức độ lệch** phân bố thực tế so với target.\n",
            "- Target: G1 (dễ) → L1+L2, G2 (vừa) → L3+L4, G3 (khó) → L5+L6\n",
            "- KL **< 0.2**: Tốt — phân bố gần với kỳ vọng chương trình.\n",
            "- KL **0.2–0.5**: Trung bình — một số Bloom levels thiếu hoặc thừa.\n",
            "- KL **> 0.5**: Cao — quá nhiều câu hỏi ở một số Bloom levels.\n",
            "\n### Cách đọc kết quả\n",
            f"> KL = {kl:.4f}: " + (
                "✅ Tốt — phân bố gần target."
                if kl < 0.2 else
                "⚠️  Trung bình — một số Bloom levels thiếu/thừa."
                if kl < 0.5 else
                "❌ Cao — phân bố Bloom không cân bằng."
            ) + "\n",
        ]

    # ── 4. Answer Ratio ─────────────────────────────────────────────────────
    ar = results.get("answer_ratio", {})
    if ar:
        single_pct = ar.get("single_pct", 0)
        multi_pct  = ar.get("multiple_pct", 0)
        lines += [
            "\n## 4. Answer Ratio (Single vs Multiple)\n",
            f"| Type | Count | Percentage |\n",
            "|--------|-------|------------|\n",
            f"| Single correct (1 đáp án) | {ar.get('single_correct', 0)} | {single_pct}% |\n",
            f"| Multiple correct (>1 đáp án) | {ar.get('multiple_correct', 0)} | {multi_pct}% |\n",
            f"| **Total** | **{ar.get('total', 0)}** | 100% |\n",
            "\n### Tại sao dùng metric này?\n",
            "Tỉ lệ single vs multiple answer ảnh hưởng đến **độ khó và tính thực tế** của đề thi:\n",
            "- Câu single correct: thường dễ hơn, phù hợp G1/G2\n",
            "- Câu multiple correct: đòi hỏi hiểu sâu hơn, phù hợp G2/G3\n",
            "\n### Target trong pipeline\n",
            "- **80% single correct** → đảm bảo đề thi không quá khó\n",
            "- **20% multiple correct** → đảm bảo có câu hỏi nâng cao (5-7/30 câu)\n",
            "\n### Cách đọc kết quả\n",
            f"> Single: {single_pct}% (target: 80%)  |  Multiple: {multi_pct}% (target: 20%)\n",
            "> " + (
                "✅ Tỉ lệ phù hợp với target."
                if abs(single_pct - 80) <= 15 else
                "⚠️  Tỉ lệ lệch khỏi target."
            ) + "\n",
        ]

    # ── 5. Diversity Openings ─────────────────────────────────────────────
    div = results.get("diversity_openings", {})
    if div:
        weak_pct = div.get("weak_pct", 0)
        weak_cnt = div.get("weak_count", 0)
        total    = div.get("total", 0)
        lines += [
            "\n## 5. Diversity Openings (Đa dạng cách đặt câu hỏi)\n",
            f"| Metric | Value |\n",
            "|--------|-------|\n",
            f"| Weak openings | {weak_cnt}/{total} ({weak_pct}%) |\n",
            "\n### Tại sao dùng metric này?\n",
            "Câu hỏi bắt đầu bằng các cụm từ yếu gây **nhàm chán và thiếu tò mò** cho người học:\n",
            "❌ Các cách mở đầu **nên tránh**:\n",
            "- \"Hãy xác định...\", \"khi nào\", \"ở đâu\", \"đâu là\"\n",
            "- \"Trong quá trình...\", \"Cho biết...\", \"Trong các phương pháp...\"\n",
            "\n✅ Các cách mở đầu **nên dùng** (tạo sự tò mò):\n",
            "- \"Điều gì khiến...\", \"Đâu là điểm khác biệt giữa...\"\n",
            "- \"Nếu phải chọn giữa... và..., bạn sẽ ưu tiên điều gì?\"\n",
            "- \"Một mô hình có đặc điểm... sẽ hoạt động ra sao khi...?\"\n",
            "- \"Trường hợp nào sau đây minh họa đúng nhất về...?\"\n",
            "\n### Cách đọc kết quả\n",
            f"> Weak openings: {weak_pct}% \n",
            "> " + (
                "✅ Tốt — phần lớn câu hỏi có cách mở đầu đa dạng."
                if weak_pct < 20 else
                "⚠️  Cần cải thiện — nhiều câu hỏi dùng cách mở đầu yếu."
            ) + "\n",
        ]

    # ── 6. Human vs LLM Agreement ────────────────────────────────────────
    ha = results.get("human_llm_agreement", {})
    if "error" not in ha:
        cm = ha.get("confusion_matrix", {})
        lines += [
            "\n## 6. Human vs LLM Judge Agreement\n",
            "| Metric | Value | Interpretation |\n",
            "|--------|-------|---------------|\n",
            f"| Questions labeled | {ha.get('n_matched', 0)} | Human vs LLM both rated |\n",
            f"| **Cohen's κ** | **{ha.get('cohens_kappa', 0)}** | {ha.get('kappa_interpretation','')} |\n",
            f"| Agreement rate | {ha.get('agreement_rate', '')} | Both agree |\n",
            f"| Accuracy | {ha.get('accuracy',0)} | Overall correct match |\n",
            f"| Precision | {ha.get('precision',0)} | LLM accepts what human accepts |\n",
            f"| Recall | {ha.get('recall',0)} | Human accepts what LLM accepts |\n",
            f"| **F1 Score** | **{ha.get('f1_score',0)}** | Harmonic mean of P & R |\n",
            "\n### Confusion Matrix\n",
            "| | LLM Accept | LLM Reject |\n",
            "|---|---|---|\n",
            f"| Human Accept | TP={cm.get('TP',0)} | FN={cm.get('FN',0)} |\n",
            f"| Human Reject | FP={cm.get('FP',0)} | TN={cm.get('TN',0)} |\n",
            "\n### Tại sao dùng Cohen's Kappa?\n",
            "Cohen's κ đo **độ đồng thuận giữa hai người đánh giá** (human và LLM judge),\n",
            "ngoài yếu tố ngẫu nhiên:\n",
            "- κ > 0.8: Almost perfect\n",
            "- κ 0.6–0.8: Substantial\n",
            "- κ 0.4–0.6: Moderate\n",
            "- κ < 0.4: Fair / Poor (LLM judge có thể không align với human judgment)\n",
            "\n### Tại sao dùng F1 Score?\n",
            "F1 cân bằng **precision** (LLM tránh false accepts?) và **recall**\n",
            "(LLM nhận ra tất cả câu hỏi hợp lệ?).\n",
            "Dùng vì accept/reject là bài toán **classification không cân bằng**.\n",
            "\n### Cách đọc kết quả\n",
            f"> κ = {ha.get('cohens_kappa', 0)} ({ha.get('kappa_interpretation','')}):\n",
            f"> " + (
                "✅ LLM judge align tốt với human judgment — dùng được để filtering."
                if ha.get('cohens_kappa', 0) >= 0.6 else
                "⚠️  LLM judge align một phần với humans — nên review thủ công các câu borderline."
                if ha.get('cohens_kappa', 0) >= 0.4 else
                "❌ LLM judge lệch đáng kể với human judgment — cần xem lại criteria."
            ) + "\n",
            f"\n**Human accepts:** {ha.get('human_accepts',0)}  |  **Human rejects:** {ha.get('human_rejects',0)}\n",
            f"**LLM accepts:** {ha.get('llm_accepts',0)}  |  **LLM rejects:** {ha.get('llm_rejects',0)}\n",
        ]
    else:
        lines += [
            "\n## 6. Human vs LLM Judge Agreement\n",
            f"> Chưa tính: {ha.get('error')}\n",
            "> Để enable metric này, tạo file CSV `--human-csv` theo format:\n",
            "```\n",
            "question_id,human_label\n",
            "ch04_t01_q0,accept\n",
            "ch04_t01_q1,reject\n",
            "```\n",
        ]

    # ── Footer ─────────────────────────────────────────────────────────────
    lines += [
        "\n---\n",
        "## Summary\n",
        "| Metric | Target | Result | Status |\n",
        "|--------|-------|--------|--------|\n",
    ]

    def _status(key: str, threshold: float, higher: bool = True) -> str:
        if key == "topic_coverage":
            v = results.get("topic_coverage", {}).get("coverage_ratio", 0) * 100
        elif key == "judge_pass":
            v = results.get("judge_pass_rate", {}).get("final_pass_rate", 0) * 100
        elif key == "bloom_kl":
            v = results.get("bloom_kl_divergence", {}).get("kl_divergence", 999)
            higher = False
        elif key == "answer_ratio":
            v = results.get("answer_ratio", {}).get("single_pct", 0)
        elif key == "diversity":
            v = 100 - results.get("diversity_openings", {}).get("weak_pct", 100)
        elif key == "cohens_kappa":
            v = results.get("human_llm_agreement", {}).get("cohens_kappa", 0)
        else:
            v = 0.0
        if higher:
            return "✅" if v >= threshold else "⚠️" if v >= threshold * 0.75 else "❌"
        else:
            return "✅" if v <= threshold else "⚠️" if v <= threshold * 2 else "❌"

    rows = [
        ("Topic Coverage",         "topic_coverage", "≥ 80%",  80,  True),
        ("LLM Judge Pass Rate",    "judge_pass",     "≥ 70%",  70,  True),
        ("Bloom KL Div.",          "bloom_kl",       "≤ 0.3",  0.3, False),
        ("Answer Ratio (Single)",   "answer_ratio",   "≈ 80%",  65,  True),
        ("Diversity Openings",      "diversity",      "≥ 80%",  80,  True),
        ("Cohen's κ (human vs LLM)","cohens_kappa",   "≥ 0.6",  0.6, True),
    ]
    for label, key, target, threshold, higher in rows:
        ar2 = results.get("answer_ratio", {})
        val_str = ""
        if key == "topic_coverage":
            v = results.get("topic_coverage", {}).get("coverage_ratio", 0) * 100
            val_str = f"{v:.1f}%"
        elif key == "judge_pass":
            v = results.get("judge_pass_rate", {}).get("final_pass_rate", 0) * 100
            val_str = f"{v:.1f}%"
        elif key == "bloom_kl":
            v = results.get("bloom_kl_divergence", {}).get("kl_divergence", 0)
            val_str = f"{v:.4f}"
        elif key == "answer_ratio":
            v = ar2.get("single_pct", 0)
            val_str = f"{v:.1f}% single"
        elif key == "diversity":
            v = 100 - results.get("diversity_openings", {}).get("weak_pct", 0)
            val_str = f"{v:.1f}% good"
        elif key == "cohens_kappa":
            v = results.get("human_llm_agreement", {}).get("cohens_kappa", 0)
            val_str = f"{v:.4f}" if v else "N/A"
        lines.append(f"| {label} | {target} | {val_str} | {_status(key, threshold, higher)} |\n")

    lines += [
        "\n---\n",
        "*Generated by `eval_metrics.py` — MCQGen Pipeline*\n",
    ]
    return "".join(lines)


if __name__ == "__main__":
    main()
