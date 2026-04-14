"""
compute_metrics.py — Entry point for quantitative MCQ metrics
=============================================================
Chạy: python scripts/compute_metrics.py [--annotations <file>]

Outputs:
  output/exp_01_baseline/metrics_report.json
  output/exp_01_baseline/review/metrics_with_human_kappa.json (nếu có annotations)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import Config, load_jsonl
from eval.eval_metrics import (
    compute_bleu_rouge,
    compute_bertscore,
    compute_topic_coverage,
    compute_judge_pass_rate,
    compute_bloom_kl_divergence,
    compute_inter_rater_kappa,
)


def load_topic_list() -> dict:
    with open(Config.TOPIC_LIST_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_trusted_ref() -> list[dict]:
    """Load only trusted_quiz items from assessment_items.jsonl."""
    all_items = load_jsonl(Config.ASSESSMENT_ITEMS_FILE)
    return [q for q in all_items if q.get("quality_level") == "trusted_quiz"]


def load_gen_mcqs() -> list[dict]:
    """Load final accepted questions."""
    path = Config.EVAL_IWF_OUTPUT / "final_accepted_questions.jsonl"
    if not path.exists():
        # Fallback: try intermediate output
        path = Config.GEN_COT_OUTPUT / "all_final_mcqs.jsonl"
    return load_jsonl(path)


def main():
    parser = argparse.ArgumentParser(description="Compute MCQ quantitative metrics")
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="Path to human review annotations JSON (enables Cohen's κ computation)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CS431MCQGen — Quantitative Evaluation Metrics")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────
    print("\n📂 Loading data...")
    gen_mcqs    = load_gen_mcqs()
    trusted_ref = load_trusted_ref()
    topic_list  = load_topic_list()

    print(f"   Gen MCQs:    {len(gen_mcqs)} questions")
    print(f"   Trusted ref: {len(trusted_ref)} questions (quality_level=trusted_quiz)")
    print(f"   Topics:      {len(topic_list)} chapters")

    # ── Compute metrics ────────────────────────────────────────────
    results: dict = {
        "experiment":     Config.EXP_NAME,
        "total_questions": len(gen_mcqs),
    }

    # 1. BLEU + ROUGE
    print("\n📐 [1/5] BLEU-1/2/4 + ROUGE-L (reference-based)...")
    bleu_rouge = compute_bleu_rouge(gen_mcqs, trusted_ref)
    if "error" not in bleu_rouge:
        print(f"   BLEU-1: {bleu_rouge.get('bleu1_mean', 'N/A'):.4f} ± {bleu_rouge.get('bleu1_std', 0):.4f}")
        print(f"   BLEU-2: {bleu_rouge.get('bleu2_mean', 'N/A'):.4f} ± {bleu_rouge.get('bleu2_std', 0):.4f}")
        print(f"   BLEU-4: {bleu_rouge.get('bleu4_mean', 'N/A'):.4f} ± {bleu_rouge.get('bleu4_std', 0):.4f}")
        print(f"   ROUGE-L: {bleu_rouge.get('rouge_l_mean', 'N/A'):.4f} ± {bleu_rouge.get('rouge_l_std', 0):.4f}")
        print(f"   Matched: {bleu_rouge.get('n_matched', 0)} questions")
    else:
        print(f"   ⚠️  {bleu_rouge['error']}")
    results["bleu_rouge"] = bleu_rouge

    # 2. BERTScore
    print("\n📐 [2/5] BERTScore-F1 (semantic similarity)...")
    bertscore = compute_bertscore(gen_mcqs, trusted_ref)
    if "error" not in bertscore:
        print(f"   Precision: {bertscore.get('bertscore_precision_mean', 'N/A'):.4f}")
        print(f"   Recall:    {bertscore.get('bertscore_recall_mean', 'N/A'):.4f}")
        print(f"   F1:        {bertscore.get('bertscore_f1_mean', 'N/A'):.4f}")
        print(f"   Matched:   {bertscore.get('n_matched', 0)} questions")
    else:
        print(f"   ⚠️  {bertscore['error']}")
    results["bertscore"] = bertscore

    # 3. Topic Coverage
    print("\n📐 [3/5] Topic Coverage...")
    topic_cov = compute_topic_coverage(gen_mcqs, topic_list)
    print(f"   Coverage: {topic_cov.get('coverage_ratio', 0):.1%}")
    print(f"   Topics covered: {topic_cov.get('num_covered', 0)}/{topic_cov.get('num_total', 0)}")
    if topic_cov.get("topics_missing"):
        print(f"   Missing topics: {', '.join(topic_cov['topics_missing'][:5])}...")
    results["topic_coverage"] = topic_cov

    # 4. LLM Judge Pass Rate
    print("\n📐 [4/5] LLM Judge Pass Rate...")
    evaluated_file = Config.EVAL_OUTPUT / "evaluated_questions.jsonl"
    iwf_file       = Config.EVAL_IWF_OUTPUT / "final_accepted_questions.jsonl"
    judge_rate = compute_judge_pass_rate(evaluated_file, iwf_file)
    if "error" not in judge_rate:
        print(f"   Final pass rate: {judge_rate.get('final_pass_rate', 0):.1%}")
        print(f"   Evaluated: {judge_rate.get('total_evaluated', 0)}, "
              f"Accepted: {judge_rate.get('total_accepted', 0)}, "
              f"Rejected: {judge_rate.get('total_rejected', 0)}")
        print(f"   IWF pass rate: {judge_rate.get('iwf_pass_rate', 0):.1%}")
        print("   Per-criterion:")
        for c, rate in judge_rate.get("criterion_pass_rates", {}).items():
            print(f"     {c}: {rate:.1%}")
        qs_stats = judge_rate.get("quality_score_stats", {})
        if qs_stats.get("quality_score_mean") is not None:
            print(f"   Quality score: {qs_stats.get('quality_score_mean', 0):.3f} "
                  f"± {qs_stats.get('quality_score_std', 0):.3f}")
    else:
        print(f"   ⚠️  {judge_rate['error']}")
    results["llm_judge"] = judge_rate

    # 5. Bloom KL Divergence
    print("\n📐 [5/5] Bloom Distribution KL Divergence...")
    bloom = compute_bloom_kl_divergence(gen_mcqs)
    print(f"   KL Divergence: {bloom.get('kl_divergence', 'N/A'):.4f}")
    dist = bloom.get("actual_distribution", [])
    print(f"   Actual Bloom dist (L1-L6): {[f'{d:.3f}' for d in dist]}")
    target = bloom.get("target_distribution", [])
    print(f"   Target Bloom dist (L1-L6): {[f'{t:.3f}' for t in target]}")
    results["bloom"] = bloom

    # ── Human review: Cohen's κ ────────────────────────────────────
    if args.annotations:
        ann_path = Path(args.annotations)
        print(f"\n📐 [Human] Cohen's κ — annotations from: {ann_path}")
        kappa_result = compute_inter_rater_kappa(ann_path, evaluated_file)
        if "error" not in kappa_result:
            print(f"   Questions annotated: {kappa_result.get('n_questions', 0)}")
            print(f"   Overall κ: {kappa_result.get('overall_kappa', 'N/A')} "
                  f"({kappa_result.get('overall_interpretation', 'N/A')})")
            print(f"   Overall agreement: {kappa_result.get('overall_agreement', 'N/A'):.1%}")
            print("   Per-criterion:")
            for c, info in kappa_result.get("per_criterion", {}).items():
                k = info.get("kappa")
                a = info.get("agreement")
                if k is not None:
                    print(f"     {c}: κ={k:.4f} ({info.get('interpretation','?')}) "
                          f"[agree={a:.1%}]")
        else:
            print(f"   ⚠️  {kappa_result['error']}")
        results["human_vs_llm"] = kappa_result

        # Save combined
        out_dir = Config.PROJECT_ROOT / "output" / Config.EXP_NAME / "review"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "metrics_with_human_kappa.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Saved: {out_file}")
    else:
        # Save base metrics
        out_dir = Config.PROJECT_ROOT / "output" / Config.EXP_NAME
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "metrics_report.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Saved: {out_file}")

    print("\n" + "=" * 60)
    print("Done. Run with --annotations <file> to compute Cohen's κ.")
    print("=" * 60)


if __name__ == "__main__":
    main()
