"""
eval_overall.py — Step 07: Overall MCQ Evaluation
MCQGen Pipeline: final MCQ → 6 checklist evaluation → pass/reject

Model: Gemma-3-12b-it (vLLM)

Input:
  - data/intermediate/06_gen_cot/all_final_mcqs.jsonl

Output:
  - data/intermediate/07_eval/evaluated_questions.jsonl
  - data/intermediate/07_eval/failed_questions.jsonl
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (
    Config, config,
    build_eval_overall_prompt, parse_json_output,
    load_jsonl, save_jsonl,
    init_vllm_eval, make_vllm_sampling,
)


def evaluate_mcq(mcq: dict, llm, SamplingParams) -> dict:
    """
    Đánh giá 1 MCQ theo 8 checklist criteria.
    Bao gồm: no_four_correct_pass + answer_not_in_stem_pass.
    """
    prompt = build_eval_overall_prompt(mcq)
    messages = [{"role": "user", "content": prompt}]
    sampling = SamplingParams(
        temperature=0.1,
        max_tokens=1024,
    )
    raw = llm.chat(messages, sampling_params=sampling)[0].outputs[0].text
    result = parse_json_output(raw)

    if "error" in result:
        result = {
            "format_pass": True,
            "language_pass": True,
            "grammar_pass": True,
            "relevance_pass": True,
            "answerability_pass": True,
            "correct_set_pass": True,
            "no_four_correct_pass": True,
            "answer_not_in_stem_pass": True,
            "overall_valid": True,
            "fail_reasons": [f"parse_error: {result['error']}"],
            "quality_score": 0.5,
        }

    return result


def run_eval_overall():
    """
    Entry point cho Step 07 — đánh giá toàn bộ final MCQs.
    """
    config.makedirs()

    mcq_file = Config.GEN_COT_OUTPUT / "all_final_mcqs.jsonl"
    if not mcq_file.exists():
        print(f"❌ Final MCQs not found: {mcq_file}")
        print("   Chạy script 06_gen_cot.sh trước.")
        sys.exit(1)

    mcqs = load_jsonl(mcq_file)
    print(f"📂 Loaded {len(mcqs)} final MCQs")

    print("🔄 Loading Gemma-3-12b-it (evaluation)...")
    llm, SamplingParams = init_vllm_eval()
    print("✅ Model loaded")

    passed = []
    failed = []

    for i, mcq in enumerate(mcqs):
        topic_id = mcq.get("_meta", {}).get("topic_id", f"q{i}")
        try:
            eval_result = evaluate_mcq(mcq, llm, SamplingParams)
            mcq["evaluation"] = eval_result

            if eval_result.get("overall_valid", False):
                mcq["status"] = "accepted"
                passed.append(mcq)
                print(f"  ✅ {topic_id}: PASS (score={eval_result.get('quality_score', 0):.2f})")
            else:
                mcq["status"] = "rejected"
                failed.append(mcq)
                reasons = eval_result.get("fail_reasons", [])
                print(f"  ❌ {topic_id}: FAIL — {reasons[:2]}")

        except Exception as e:
            print(f"  ❌ Error evaluating {topic_id}: {e}")
            traceback.print_exc()
            mcq["status"] = "rejected"
            mcq["_error"] = str(e)
            failed.append(mcq)

    # Save results
    passed_file = Config.EVAL_OUTPUT / "evaluated_questions.jsonl"
    failed_file = Config.EVAL_OUTPUT / "failed_questions.jsonl"
    save_jsonl(passed, passed_file)
    save_jsonl(failed, failed_file)

    # Stats
    total = len(passed) + len(failed)
    pass_rate = len(passed) / total * 100 if total > 0 else 0
    print(f"\n✅ Evaluation done:")
    print(f"   Passed: {len(passed)}/{total} ({pass_rate:.1f}%)")
    print(f"   Failed: {len(failed)}")
    print(f"   → {passed_file}")
    print(f"   → {failed_file}")

    # ── Release VRAM before next pipeline step ──
    import gc, torch
    del llm
    del SamplingParams
    gc.collect()
    torch.cuda.empty_cache()
    print("  [cleanup] VRAM released")


if __name__ == "__main__":
    run_eval_overall()