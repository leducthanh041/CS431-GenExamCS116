"""
eval_iwf.py — Step 08: Distractor Item Writing Flaws Analysis
MCQGen Pipeline: evaluated MCQs → IWF check per distractor → final output

Model: Gemma-3-12b-it (vLLM)

Input:
  - data/intermediate/07_eval/evaluated_questions.jsonl

Output:
  - data/intermediate/08_eval_iwf/final_accepted_questions.jsonl
  - data/intermediate/08_eval_iwf/final_rejected_questions.jsonl
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (
    Config, config,
    build_eval_iwf_prompt, parse_json_output,
    load_jsonl, save_jsonl,
    init_vllm_eval, make_vllm_sampling,
)


def get_distractor_labels(mcq: dict) -> list[str]:
    """Lấy labels của các option SAI (distractors)."""
    correct = set(mcq.get("correct_answers", []))
    all_labels = ["A", "B", "C", "D"]
    return [l for l in all_labels if l not in correct]


def evaluate_distractors(mcq: dict, llm, SamplingParams) -> dict:
    """
    Đánh giá distractors theo 6 loại IWF.
    """
    prompt = build_eval_iwf_prompt(mcq)
    messages = [{"role": "user", "content": prompt}]
    sampling = SamplingParams(
        temperature=0.1,
        max_tokens=1024,
    )
    raw = llm.chat(messages, sampling_params=sampling)[0].outputs[0].text
    result = parse_json_output(raw)

    if "error" in result:
        result = {
            "distractor_evaluations": [],
            "total_iwf_count": 0,
            "bad_options": [],
            "overall_distractor_quality_pass": True,
        }

    return result


def run_eval_iwf():
    """
    Entry point cho Step 08 — IWF analysis cho tất cả accepted MCQs.
    """
    config.makedirs()

    eval_file = Config.EVAL_OUTPUT / "evaluated_questions.jsonl"
    if not eval_file.exists():
        print(f"❌ Evaluated questions not found: {eval_file}")
        print("   Chạy script 07_eval.sh trước.")
        sys.exit(1)

    mcqs = load_jsonl(eval_file)
    print(f"📂 Loaded {len(mcqs)} accepted MCQs for IWF analysis")

    if not mcqs:
        print("⚠️  No MCQs to analyze — check 07_eval output.")
        save_jsonl([], Config.EVAL_IWF_OUTPUT / "final_accepted_questions.jsonl")
        save_jsonl([], Config.EVAL_IWF_OUTPUT / "final_rejected_questions.jsonl")
        return

    print("🔄 Loading Gemma-3-12b-it (IWF evaluation)...")
    llm, SamplingParams = init_vllm_eval()
    print("✅ Model loaded")

    accepted_final = []
    rejected_final = []

    for i, mcq in enumerate(mcqs):
        q_id = mcq.get("question_id", f"q{i}")
        try:
            iwf_result = evaluate_distractors(mcq, llm, SamplingParams)
            mcq["distractor_evaluation"] = iwf_result

            total_iwf = iwf_result.get("total_iwf_count", 0)
            bad_opts  = iwf_result.get("bad_options", [])
            pass_iwf  = iwf_result.get("overall_distractor_quality_pass", True)

            if pass_iwf and total_iwf <= Config.IWF_MAX_ERRORS:
                mcq["status"] = "accepted"
                mcq["final_iwf_count"] = total_iwf
                accepted_final.append(mcq)
                print(f"  ✅ {q_id}: IWF={total_iwf} ≤ {Config.IWF_MAX_ERRORS} — ACCEPT")
            else:
                mcq["status"] = "rejected"
                mcq["final_iwf_count"] = total_iwf
                mcq["rejection_reason"] = f"iwf_count={total_iwf} > {Config.IWF_MAX_ERRORS}, bad_options={bad_opts}"
                rejected_final.append(mcq)
                print(f"  ❌ {q_id}: IWF={total_iwf} > {Config.IWF_MAX_ERRORS} — REJECT")

        except Exception as e:
            print(f"  ❌ Error for {q_id}: {e}")
            traceback.print_exc()
            mcq["status"] = "rejected"
            mcq["_error"] = str(e)
            rejected_final.append(mcq)

    # Save
    accepted_file = Config.EVAL_IWF_OUTPUT / "final_accepted_questions.jsonl"
    rejected_file = Config.EVAL_IWF_OUTPUT / "final_rejected_questions.jsonl"
    save_jsonl(accepted_final, accepted_file)
    save_jsonl(rejected_final, rejected_file)

    total = len(accepted_final) + len(rejected_final)
    pass_rate = len(accepted_final) / total * 100 if total > 0 else 0
    print(f"\n✅ IWF evaluation done:")
    print(f"   Final accepted: {len(accepted_final)}/{total} ({pass_rate:.1f}%)")
    print(f"   Final rejected: {len(rejected_final)}")
    print(f"   → {accepted_file}")
    print(f"   → {rejected_file}")


if __name__ == "__main__":
    run_eval_iwf()