"""
eval_wrapper.py — Load model ONCE, run ALL eval steps (eval_overall → eval_iwf).
Chạy trong 1 SLURM job trên GPU L40s: Gemma-3-12b-it (vLLM).
Single model load → single exit → no repeated init → no VRAM conflict.

Usage:  python -u src/eval/eval_wrapper.py
"""
from __future__ import annotations

import gc
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (
    Config, config,
    build_eval_overall_prompt, build_eval_iwf_prompt,
    parse_json_output,
    load_jsonl, save_jsonl,
    init_vllm_eval, make_vllm_sampling,
)


def get_distractor_labels(mcq):
    correct = set(mcq.get("correct_answers", []))
    all_labels = ["A", "B", "C", "D"]
    return [l for l in all_labels if l not in correct]


def evaluate_mcq(mcq, llm, SamplingParams):
    prompt = build_eval_overall_prompt(mcq)
    messages = [{"role": "user", "content": prompt}]
    sampling = SamplingParams(temperature=0.1, max_tokens=1024)
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
            "overall_valid": True,
            "fail_reasons": [f"parse_error: {result['error']}"],
            "quality_score": 0.5,
        }
    return result


def evaluate_distractors(mcq, llm, SamplingParams):
    prompt = build_eval_iwf_prompt(mcq)
    messages = [{"role": "user", "content": prompt}]
    sampling = SamplingParams(temperature=0.1, max_tokens=1024)
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


def main():
    config.makedirs()

    # Load model ONCE
    print("[INFO] Loading Gemma-3-12b-it (vLLM — one-time load)...")
    llm, SamplingParams = init_vllm_eval()
    print("[INFO] Model loaded")

    # ── Step 1: eval_overall ─────────────────────────────────────────────
    mcq_file = Config.GEN_COT_OUTPUT / "all_final_mcqs.jsonl"
    if not mcq_file.exists():
        print(f"[ERROR] Final MCQs not found: {mcq_file}")
        sys.exit(1)

    mcqs = load_jsonl(mcq_file)
    print(f"[INFO] Loaded {len(mcqs)} final MCQs")

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
            print(f"  ❌ eval error {topic_id}: {e}")
            traceback.print_exc()
            mcq["status"] = "rejected"
            mcq["_error"] = str(e)
            failed.append(mcq)

    passed_file = Config.EVAL_OUTPUT / "evaluated_questions.jsonl"
    failed_file = Config.EVAL_OUTPUT / "failed_questions.jsonl"
    save_jsonl(passed, passed_file)
    save_jsonl(failed, failed_file)

    total = len(passed) + len(failed)
    pass_rate = len(passed) / total * 100 if total > 0 else 0
    print(f"[INFO] eval_overall done: {len(passed)}/{total} ({pass_rate:.1f}%)")
    print(f"[INFO]   → {passed_file}")
    print(f"[INFO]   → {failed_file}")

    # ── Step 2: eval_iwf ─────────────────────────────────────────────────
    eval_file = Config.EVAL_OUTPUT / "evaluated_questions.jsonl"
    if not eval_file.exists():
        print(f"[ERROR] Evaluated questions not found: {eval_file}")
        sys.exit(1)

    mcqs_iwf = load_jsonl(eval_file)
    print(f"[INFO] Loaded {len(mcqs_iwf)} MCQs for IWF analysis")

    if not mcqs_iwf:
        print("[WARN] No MCQs to analyze — check eval_overall output.")
        save_jsonl([], Config.EVAL_IWF_OUTPUT / "final_accepted_questions.jsonl")
        save_jsonl([], Config.EVAL_IWF_OUTPUT / "final_rejected_questions.jsonl")
    else:
        accepted_final = []
        rejected_final = []

        for i, mcq in enumerate(mcqs_iwf):
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
                print(f"  ❌ IWF error {q_id}: {e}")
                traceback.print_exc()
                mcq["status"] = "rejected"
                mcq["_error"] = str(e)
                rejected_final.append(mcq)

        accepted_file = Config.EVAL_IWF_OUTPUT / "final_accepted_questions.jsonl"
        rejected_file = Config.EVAL_IWF_OUTPUT / "final_rejected_questions.jsonl"
        save_jsonl(accepted_final, accepted_file)
        save_jsonl(rejected_final, rejected_file)

        iwf_total = len(accepted_final) + len(rejected_final)
        iwf_rate = len(accepted_final) / iwf_total * 100 if iwf_total > 0 else 0
        print(f"[INFO] eval_iwf done: {len(accepted_final)}/{iwf_total} ({iwf_rate:.1f}%)")
        print(f"[INFO]   → {accepted_file}")
        print(f"[INFO]   → {rejected_file}")

    # ── Release VRAM ──────────────────────────────────────────────────────
    del llm
    del SamplingParams
    gc.collect()
    torch.cuda.empty_cache()
    free = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"[INFO] VRAM released. Free: {free:.1f} GiB")


if __name__ == "__main__":
    main()
