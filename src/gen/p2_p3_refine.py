"""
p2_p3_refine.py — Step 04: Self-Refine Stem (P2 + P3)
MCQGen Pipeline: P1 stem → P2 suggest → P3 refined stem

Input:
  - data/intermediate/03_gen_stem/all_p1_results.jsonl

Output:
  - data/intermediate/04_gen_refine/all_refined_results.jsonl
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (
    Config, config,
    build_p2_refine_suggest, build_p3_refined_stem,
    parse_json_output,
    load_jsonl, save_jsonl,
    init_vllm_gen, make_vllm_sampling,
)


def run_refine_chain(
    p1_result: dict,
    llm,
    SamplingParams,
) -> dict | None:
    """
    Chạy P2 (suggest) → P3 (apply) cho 1 stem.
    Fallback: nếu P2/P3 fail thì dùng lại P1 result.
    """
    topic_id  = p1_result.get("_meta", {}).get("topic_id", "unknown")
    difficulty = p1_result.get("_meta", {}).get("difficulty", "G2")
    seq       = p1_result.get("_meta", {}).get("seq", 0)

    # ─── P2: đề xuất cải tiến ───────────────────────────────────────
    prompt_p2 = build_p2_refine_suggest(p1_result, difficulty)
    messages_p2 = [{"role": "user", "content": prompt_p2}]
    sampling_p2 = SamplingParams(
        **make_vllm_sampling(0.1, 512),
    )
    raw_p2 = llm.chat(messages_p2, sampling_params=sampling_p2)[0].outputs[0].text
    p2_parsed = parse_json_output(raw_p2)

    if "error" in p2_parsed:
        print(f"  ⚠️  P2 parse error for {topic_id} seq={seq}: {p2_parsed['error']}")
        p2_parsed = {"improvement_suggestion": "", "why_it_is_exam_appropriate": "", "difficulty_effect": "không đổi"}

    # ─── P3: áp dụng cải tiến ─────────────────────────────────────
    prompt_p3 = build_p3_refined_stem(p1_result, p2_parsed)
    messages_p3 = [{"role": "user", "content": prompt_p3}]
    sampling_p3 = SamplingParams(
        **make_vllm_sampling(Config.GEN_TEMPERATURE, Config.GEN_MAX_TOKENS),
    )
    raw_p3 = llm.chat(messages_p3, sampling_params=sampling_p3)[0].outputs[0].text
    p3_parsed = parse_json_output(raw_p3)

    if "error" in p3_parsed:
        print(f"  ⚠️  P3 parse error for {topic_id} seq={seq}: {p3_parsed['error']}")
        # Fallback: dùng lại P1 stem
        p3_parsed = {
            "refined_question_text": p1_result.get("question_text", ""),
            "question_type": p1_result.get("question_type", "single_correct"),
            "correct_answers_content": p1_result.get("correct_answers_content", []),
            "correct_answer_count": p1_result.get("correct_answer_count", 1),
            "topic": p1_result.get("topic", ""),
            "subtopic": p1_result.get("subtopic", ""),
            "difficulty_label": p1_result.get("difficulty_label", difficulty),
            "style_alignment_note": "fallback: P3 failed, used P1 stem",
        }

    # ─── Merge metadata ─────────────────────────────────────────
    refined = p3_parsed.copy()
    refined["_meta"] = {
        **p1_result.get("_meta", {}),
        "p2_suggestion": p2_parsed.get("improvement_suggestion", ""),
        "p2_why_appropriate": p2_parsed.get("why_it_is_exam_appropriate", ""),
        "p2_difficulty_effect": p2_parsed.get("difficulty_effect", ""),
        "p1_question_text": p1_result.get("question_text", ""),
        "p1_correct_answers": p1_result.get("correct_answers_content", []),
    }
    return refined


def run_p2_p3_refine():
    """
    Entry point cho Step 04 — chạy P2+P3 refine cho tất cả P1 results.
    """
    config.makedirs()

    # Load P1 results
    p1_file = Config.GEN_STEM_OUTPUT / "all_p1_results.jsonl"
    if not p1_file.exists():
        print(f"❌ P1 results not found: {p1_file}")
        print("   Chạy script 03_gen_stem.sh trước.")
        sys.exit(1)

    p1_results = load_jsonl(p1_file)
    print(f"📂 Loaded {len(p1_results)} P1 results")

    # Khởi tạo vLLM
    print("🔄 Loading Qwen2.5-14B-Instruct (refinement)...")
    llm, SamplingParams = init_vllm_gen()
    print("✅ Model loaded")

    all_refined = []
    for i, p1_result in enumerate(p1_results):
        topic_id = p1_result.get("_meta", {}).get("topic_id", "unknown")
        seq = p1_result.get("_meta", {}).get("seq", 0)
        try:
            refined = run_refine_chain(p1_result, llm, SamplingParams)
            all_refined.append(refined)
            print(f"  ✅ P2+P3 {topic_id} seq={seq}")
        except Exception as e:
            print(f"  ❌ Error {topic_id} seq={seq}: {e}")
            traceback.print_exc()

    # Save
    out_file = Config.GEN_REFINE_OUTPUT / "all_refined_results.jsonl"
    save_jsonl(all_refined, out_file)
    print(f"\n✅ P2+P3 done. Total: {len(all_refined)} refined → {out_file}")

    # ── Release VRAM before next pipeline step ──
    import gc, torch
    del llm
    del SamplingParams
    gc.collect()
    torch.cuda.empty_cache()
    print("  [cleanup] VRAM released")


if __name__ == "__main__":
    run_p2_p3_refine()
