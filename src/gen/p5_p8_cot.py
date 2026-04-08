"""
p5_p8_cot.py — Step 06: CoT Distractor Selection (P5 → P6 → P7 → P8)
MCQGen Pipeline: 6 candidates → P5 evaluate → P6 remove bad → P7 select → P8 assemble

Input:
  - data/intermediate/05_gen_distractors/all_candidates_results.jsonl

Output:
  - data/intermediate/06_gen_cot/all_final_mcqs.jsonl
"""

from __future__ import annotations

import json
import random
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (
    Config, config,
    build_p5_cot_evaluate, build_p6_remove_bad,
    build_p7_select_final, build_p8_assemble,
    parse_json_output,
    load_jsonl, save_jsonl,
    init_vllm_gen, make_vllm_sampling,
)


def assemble_mcq_fallback(
    refined_stem: dict,
    all_options: list[str],
    correct_options: list[str],
    question_id: str,
) -> dict:
    """
    Fallback deterministic nếu P8 fail: xáo trộn options + gán A/B/C/D.
    """
    all_opts = correct_options + all_options[:4 - len(correct_options)]
    random.seed(42)
    random.shuffle(all_opts)
    labels = ["A", "B", "C", "D"]
    opts = {labels[i]: all_opts[i] for i in range(min(4, len(all_opts)))}

    correct_labels = [l for l in labels if opts.get(l) in correct_options]
    q_type = refined_stem.get("question_type", "single_correct")
    type_label = Config.MCQ_LABEL_MULTIPLE if q_type == "multiple_correct" else Config.MCQ_LABEL_SINGLE
    difficulty = refined_stem.get("difficulty_label", "G2")
    stem = refined_stem.get("refined_question_text", refined_stem.get("question_text", ""))
    question_text = f"Câu 1. {type_label} ({difficulty}) {stem}"

    return {
        "question_id": question_id,
        "question_text": question_text,
        "question_type": q_type,
        "options": opts,
        "correct_answers": sorted(correct_labels),
        "correct_answer_count": len(correct_labels),
        "topic": refined_stem.get("topic", ""),
        "difficulty_label": difficulty,
        "used_concept_chunk_ids": refined_stem.get("_meta", {}).get("p1_question_text", ""),
        "used_assessment_item_ids": [],
        "style_alignment_note": "assembled via deterministic fallback",
        "_assembly_method": "fallback",
    }


def run_cot_chain_for_item(
    item: dict,
    llm,
    SamplingParams,
    q_counter: dict,
) -> dict | None:
    """
    Chạy P5 → P6 → P7 → P8 cho 1 item.
    """
    topic_id  = item.get("_meta", {}).get("topic_id", "unknown")
    chapter_id = item.get("_meta", {}).get("chapter_id", "unknown")
    seq       = item.get("_meta", {}).get("seq", 0)

    refined_stem = item.get("_refined_stem", item)
    candidates   = item.get("candidate_distractors", [])

    if not candidates:
        print(f"  ⚠️  No candidates for {topic_id} seq={seq}")
        return None

    correct_answers_content = refined_stem.get("correct_answers_content", [])
    correct_answer_count    = refined_stem.get("correct_answer_count", 1)
    num_distractors_needed  = 4 - correct_answer_count

    # ─── P5: evaluate all candidates ──────────────────────────────
    prompt_p5 = build_p5_cot_evaluate(refined_stem, candidates, correct_answers_content)
    raw_p5 = llm.chat(
        [{"role": "user", "content": prompt_p5}],
        SamplingParams(**make_vllm_sampling(0.1, 1024)),
    )[0].outputs[0].text
    p5_parsed = parse_json_output(raw_p5)
    p5_evals = p5_parsed.get("evaluations", [])

    # ─── P6: remove bad distractors ────────────────────────────────
    # Xây lại all_candidates list (candidates + correct answers để P6 đánh giá)
    all_opts_for_eval = candidates + correct_answers_content
    prompt_p6 = build_p6_remove_bad(refined_stem, all_opts_for_eval, p5_evals)
    raw_p6 = llm.chat(
        [{"role": "user", "content": prompt_p6}],
        SamplingParams(**make_vllm_sampling(0.1, 1024)),
    )[0].outputs[0].text
    p6_parsed = parse_json_output(raw_p6)
    kept_raw  = p6_parsed.get("kept_options", [])

    # Filter: chỉ giữ distractors (không phải correct answers)
    kept_distractors = [
        opt for opt in kept_raw
        if opt.get("option_text") not in correct_answers_content
    ]

    if len(kept_distractors) < num_distractors_needed:
        # Bổ sung từ candidates gốc nếu thiếu
        for c in candidates:
            if len(kept_distractors) >= num_distractors_needed:
                break
            if not any(k.get("option_text") == c for k in kept_distractors):
                kept_distractors.append({"option_text": c, "reason": "added_from_candidates_fallback"})

    # ─── P7: select final distractors ─────────────────────────────
    prompt_p7 = build_p7_select_final(refined_stem, kept_distractors, correct_answer_count)
    raw_p7 = llm.chat(
        [{"role": "user", "content": prompt_p7}],
        SamplingParams(**make_vllm_sampling(0.3, 1024)),
    )[0].outputs[0].text
    p7_parsed = parse_json_output(raw_p7)
    selected  = p7_parsed.get("selected_distractors", [])

    if len(selected) < num_distractors_needed:
        # Fallback: lấy đủ từ kept_distractors
        for kd in kept_distractors:
            if len(selected) >= num_distractors_needed:
                break
            if not any(s.get("option_text") == kd.get("option_text") for s in selected):
                selected.append({**kd, "error_type": kd.get("reason", "unknown"), "misleading_score": 5})

    # ─── P8: assemble final MCQ ───────────────────────────────────
    prompt_p8 = build_p8_assemble(
        refined_stem,
        selected[:num_distractors_needed],
        correct_answers_content,
    )
    raw_p8 = llm.chat(
        [{"role": "user", "content": prompt_p8}],
        SamplingParams(**make_vllm_sampling(Config.GEN_TEMPERATURE, Config.GEN_MAX_TOKENS)),
    )[0].outputs[0].text
    p8_parsed = parse_json_output(raw_p8)

    # Fallback assembly
    if "error" in p8_parsed or not p8_parsed.get("options"):
        q_counter[chapter_id] = q_counter.get(chapter_id, 0) + 1
        q_id = f"cs116_{chapter_id}_q_{q_counter[chapter_id]:04d}"
        p8_parsed = assemble_mcq_fallback(
            refined_stem,
            [s.get("option_text", "") for s in selected[:num_distractors_needed]],
            correct_answers_content,
            q_id,
        )

    # Gắn metadata
    p8_parsed["_meta"] = {
        **item.get("_meta", {}),
        "p5_evaluations_count": len(p5_evals),
        "p6_kept_distractors_count": len(kept_distractors),
        "p7_selected_count": len(selected),
        "p8_assembly_method": p8_parsed.get("_assembly_method", "llm"),
    }
    p8_parsed["_cot_steps"] = {
        "p5": p5_parsed,
        "p6": p6_parsed,
        "p7": p7_parsed,
    }

    return p8_parsed


def run_p5_p8_cot():
    """
    Entry point cho Step 06 — CoT distractor selection cho tất cả candidates.
    """
    config.makedirs()

    cand_file = Config.GEN_DISTR_OUTPUT / "all_candidates_results.jsonl"
    if not cand_file.exists():
        print(f"❌ Candidates not found: {cand_file}")
        print("   Chạy script 05_gen_distractors.sh trước.")
        sys.exit(1)

    candidates_items = load_jsonl(cand_file)
    print(f"📂 Loaded {len(candidates_items)} candidate items")

    print("🔄 Loading Qwen2.5-14B-Instruct (CoT distractor selection)...")
    llm, SamplingParams = init_vllm_gen()
    print("✅ Model loaded")

    q_counter = {}
    all_final = []
    for item in candidates_items:
        topic_id = item.get("_meta", {}).get("topic_id", "unknown")
        seq      = item.get("_meta", {}).get("seq", 0)
        try:
            result = run_cot_chain_for_item(item, llm, SamplingParams, q_counter)
            if result:
                all_final.append(result)
                print(f"  ✅ CoT {topic_id} seq={seq}: {result.get('question_type')}")
        except Exception as e:
            print(f"  ❌ Error {topic_id} seq={seq}: {e}")
            traceback.print_exc()

    out_file = Config.GEN_COT_OUTPUT / "all_final_mcqs.jsonl"
    save_jsonl(all_final, out_file)
    print(f"\n✅ CoT done. Total: {len(all_final)} final MCQs → {out_file}")


if __name__ == "__main__":
    run_p5_p8_cot()
