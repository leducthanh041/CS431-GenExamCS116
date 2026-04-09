"""
gen_wrapper.py — Load model ONCE, run ALL generation steps (P1→P2+P3→P4→P5-P8).
Chạy trong 1 SLURM job trên GPU L40s: Qwen2.5-14B-Instruct (vLLM).
Single model load → single exit → no repeated init → no VRAM conflict.

Usage:  python -u src/gen/gen_wrapper.py
"""
from __future__ import annotations

import gc
import random
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (
    Config, config,
    build_p1_gen_stem_key,
    build_p2_refine_suggest, build_p3_refined_stem,
    build_p4_option_candidates,
    build_p5_cot_evaluate, build_p6_remove_bad,
    build_p7_select_final, build_p8_assemble,
    parse_json_output,
    load_jsonl, save_jsonl,
    load_topic_list,
    init_vllm_gen, make_vllm_sampling,
)

# ─── P1 logic (extracted from p1_gen_stem.py) ─────────────────────────────────

def format_context(blocks):
    lines = []
    for i, blk in enumerate(blocks, 1):
        lines.append(f"--- Context block {i} ---")
        if blk.get("section_title"):
            lines.append(f"Tiêu đề: {blk['section_title']}")
        if blk.get("topic"):
            lines.append(f"Chủ đề: {blk['topic']}")
        if blk.get("text"):
            lines.append(blk["text"][:1500])
        lines.append("")
    return "\n".join(lines)


def run_p1_for_topic(topic_entry, retrieved_blocks, llm, SamplingParams):
    topic_id   = topic_entry["topic_id"]
    topic_name = topic_entry["topic_name"]
    difficulty = topic_entry.get("difficulty", "G2")
    num_q      = topic_entry.get("num_questions", 3)

    num_single = int(num_q * Config.SINGLE_CORRECT_RATIO)
    num_multi  = num_q - num_single
    num_two_correct    = int(num_multi * 0.5)
    num_three_correct = num_multi - num_two_correct

    context_str = format_context(retrieved_blocks)
    if not context_str.strip():
        print(f"  ⚠️  No context blocks for {topic_id} — skip")
        return []

    results = []
    for seq in range(num_q):
        if seq < num_single:
            q_type = "single_correct"
            correct_count = 1
        elif seq < num_single + num_two_correct:
            q_type = "multiple_correct"
            correct_count = 2
        else:
            q_type = "multiple_correct"
            correct_count = 3

        prompt = build_p1_gen_stem_key(
            topic=topic_name,
            difficulty_target=difficulty,
            concept_context_blocks=context_str,
            question_type_target=q_type,
            correct_answer_count_target=correct_count,
            num_questions_total=num_q,
            num_single_correct=num_single,
            num_multiple_correct=num_multi,
            num_two_correct=num_two_correct,
            num_three_correct=num_three_correct,
        )
        messages = [{"role": "user", "content": prompt}]
        sampling = SamplingParams(
            **make_vllm_sampling(Config.GEN_TEMPERATURE, Config.GEN_MAX_TOKENS),
        )
        raw = llm.chat(messages, sampling_params=sampling)[0].outputs[0].text
        parsed = parse_json_output(raw)

        if "error" in parsed:
            print(f"  ⚠️  P1 parse error {topic_id} seq={seq}: {parsed['error']}")
            continue

        parsed["_meta"] = {
            "topic_id": topic_id,
            "topic_name": topic_name,
            "difficulty": difficulty,
            "seq": seq,
            "prompt_version": "v3",
        }
        results.append(parsed)
        print(f"  ✅ P1 {topic_id} seq={seq}: type={q_type}, count={correct_count}")

    return results


# ─── P2+P3 logic (extracted from p2_p3_refine.py) ──────────────────────────────

def run_refine_chain(p1_result, llm, SamplingParams):
    topic_id    = p1_result.get("_meta", {}).get("topic_id", "unknown")
    difficulty  = p1_result.get("_meta", {}).get("difficulty", "G2")
    seq         = p1_result.get("_meta", {}).get("seq", 0)

    prompt_p2 = build_p2_refine_suggest(p1_result, difficulty)
    raw_p2 = llm.chat(
        [{"role": "user", "content": prompt_p2}],
        SamplingParams(**make_vllm_sampling(0.1, 512)),
    )[0].outputs[0].text
    p2_parsed = parse_json_output(raw_p2)

    if "error" in p2_parsed:
        print(f"  ⚠️  P2 parse error {topic_id} seq={seq}: {p2_parsed['error']}")
        p2_parsed = {
            "improvement_suggestion": "",
            "why_it_is_exam_appropriate": "",
            "difficulty_effect": "không đổi",
        }

    prompt_p3 = build_p3_refined_stem(p1_result, p2_parsed)
    raw_p3 = llm.chat(
        [{"role": "user", "content": prompt_p3}],
        SamplingParams(**make_vllm_sampling(Config.GEN_TEMPERATURE, Config.GEN_MAX_TOKENS)),
    )[0].outputs[0].text
    p3_parsed = parse_json_output(raw_p3)

    if "error" in p3_parsed:
        print(f"  ⚠️  P3 parse error {topic_id} seq={seq}: {p3_parsed['error']}")
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


# ─── P4 logic (extracted from p4_candidates.py) ────────────────────────────────

def run_p4_for_item(refined_item, llm, SamplingParams):
    topic_id = refined_item.get("_meta", {}).get("topic_id", "unknown")
    seq      = refined_item.get("_meta", {}).get("seq", 0)

    prompt = build_p4_option_candidates(
        refined_stem_key_json=refined_item,
        num_candidates=Config.NUM_CANDIDATE_DISTRACTORS,
    )
    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        SamplingParams(**make_vllm_sampling(Config.GEN_TEMPERATURE, Config.GEN_MAX_TOKENS)),
    )[0].outputs[0].text
    parsed = parse_json_output(raw)

    if "error" in parsed:
        print(f"  ⚠️  P4 parse error {topic_id} seq={seq}: {parsed['error']}")
        return None

    result = parsed.copy()
    result["_meta"] = {
        **refined_item.get("_meta", {}),
        "p4_candidates_count": len(parsed.get("candidate_distractors", [])),
    }
    result["_refined_stem"] = refined_item
    print(f"  ✅ P4 {topic_id} seq={seq}: {len(parsed.get('candidate_distractors', []))} candidates")
    return result


# ─── P5-P8 logic (extracted from p5_p8_cot.py) ────────────────────────────────

def assemble_mcq_fallback(refined_stem, all_options, correct_options, question_id):
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


def run_cot_chain_for_item(item, llm, SamplingParams, q_counter):
    topic_id   = item.get("_meta", {}).get("topic_id", "unknown")
    chapter_id = item.get("_meta", {}).get("chapter_id", "unknown")
    seq        = item.get("_meta", {}).get("seq", 0)

    refined_stem = item.get("_refined_stem", item)
    candidates  = item.get("candidate_distractors", [])

    if not candidates:
        print(f"  ⚠️  No candidates for {topic_id} seq={seq}")
        return None

    correct_answers_content = refined_stem.get("correct_answers_content", [])
    correct_answer_count    = refined_stem.get("correct_answer_count", 1)
    num_distractors_needed   = 4 - correct_answer_count

    # P5: evaluate all candidates
    prompt_p5 = build_p5_cot_evaluate(refined_stem, candidates, correct_answers_content)
    raw_p5 = llm.chat(
        [{"role": "user", "content": prompt_p5}],
        SamplingParams(**make_vllm_sampling(0.1, 1024)),
    )[0].outputs[0].text
    p5_parsed = parse_json_output(raw_p5)
    p5_evals  = p5_parsed.get("evaluations", [])

    # P6: remove bad distractors
    all_opts_for_eval = candidates + correct_answers_content
    prompt_p6 = build_p6_remove_bad(refined_stem, all_opts_for_eval, p5_evals)
    raw_p6 = llm.chat(
        [{"role": "user", "content": prompt_p6}],
        SamplingParams(**make_vllm_sampling(0.1, 1024)),
    )[0].outputs[0].text
    p6_parsed = parse_json_output(raw_p6)
    kept_raw  = p6_parsed.get("kept_options", [])

    # Filter: only distractors (not correct answers)
    correct_set = {c.get("option_text", "") for c in correct_answers_content}
    kept_distractors = [
        o for o in kept_raw
        if o.get("option_text", "") not in correct_set
    ]

    # P7: select final distractors
    prompt_p7 = build_p7_select_final(refined_stem, kept_distractors, correct_answer_count)
    raw_p7 = llm.chat(
        [{"role": "user", "content": prompt_p7}],
        SamplingParams(**make_vllm_sampling(0.3, 1024)),
    )[0].outputs[0].text
    p7_parsed = parse_json_output(raw_p7)
    selected  = p7_parsed.get("selected_distractors", [])

    if len(selected) < num_distractors_needed:
        for kd in kept_distractors:
            if len(selected) >= num_distractors_needed:
                break
            if not any(s.get("option_text") == kd.get("option_text") for s in selected):
                selected.append({**kd, "error_type": kd.get("reason", "unknown"), "misleading_score": 5})

    # P8: assemble final MCQ
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

    if "error" in p8_parsed or not p8_parsed.get("options"):
        q_counter[chapter_id] = q_counter.get(chapter_id, 0) + 1
        q_id = f"cs116_{chapter_id}_q_{q_counter[chapter_id]:04d}"
        p8_parsed = assemble_mcq_fallback(
            refined_stem,
            [s.get("option_text", "") for s in selected[:num_distractors_needed]],
            correct_answers_content,
            q_id,
        )

    p8_parsed["_meta"] = {
        **item.get("_meta", {}),
        "p5_evaluations_count": len(p5_evals),
        "p6_kept_distractors_count": len(kept_distractors),
        "p7_selected_count": len(selected),
        "p8_assembly_method": p8_parsed.get("_assembly_method", "llm"),
    }
    p8_parsed["_cot_steps"] = {"p5": p5_parsed, "p6": p6_parsed, "p7": p7_parsed}
    return p8_parsed


# ─── Main: load model once, run all steps ──────────────────────────────────────

def main():
    config.makedirs()

    # Load model ONCE
    print("[INFO] Loading Qwen2.5-14B-Instruct (vLLM — one-time load)...")
    llm, SamplingParams = init_vllm_gen()
    print("[INFO] Model loaded")

    # ── Step 1: P1 Gen Stem ─────────────────────────────────────────────────
    print("[INFO] P1 Gen Stem...")
    topics = load_topic_list()
    topic_entries = []
    for ch in topics:
        for t in ch.get("topics", []):
            t["chapter_id"] = ch["chapter_id"]
            t["chapter_name"] = ch["chapter_name"]
            topic_entries.append(t)

    all_p1 = []
    for entry in topic_entries:
        topic_id = entry["topic_id"]
        retrieve_file = Config.RETRIEVE_OUTPUT / f"{topic_id}.jsonl"
        if not retrieve_file.exists():
            print(f"  ⚠️  Retrieval not found: {retrieve_file} — skip {topic_id}")
            continue
        retrieved = load_jsonl(retrieve_file)
        blocks = retrieved[0].get("context_blocks", []) if retrieved else []
        try:
            results = run_p1_for_topic(entry, blocks, llm, SamplingParams)
            all_p1.extend(results)
        except Exception as e:
            print(f"  ❌ P1 error {topic_id}: {e}")
            traceback.print_exc()

    p1_file = Config.GEN_STEM_OUTPUT / "all_p1_results.jsonl"
    save_jsonl(all_p1, p1_file)
    print(f"[INFO] P1 done: {len(all_p1)} stems → {p1_file}")

    # ── Step 2: P2+P3 Refine ──────────────────────────────────────────────
    print("[INFO] P2+P3 Refine...")
    all_refined = []
    for p1_result in all_p1:
        topic_id = p1_result.get("_meta", {}).get("topic_id", "unknown")
        seq = p1_result.get("_meta", {}).get("seq", 0)
        try:
            refined = run_refine_chain(p1_result, llm, SamplingParams)
            all_refined.append(refined)
            print(f"  ✅ P2+P3 {topic_id} seq={seq}")
        except Exception as e:
            print(f"  ❌ P2+P3 error {topic_id} seq={seq}: {e}")
            traceback.print_exc()

    refine_file = Config.GEN_REFINE_OUTPUT / "all_refined_results.jsonl"
    save_jsonl(all_refined, refine_file)
    print(f"[INFO] P2+P3 done: {len(all_refined)} refined → {refine_file}")

    # ── Step 3: P4 Candidates ──────────────────────────────────────────────
    print("[INFO] P4 Candidates...")
    all_candidates = []
    for item in all_refined:
        topic_id = item.get("_meta", {}).get("topic_id", "unknown")
        seq = item.get("_meta", {}).get("seq", 0)
        try:
            result = run_p4_for_item(item, llm, SamplingParams)
            if result:
                all_candidates.append(result)
        except Exception as e:
            print(f"  ❌ P4 error {topic_id} seq={seq}: {e}")
            traceback.print_exc()

    cand_file = Config.GEN_DISTR_OUTPUT / "all_candidates_results.jsonl"
    save_jsonl(all_candidates, cand_file)
    print(f"[INFO] P4 done: {len(all_candidates)} → {cand_file}")

    # ── Step 4: P5-P8 CoT ────────────────────────────────────────────────
    print("[INFO] P5-P8 CoT...")
    q_counter = {}
    all_final = []
    for item in all_candidates:
        topic_id = item.get("_meta", {}).get("topic_id", "unknown")
        seq = item.get("_meta", {}).get("seq", 0)
        try:
            result = run_cot_chain_for_item(item, llm, SamplingParams, q_counter)
            if result:
                all_final.append(result)
                print(f"  ✅ CoT {topic_id} seq={seq}: {result.get('question_type')}")
        except Exception as e:
            print(f"  ❌ CoT error {topic_id} seq={seq}: {e}")
            traceback.print_exc()

    cot_file = Config.GEN_COT_OUTPUT / "all_final_mcqs.jsonl"
    save_jsonl(all_final, cot_file)
    print(f"[INFO] P5-P8 CoT done: {len(all_final)} MCQs → {cot_file}")

    # ── Release VRAM ──────────────────────────────────────────────────────
    del llm
    del SamplingParams
    gc.collect()
    torch.cuda.empty_cache()
    free = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"[INFO] VRAM released. Free: {free:.1f} GiB")


if __name__ == "__main__":
    main()
