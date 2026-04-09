"""
p1_gen_stem.py — Step 03: Generate Stem + Key
MCQGen Pipeline: RAG → P1 → P2 → P3 → P4 → P5-P8 → MCQ
Dùng: Qwen2.5-14B-Instruct (vLLM)

Input:
  - data/intermediate/02_retrieval/<topic_id>.jsonl  (retrieved context blocks)
  - input/topic_list.json                           (topic metadata)

Output:
  - data/intermediate/03_gen_stem/<topic_id>.jsonl
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

# Add parent to path for common import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (
    Config, config,
    build_p1_gen_stem_key,
    parse_json_output,
    load_jsonl, save_jsonl,
    load_topic_list,
    init_vllm_gen, make_vllm_sampling,
)


def format_context(blocks: list[dict]) -> str:
    """Gộp nhiều context blocks thành 1 chuỗi cho prompt."""
    lines = []
    for i, blk in enumerate(blocks, 1):
        lines.append(f"--- Context block {i} ---")
        if blk.get("section_title"):
            lines.append(f"Tiêu đề: {blk['section_title']}")
        if blk.get("topic"):
            lines.append(f"Chủ đề: {blk['topic']}")
        if blk.get("text"):
            lines.append(blk["text"][:1500])  # Giới hạn độ dài
        lines.append("")
    return "\n".join(lines)


def run_p1_for_topic(
    topic_entry: dict,
    retrieved_blocks: list[dict],
    llm,
    SamplingParams,
) -> list[dict]:
    """
    Chạy P1 cho 1 topic — sinh stem + key answers.
    Batch mode: sinh nhiều câu hỏi cùng lúc.
    """
    topic_id   = topic_entry["topic_id"]
    topic_name = topic_entry["topic_name"]
    difficulty = topic_entry.get("difficulty", "G2")
    num_q      = topic_entry.get("num_questions", 3)

    # Tính mix câu hỏi
    num_single = int(num_q * Config.SINGLE_CORRECT_RATIO)
    num_multi  = num_q - num_single

    # Đếm single_correct: đúng 1 đáp án
    num_two_correct   = int(num_multi * 0.5)
    num_three_correct = num_multi - num_two_correct

    context_str = format_context(retrieved_blocks)
    if not context_str.strip():
        print(f"  ⚠️  No context blocks for {topic_id} — skip")
        return []

    results = []
    for seq in range(num_q):
        # Gán kiểu câu hỏi cho câu này
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
        outputs = llm.chat(messages, sampling_params=sampling)
        raw = outputs[0].outputs[0].text

        parsed = parse_json_output(raw)
        if "error" in parsed:
            print(f"  ⚠️  Parse error for {topic_id} seq={seq}: {parsed['error']}")
            continue

        # Gắn metadata
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


def run_p1_gen_stem():
    """
    Entry point cho Step 03 — chạy P1 cho tất cả topics.
    """
    config.makedirs()

    # Load topic list
    topics = load_topic_list()
    # Flatten thành list entries
    topic_entries = []
    for ch in topics:
        for t in ch.get("topics", []):
            t["chapter_id"] = ch["chapter_id"]
            t["chapter_name"] = ch["chapter_name"]
            topic_entries.append(t)

    # Khởi tạo vLLM generation model
    print("🔄 Loading Qwen2.5-14B-Instruct (generation)...")
    llm, SamplingParams = init_vllm_gen()
    print("✅ Model loaded")

    all_results = []
    for entry in topic_entries:
        topic_id = entry["topic_id"]

        # Đọc retrieval results từ Step 02
        retrieve_file = Config.RETRIEVE_OUTPUT / f"{topic_id}.jsonl"
        if not retrieve_file.exists():
            print(f"⚠️  Retrieval file not found: {retrieve_file} — skip {topic_id}")
            continue

        retrieved = load_jsonl(retrieve_file)
        context_str = retrieved[0]["context_blocks_str"] if retrieved else ""
        blocks = retrieved[0].get("context_blocks", []) if retrieved else []

        try:
            results = run_p1_for_topic(entry, blocks, llm, SamplingParams)
            all_results.extend(results)
        except Exception as e:
            print(f"❌ Error for {topic_id}: {e}")
            traceback.print_exc()

    # Save results
    out_file = Config.GEN_STEM_OUTPUT / "all_p1_results.jsonl"
    save_jsonl(all_results, out_file)
    print(f"\n✅ P1 done. Total: {len(all_results)} stems → {out_file}")

    # ── Release VRAM before next pipeline step ──
    import gc, torch
    del llm
    del SamplingParams
    gc.collect()
    torch.cuda.empty_cache()
    print("  [cleanup] VRAM released")


if __name__ == "__main__":
    run_p1_gen_stem()
