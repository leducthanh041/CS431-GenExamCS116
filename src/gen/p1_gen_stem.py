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
import os
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
    format_context_block,
)

# ── Override EXP_NAME from environment (set by deploy_pipeline.sh) ────────────
_exp_name = os.environ.get("EXP_NAME", "")
if _exp_name:
    Config.EXP_NAME = _exp_name
    Config.OUTPUT_DIR = Config.PROJECT_ROOT / "output" / Config.EXP_NAME
    Config.RETRIEVE_OUTPUT = Config.OUTPUT_DIR / "02_retrieval"
    Config.GEN_STEM_OUTPUT = Config.OUTPUT_DIR / "03_gen_stem"
    Config.GEN_REFINE_OUTPUT = Config.OUTPUT_DIR / "04_gen_refine"
    Config.GEN_DISTR_OUTPUT = Config.OUTPUT_DIR / "05_gen_distractors"
    Config.GEN_COT_OUTPUT = Config.OUTPUT_DIR / "06_gen_cot"
    Config.EVAL_OUTPUT = Config.OUTPUT_DIR / "07_eval"
    Config.EVAL_IWF_OUTPUT = Config.OUTPUT_DIR / "08_eval_iwf"
    print(f"[p1_gen_stem] EXP_NAME overridden: {Config.EXP_NAME}")


def format_context(blocks: list[dict]) -> str:
    """Gộp context blocks thành chuỗi cho prompt, có citation metadata."""
    if not blocks:
        return ""
    lines = []
    for i, blk in enumerate(blocks, 1):
        lines.append(f"--- Context {i} ---")
        # Dùng format_context_block để có citation: [ch04] | Video | YouTube URL | Thời điểm
        ctx = format_context_block(blk)
        lines.append(ctx[:2000])  # Giới hạn độ dài mỗi block
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
    # Dùng round() thay vì int() để tránh floor bias gây lệch tỉ lệ
    # VD: num_q=1, ratio=0.8 → round(0.8)=1 (đúng), int(0.8)=0 (sai → luôn multiple)
    num_single = round(num_q * Config.SINGLE_CORRECT_RATIO)
    num_multi  = num_q - num_single
    # Ensure at least 1 single if ratio >= 0.5, else at least 1 multi
    if num_single == 0 and num_q >= 1 and Config.SINGLE_CORRECT_RATIO >= 0.5:
        num_single = 1
        num_multi  = num_q - num_single
    if num_multi == 0 and num_q >= 1 and Config.SINGLE_CORRECT_RATIO < 0.5:
        num_multi = 1
        num_single = num_q - num_multi

    # Đếm single_correct: đúng 1 đáp án
    num_two_correct   = round(num_multi * 0.5) if num_multi > 0 else 0
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

        # ── FORCE type to match what was requested ──────────────────────────────
        # The LLM sometimes ignores the HARD CONSTRAINT in the prompt.
        # Override to guarantee correct ratio.
        parsed["question_type"] = q_type
        parsed["correct_answer_count"] = correct_count
        # Also fix the correct_answers_content list length if LLM miscounted
        correct_content = parsed.get("correct_answers_content", [])
        if isinstance(correct_content, list):
            if len(correct_content) != correct_count:
                print(f"  ⚠️  Fixing answer count: {len(correct_content)} → {correct_count} for {topic_id} seq={seq}")
                if correct_count == 1 and len(correct_content) > 1:
                    parsed["correct_answers_content"] = correct_content[:1]
                elif correct_count >= 2 and len(correct_content) < correct_count:
                    # Pad with existing content
                    while len(parsed["correct_answers_content"]) < correct_count:
                        parsed["correct_answers_content"].append(correct_content[-1])

        # ── Sources: placeholder — filled at Step 09 (explanation) ──────────
        # We no longer attach YouTube/slide sources at P1 step.
        # The HybridRetriever in explain_mcq.py will attach proper citations
        # with YouTube timestamps + slide page numbers + web search results.
        parsed["sources"] = []

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
