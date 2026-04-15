"""
p4_candidates.py — Step 05: Generate Distractor Candidates (P4)
MCQGen Pipeline: refined stem → P4 → 6 distractor candidates

Input:
  - data/intermediate/04_gen_refine/all_refined_results.jsonl

Output:
  - data/intermediate/05_gen_distractors/all_candidates_results.jsonl
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (
    Config, config,
    build_p4_option_candidates,
    parse_json_output,
    load_jsonl, save_jsonl,
    init_vllm_gen, make_vllm_sampling,
)

# ── Override EXP_NAME from environment ────────────────────────────────────────
_exp_name = os.environ.get("EXP_NAME", "")
if _exp_name:
    Config.EXP_NAME = _exp_name
    Config.OUTPUT_DIR = Config.PROJECT_ROOT / "output" / Config.EXP_NAME
    Config.GEN_STEM_OUTPUT = Config.OUTPUT_DIR / "03_gen_stem"
    Config.GEN_REFINE_OUTPUT = Config.OUTPUT_DIR / "04_gen_refine"
    Config.GEN_DISTR_OUTPUT = Config.OUTPUT_DIR / "05_gen_distractors"
    Config.GEN_COT_OUTPUT = Config.OUTPUT_DIR / "06_gen_cot"
    Config.EVAL_OUTPUT = Config.OUTPUT_DIR / "07_eval"
    Config.EVAL_IWF_OUTPUT = Config.OUTPUT_DIR / "08_eval_iwf"
    print(f"[p4_candidates] EXP_NAME overridden: {Config.EXP_NAME}")


def run_p4_for_item(
    refined_item: dict,
    llm,
    SamplingParams,
) -> dict | None:
    """Chạy P4 cho 1 refined stem — sinh 6 distractor candidates."""
    topic_id  = refined_item.get("_meta", {}).get("topic_id", "unknown")
    seq       = refined_item.get("_meta", {}).get("seq", 0)

    prompt = build_p4_option_candidates(
        refined_stem_key_json=refined_item,
        num_candidates=Config.NUM_CANDIDATE_DISTRACTORS,
    )

    messages = [{"role": "user", "content": prompt}]
    sampling = SamplingParams(
        **make_vllm_sampling(Config.GEN_TEMPERATURE, Config.GEN_MAX_TOKENS),
    )
    raw = llm.chat(messages, sampling_params=sampling)[0].outputs[0].text
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


def run_p4_candidates():
    """
    Entry point cho Step 05 — sinh distractor candidates cho tất cả refined stems.
    """
    config.makedirs()

    refine_file = Config.GEN_REFINE_OUTPUT / "all_refined_results.jsonl"
    if not refine_file.exists():
        print(f"❌ Refine results not found: {refine_file}")
        print("   Chạy script 04_gen_refine.sh trước.")
        sys.exit(1)

    refined_items = load_jsonl(refine_file)
    print(f"📂 Loaded {len(refined_items)} refined items")

    print("🔄 Loading Qwen2.5-14B-Instruct (distractor candidates)...")
    llm, SamplingParams = init_vllm_gen()
    print("✅ Model loaded")

    all_results = []
    for item in refined_items:
        try:
            result = run_p4_for_item(item, llm, SamplingParams)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            traceback.print_exc()

    out_file = Config.GEN_DISTR_OUTPUT / "all_candidates_results.jsonl"
    save_jsonl(all_results, out_file)
    print(f"\n✅ P4 done. Total: {len(all_results)} → {out_file}")

    # ── Release VRAM before next pipeline step ──
    import gc, torch
    del llm
    del SamplingParams
    gc.collect()
    torch.cuda.empty_cache()
    print("  [cleanup] VRAM released")


if __name__ == "__main__":
    run_p4_candidates()
