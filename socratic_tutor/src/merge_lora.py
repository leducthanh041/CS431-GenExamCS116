#!/usr/bin/env python3
"""
merge_lora.py — Merge LoRA adapter into base model + optional evaluation.

Steps:
  1. Load base model (Qwen2.5-14B-Instruct)
  2. Load LoRA adapter from checkpoint
  3. Merge weights (float16) and save to outputs/merged/
  4. Run basic Socratic behavior evaluation on merge outputs

Usage:
  # Merge only
  python src/merge_lora.py --adapter outputs/checkpoints/run_001

  # Merge + quick eval
  python src/merge_lora.py --adapter outputs/checkpoints/run_001 --eval
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ── Paths ────────────────────────────────────────────────────────────────────

BASE_MODEL_PATH = "/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen_copy_2/models/Qwen2.5-14B-Instruct"
DEFAULT_ADAPTER = Path(__file__).parent.parent / "outputs" / "checkpoints" / "run_001"
DEFAULT_MERGED = Path(__file__).parent.parent / "outputs" / "merged" / "run_001"


# ── Helpers ─────────────────────────────────────────────────────────────────

def find_adapter_path(adapter_dir: Path) -> Path | None:
    """Find the actual adapter SAFETENSORS files inside the checkpoint dir."""
    # Try direct path
    if (adapter_dir / "adapter_model.safetensors").exists():
        return adapter_dir

    # Try to find the latest checkpoint subfolder
    checkpoints = sorted((adapter_dir / "checkpoint-*").glob("checkpoint-*"), reverse=True)
    if checkpoints:
        return checkpoints[0]

    # Fallback: use the directory itself
    return adapter_dir


def merge_and_save(
    base_model_path: Path,
    adapter_path: Path,
    output_path: Path,
    quantization: bool = False,
) -> None:
    """Load base + adapter, merge, and save merged model."""
    print(f"\n{'='*60}")
    print(f"  Merging LoRA Adapter → Merged Model")
    print(f"{'='*60}")
    print(f"  Base model  : {base_model_path}")
    print(f"  Adapter     : {adapter_path}")
    print(f"  Output      : {output_path}")
    print(f"  Quantization: {quantization}")

    output_path.mkdir(parents=True, exist_ok=True)

    # Quantization config
    bnb_config = None
    if quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    print("\n[1/4] Loading base model (FP16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    print("[2/4] Loading LoRA adapter...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    print("[3/4] Merging adapter into base model...")
    model = model.merge_and_unload()
    model = model.to(torch.float16)

    print("[4/4] Saving merged model...")
    model.save_pretrained(str(output_path), safe_serialization=True)
    print(f"  ✅ Merged model saved to {output_path}")


def load_tokenizer(model_path: Path) -> Any:
    """Load tokenizer."""
    return AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        use_fast=True,
    )


# ── Socratic behavior evaluation ─────────────────────────────────────────────

SOCRATIC_SYSTEM_PROMPT = (
    "Bạn là một Gia sư Socratic. Bạn KHÔNG BAO GIỜ đưa đáp án trực tiếp. "
    "Luôn dẫn dắt học sinh bằng câu hỏi gợi mở."
)

TEST_CASES = [
    {
        "id": "test_string_type",
        "misconception": "Học sinh tin rằng '5.5' là số thực vì có dấu chấm thập phân.",
        "expected_pattern": "Hỏi về dấu nháy / string / Python phân biệt kiểu",
    },
    {
        "id": "test_operator_precedence",
        "misconception": "Học sinh nghĩ True or True and False = False vì tính từ trái sang phải.",
        "expected_pattern": "Hỏi về thứ tự ưu tiên / and vs or",
    },
    {
        "id": "test_list_concat",
        "misconception": "Học sinh nghĩ [1,2] + [3,4] = [4,6] vì cộng từng phần tử.",
        "expected_pattern": "Hỏi về phép nối list / ghép vs cộng",
    },
    {
        "id": "test_power_operator",
        "misconception": "Học sinh nghĩ i**2 là i × 2.",
        "expected_pattern": "Hỏi về toán tử ** / lũy thừa vs nhân",
    },
    {
        "id": "test_dataframe_shape",
        "misconception": "Học sinh nghĩ shape=(2,3) và shape=(3,2) là như nhau vì cùng 6 phần tử.",
        "expected_pattern": "Hỏi về hàng / cột / thứ tự shape",
    },
]

DIRECT_ANSWER_INDICATORS = [
    r"đáp\s*án\s*(là|chính\s*xác)",
    r"đúng\s*(là|vậy)\s+\w+",
    r"^chính\s*xác\s*[,.\s]",
    r"bạn\s*nên\s+(chọn|dùng)\s+\w+",
    r"câu\s*trả\s*lời\s*đúng",
]


def run_behavior_eval(
    model_path: Path,
    output_path: Path,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Run quick Socratic behavior evaluation on merged model."""
    print(f"\n{'='*60}")
    print(f"  Socratic Behavior Evaluation")
    print(f"{'='*60}")

    from transformers import pipeline

    pipe = pipeline(
        "text-generation",
        model=str(model_path),
        tokenizer=str(model_path),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    results = []

    for tc in TEST_CASES:
        print(f"\n  [{tc['id']}] Testing...")
        messages = [
            {"role": "system", "content": SOCRATIC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Học sinh có quan niệm sai: {tc['misconception']}\nHãy dẫn dắt họ tự nhận ra lỗi sai bằng cách đặt câu hỏi.",
            },
        ]

        output = pipe(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )

        generated = output[0]["generated_text"]
        # Extract assistant part
        assistant_text = ""
        for m in reversed(generated):
            if m.get("role") == "assistant":
                assistant_text = m.get("content", "")
                break

        # Check for direct answer
        has_direct = any(re.search(p, assistant_text, re.IGNORECASE) for p in DIRECT_ANSWER_INDICATORS)
        has_question = "?" in assistant_text

        status = "PASS" if (has_question and not has_direct) else "FAIL"

        result = {
            "test_id": tc["id"],
            "expected": tc["expected_pattern"],
            "generated": assistant_text[:300],
            "has_question": has_question,
            "has_direct_answer": has_direct,
            "status": status,
        }
        results.append(result)
        print(f"    Status: {status} | Question: {has_question} | Direct: {has_direct}")
        print(f"    Preview: {assistant_text[:150]}...")

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    pct = passed / total * 100

    print(f"\n{'='*60}")
    print(f"  Evaluation Summary: {passed}/{total} ({pct:.0f}%) passed")
    print(f"{'='*60}")

    summary = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": pct,
        "results": results,
    }

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  Results saved: {output_path}")

    return summary


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base Qwen model.")
    parser.add_argument(
        "--adapter", "-a",
        type=Path,
        default=DEFAULT_ADAPTER,
        help="Path to LoRA adapter checkpoint directory",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(BASE_MODEL_PATH),
        help="Path to base model",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_MERGED,
        help="Output path for merged model",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Load base model in 4-bit NF4 (slower merge, smaller RAM)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run Socratic behavior evaluation after merge",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merge (use existing merged model for eval only)",
    )
    args = parser.parse_args()

    print(f"[INFO] Adapter : {args.adapter}")
    print(f"[INFO] Base    : {args.base}")
    print(f"[INFO] Output  : {args.output}")
    print(f"[INFO] Eval    : {args.eval}")

    adapter_path = find_adapter_path(args.adapter)
    if not adapter_path:
        print(f"[ERROR] Cannot find adapter at {args.adapter}")
        sys.exit(1)

    print(f"[INFO] Using adapter path: {adapter_path}")

    if not args.skip_merge:
        merge_and_save(
            base_model_path=args.base,
            adapter_path=adapter_path,
            output_path=args.output,
            quantization=args.quantize,
        )
    else:
        print("[INFO] Skipping merge — running eval on existing model")

    if args.eval:
        eval_path = args.output.parent / "eval" / "behavior_eval.json"
        run_behavior_eval(model_path=args.output, output_path=eval_path)


if __name__ == "__main__":
    main()
