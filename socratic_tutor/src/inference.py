#!/usr/bin/env python3
"""
inference.py — Interactive inference script for Socratic Tutor after fine-tuning.

Test hành vi Socratic của model:
  - Tutor không đưa đáp án trực tiếp
  - Tutor chỉ dùng câu hỏi dẫn dắt
  - Hội thoại multi-turn tự nhiên

Usage:
  # Interactive mode
  python src/inference.py --model outputs/merged/run_001

  # Single prompt
  python src/inference.py --model outputs/merged/run_001 \
    --prompt "Học sinh tin rằng '5.5' là số thực vì có dấu chấm."

  # Multi-turn interactive
  python src/inference.py --model outputs/merged/run_001 --interactive

  # Compare base vs fine-tuned
  python src/inference.py --model outputs/merged/run_001 \
    --compare-with /datastore/.../models/Qwen2.5-14B-Instruct
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
    pipeline,
)


# ── Prompt templates ──────────────────────────────────────────────────────────

SOCRATIC_SYSTEM = (
    "Bạn là một Gia sư Socratic. Bạn KHÔNG BAO GIỜ đưa đáp án trực tiếp. "
    "Nhiệm vụ của bạn là dẫn dắt học sinh tự khám phá lỗi sai bằng cách đặt "
    "câu hỏi gợi mở, khai thác misconception, và hướng học sinh tự sửa lỗi. "
    "Luôn kiên nhẫn, tôn trọng, và khuyến khích tư duy phản biện."
)

DEFAULT_MERGED = Path(__file__).parent.parent / "outputs" / "merged" / "run_001"
DEFAULT_BASE = "/datastore/uittogether/LuuTru/Thanhld/CS431MCQGen_copy_2/models/Qwen2.5-14B-Instruct"


# ── Evaluation helpers ────────────────────────────────────────────────────────

DIRECT_ANSWER_PATTERNS = [
    re.compile(r"đáp\s*án\s*(là|chính\s*xác)", re.IGNORECASE),
    re.compile(r"^chính\s*xác\s*[,.\s]", re.IGNORECASE),
    re.compile(r"bạn\s*nên\s+(chọn|dùng)\s+\w+\s*\.", re.IGNORECASE),
    re.compile(r"câu\s*trả\s*lời\s*đúng\s+\w+", re.IGNORECASE),
    re.compile(r"là\s+ma\s*trận|là\s+mảng|là\s+chuỗi", re.IGNORECASE),
    re.compile(r"đúng\s*(là|vậy)\s+\w{2,}\s*\.", re.IGNORECASE),
]


def analyze_output(text: str) -> dict[str, Any]:
    """Analyze model output for Socratic behavior quality."""
    lines = text.strip().split("\n")
    question_count = sum(1 for l in lines if "?" in l)
    tutor_count = sum(1 for l in lines if l.startswith("Gia sư"))
    student_count = sum(1 for l in lines if l.startswith("Học sinh"))

    has_direct = any(p.search(text) for p in DIRECT_ANSWER_PATTERNS)
    words = text.split()
    avg_line_len = sum(len(l) for l in lines) / max(len(lines), 1)

    return {
        "question_count": question_count,
        "tutor_turns": tutor_count,
        "student_turns": student_count,
        "has_direct_answer": has_direct,
        "word_count": len(words),
        "avg_line_length": round(avg_line_len, 1),
    }


def check_socratic_quality(analysis: dict) -> tuple[str, str]:
    """Return (grade, reason) based on analysis."""
    score = 0
    reasons = []

    if analysis["has_direct_answer"]:
        score -= 2
        reasons.append("⚠️ Model đưa đáp án trực tiếp")
    else:
        score += 2
        reasons.append("✅ Không đưa đáp án trực tiếp")

    if analysis["question_count"] >= 2:
        score += 2
        reasons.append(f"✅ Đặt {analysis['question_count']} câu hỏi dẫn dắt")
    elif analysis["question_count"] == 1:
        score += 1
        reasons.append(f"⚠️ Chỉ 1 câu hỏi — nên có thêm")
    else:
        score -= 1
        reasons.append("❌ Không đặt câu hỏi nào")

    if analysis["tutor_turns"] >= 3:
        score += 1
        reasons.append(f"✅ {analysis['tutor_turns']} lượt tutor — đủ dài")
    elif analysis["tutor_turns"] >= 1:
        reasons.append(f"⚠️ Chỉ {analysis['tutor_turns']} lượt tutor")
    else:
        reasons.append("❌ Không có lượt tutor")

    # Grade
    if score >= 4:
        grade = "🟢 EXCELLENT — Hành vi Socratic rõ ràng"
    elif score >= 2:
        grade = "🟡 GOOD — Hành vi Socratic khá tốt"
    elif score >= 0:
        grade = "🟠 NEEDS_IMPROVEMENT — Cần cải thiện"
    else:
        grade = "🔴 FAIL — Vi phạm quy tắc Socratic"

    return grade, "\n  ".join(reasons)


def run_single_inference(
    model_path: Path,
    misconception: str,
    system_prompt: str = SOCRATIC_SYSTEM,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    do_print: bool = True,
) -> tuple[str, dict]:
    """Run single-inference with Socratic prompt."""
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        use_fast=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Học sinh có quan niệm sai: {misconception}\n"
                "Hãy dẫn dắt họ tự nhận ra lỗi sai bằng cách đặt câu hỏi gợi mở."
            ),
        },
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    if do_print:
        print(f"\n{'─'*60}")
        print(f"  Misconception: {misconception}")
        print(f"{'─'*60}")
        print(f"\n  Model Output:")
        print(f"  {response}")
        print(f"{'─'*60}")

    analysis = analyze_output(response)
    grade, reasons = check_socratic_quality(analysis)

    if do_print:
        print(f"\n  Analysis:")
        print(f"  {reasons}")
        print(f"\n  Grade: {grade}")

    return response, analysis


def interactive_mode(model_path: Path):
    """Run interactive multi-turn Socratic dialogue."""
    print("\n" + "="*60)
    print("  Socratic Tutor — Interactive Mode")
    print("  Gõ 'quit' để thoát, 'reset' để bắt đầu lại")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    history: list[dict] = []

    while True:
        try:
            misconception = input("\n👤 Bạn (quan niệm sai): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if misconception.lower() in ("quit", "exit", "q"):
            break
        if misconception.lower() == "reset":
            history = []
            print("  ↺ Đã reset cuộc hội thoại.\n")
            continue
        if not misconception:
            continue

        messages = [
            {"role": "system", "content": SOCRATIC_SYSTEM},
        ]
        if history:
            messages.extend(history)
        messages.append({
            "role": "user",
            "content": (
                f"Học sinh có quan niệm sai: {misconception}\n"
                "Hãy dẫn dắt họ tự nhận ra lỗi sai."
            ),
        })

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=384,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        print(f"\n🎓 Gia sư Socratic:")
        print(f"  {response.strip()}")

        # Add to history
        history.append({"role": "user", "content": misconception})
        history.append({"role": "assistant", "content": response.strip()})


def compare_models(
    fine_tuned_path: Path,
    base_path: Path,
    misconception: str,
    max_new_tokens: int = 384,
) -> None:
    """Compare base model vs fine-tuned model on same prompt."""
    print(f"\n{'='*60}")
    print(f"  Comparing: Base vs Fine-tuned")
    print(f"{'='*60}")

    for label, path in [("BASE MODEL", base_path), ("FINE-TUNED", fine_tuned_path)]:
        print(f"\n  ── {label} ──")
        try:
            run_single_inference(
                model_path=path,
                misconception=misconception,
                max_new_tokens=max_new_tokens,
                do_print=True,
            )
        except Exception as e:
            print(f"  [ERROR] {e}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inference for Socratic Tutor (Qwen LoRA).")
    parser.add_argument(
        "--model", "-m",
        type=Path,
        default=DEFAULT_MERGED,
        help="Path to fine-tuned/merged model",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(DEFAULT_BASE),
        help="Path to base model (for --compare)",
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="Single misconception prompt to test",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run interactive multi-turn dialogue mode",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare base model vs fine-tuned model",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max new tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    if args.interactive:
        interactive_mode(args.model)
        return

    misconception = args.prompt or (
        "Học sinh tin rằng '5.5' là số thực vì có dấu chấm thập phân, "
        "trong khi đó dấu nháy trong Python biến nó thành string."
    )

    if args.compare:
        compare_models(args.model, args.base, misconception, args.max_tokens)
        return

    response, analysis = run_single_inference(
        model_path=args.model,
        misconception=misconception,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    grade, reasons = check_socratic_quality(analysis)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({
                "model": str(args.model),
                "misconception": misconception,
                "response": response,
                "analysis": analysis,
                "grade": grade,
                "reasons": reasons,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n  💾 Results saved: {args.output_json}")


if __name__ == "__main__":
    main()
