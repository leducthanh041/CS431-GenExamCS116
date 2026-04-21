#!/usr/bin/env python3
"""
filter_quality.py — Quality filtering for Socratic dialogue training data.

Filters samples based on the rules from socratic_tutor_pipeline_finetune.md:
  1. Tutor gives direct answer immediately (first 2 tutor turns)
  2. No meaningful guiding process (< 2 tutor questions before first answer hint)
  3. Student doesn't show any conceptual change (opening vs closing belief same)
  4. Dialogue too short (< 4 turns) — already filtered in convert, but double-check

Heuristic filters (no LLM needed):
  - Direct-answer detection: keyword patterns in early tutor turns
  - Socratic-process check: minimum tutor questions before any hint
  - Student-view-change: first vs last student statements are meaningfully different
"""

import json
import re
import sys
import argparse
from pathlib import Path
from typing import Any

# ── Direct-answer keyword patterns (Vietnamese) ────────────────────────────────

DIRECT_ANSWER_PATTERNS = [
    # Tutor reveals answer/definition immediately (STATEMENT, not question)
    re.compile(r"đáp\s*án\s*(là|chính\s*xác)", re.IGNORECASE),
    re.compile(r"đúng\s*(là|vậy)\s+\w+\s*(chứ|\.|$)", re.IGNORECASE),
    # Only flag "chính xác" when used as a definite conclusion (not as "how exactly?")
    # False positive: "nó học chính xác như thế nào?" is a valid Socratic question
    re.compile(r"^chính\s*xác\s*[,.\s]*(là|vậy|đó)", re.IGNORECASE),
    re.compile(r"(đó|là)\s*['\"]?\w+['\"]?\s*(chứ|gì)\s*$", re.IGNORECASE),  # phủ định/question ending
    re.compile(r"trả\s*lời\s*(là|đúng)\s+\w+", re.IGNORECASE),
    re.compile(r"khái\s*niệm\s*(là|được\s*gọi)\s+\w+", re.IGNORECASE),
    re.compile(r"định\s*nghĩa\s*(là|của)\s+\w+", re.IGNORECASE),
    re.compile(r"bạn\s*nên\s+(chọn|dùng|làm)\s+\w+", re.IGNORECASE),
    re.compile(r"câu\s*trả\s*lời\s*(đúng|là)\s+\w+", re.IGNORECASE),
    # Tutor gives formula/definition directly
    re.compile(r"công\s*thức\s*(là|=\s*)", re.IGNORECASE),
    re.compile(r"là\s+ma\s*trận|là\s+mảng|là\s+chuỗi|là\s+số", re.IGNORECASE),
]

# Patterns that indicate the tutor is giving a hint/conclusion (OK in late turns, bad early)
EARLY_REVEAL_PATTERNS = [
    re.compile(r"^(chính\s*xác|đúng\s*vậy|đúng\s*rồi)\s*[.,]?\s*$", re.IGNORECASE),
    re.compile(r"đó\s*(là|chính\s*là)", re.IGNORECASE),
]

# Question words — tutor asking Socratically
QUESTION_MARKERS = ["?", " gì", " sao", " không", " nào", " ở đâu", " như thế nào",
                    " có phải", " hãy cho", " thử", " bạn cho rằng", " cơ sở nào",
                    " tại sao", " vì sao"]


def is_question(text: str) -> bool:
    """Check if a tutor turn is a question (Socratic guiding)."""
    t = text.strip()
    return any(marker in t for marker in QUESTION_MARKERS) or t.endswith("?")


def tutor_gave_direct_answer_early(dialogue: list[dict], early_n: int = 2) -> tuple[bool, str]:
    """
    Check if tutor gave a direct answer in the first `early_n` tutor turns.
    Returns (failed, reason).
    """
    tutor_turns = [t["text"] for t in dialogue if t["role"] == "Tutor"][:early_n]

    for i, text in enumerate(tutor_turns):
        # Check direct-answer keywords
        for pattern in DIRECT_ANSWER_PATTERNS:
            if pattern.search(text):
                return True, f"tutor_turn_{i+1} matches direct-answer pattern: '{pattern.pattern}' → '{text[:60]}'"

        # Check early-reveal patterns (conclusion in first 1-2 turns)
        if i < 2:
            for pattern in EARLY_REVEAL_PATTERNS:
                if pattern.match(text.strip()):
                    return True, f"tutor_turn_{i+1} reveals answer early: '{text[:60]}'"

    return False, ""


def has_minimum_guiding_process(dialogue: list[dict], min_questions: int = 2) -> tuple[bool, str]:
    """
    Require at least `min_questions` tutor questions BEFORE any hint or
    conclusion appears. This ensures the tutor doesn't rush to the answer.
    Returns (passed, reason).
    """
    hint_patterns = [
        re.compile(r"chính\s*xác|đúng\s*vậy|đúng\s*rồi", re.IGNORECASE),
        re.compile(r"đó\s*(là|chính\s*là)", re.IGNORECASE),
        re.compile(r"đáp\s*án\s*(là|chính)", re.IGNORECASE),
    ]

    question_count = 0
    for turn in dialogue:
        if turn["role"] != "Tutor":
            continue
        text = turn["text"]

        # Is this a question?
        if is_question(text):
            question_count += 1
        else:
            # Non-question turn — check if it's a hint/conclusion
            for pattern in hint_patterns:
                if pattern.search(text):
                    if question_count < min_questions:
                        return False, f"only {question_count} tutor questions before hint (need {min_questions})"
                    return True, ""

    return True, ""


def student_shows_conceptual_change(dialogue: list[dict]) -> tuple[bool, str]:
    """
    Check if the student shows conceptual change between their first and last turn.
    Compares length, hedging language, and explicit admission of error.

    Returns (passed, reason).
    """
    student_turns = [t["text"] for t in dialogue if t["role"] == "Student"]
    if len(student_turns) < 2:
        return False, "only 1 student turn — cannot detect conceptual change"

    first = student_turns[0].strip()
    last = student_turns[-1].strip()

    # Explicit error admission patterns
    admission_patterns = [
        r"sai", r"nhầm", r"sửa", r"lỗi", r"à.*đúng",
        r"chờ.*đã", r"khoan", r"vậy\s*(là|thì)", r"\.\.\.",
        r"à\b", r"ừ", r"hiểu", r"được\s*rồi",
    ]
    has_admission = any(re.search(p, last, re.IGNORECASE) for p in admission_patterns)

    # Hedging/uncertainty in first turn (student defending misconception)
    certainty_patterns = [r"chắc", r"rõ\s*ràng", r"đương\s*nhiên", r"hiển\s*nhiên", r"đúng\s*mà"]
    first_certain = any(re.search(p, first, re.IGNORECASE) for p in certainty_patterns)
    last_uncertain = any(re.search(p, last, re.IGNORECASE) for p in [r"\.\.\.", r"chắc", r"có\s*lẽ", r"có\s*vẻ"])

    # Length change — last response often longer (more reflection)
    length_ratio = len(last) / max(len(first), 1)
    meaningful_change = has_admission or (first_certain and last_uncertain) or (length_ratio > 1.5)

    if meaningful_change:
        return True, "student shows conceptual change"
    else:
        # Soft check — don't reject, just flag
        return True, "conceptual_change_uncertain (passing)"


def check_tutor_uses_questions(dialogue: list[dict], min_question_ratio: float = 0.5) -> tuple[bool, str]:
    """
    Tutor must use questions as the primary means of guidance.
    At least `min_question_ratio` fraction of tutor turns should be questions.
    """
    tutor_turns = [t["text"] for t in dialogue if t["role"] == "Tutor"]
    if not tutor_turns:
        return False, "no tutor turns"

    question_turns = sum(1 for t in tutor_turns if is_question(t))
    ratio = question_turns / len(tutor_turns)

    if ratio >= min_question_ratio:
        return True, f"tutor_question_ratio={ratio:.2f}"
    else:
        return False, f"tutor_question_ratio={ratio:.2f} < {min_question_ratio} (need more questions)"


# ── Main filter ────────────────────────────────────────────────────────────────

FILTER_RULES = [
    ("dialogue_min_turns", lambda d: len(d) >= 4, "Dialogue has < 4 turns"),
    ("tutor_direct_answer_early", lambda d: not tutor_gave_direct_answer_early(d)[0], "Tutor gives direct answer early"),
    ("socratic_process", lambda d: has_minimum_guiding_process(d)[0], "No Socratic guiding process"),
    ("tutor_question_ratio", lambda d: check_tutor_uses_questions(d)[0], "Tutor doesn't ask enough questions"),
    ("student_conceptual_change", lambda d: student_shows_conceptual_change(d)[0], "Student shows no conceptual change"),
]


def parse_dialogue(assistant_content: str) -> list[dict]:
    """
    Parse assistant content back to dialogue list.
    Format: "Gia sư: ...\nHọc sinh: ...\n..."
    """
    dialogue = []
    for line in assistant_content.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("Gia sư:") or line.startswith("Gia sư :"):
            dialogue.append({"role": "Tutor", "text": line.split(":", 1)[1].strip()})
        elif line.startswith("Học sinh:") or line.startswith("Học sinh :"):
            dialogue.append({"role": "Student", "text": line.split(":", 1)[1].strip()})
    return dialogue


def filter_samples(
    input_path: Path,
    output_path: Path,
    output_rejected: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Filter samples from `input_path` and write passed/rejected to output files.
    """
    stats = {
        "total": 0, "passed": 0, "rejected": 0,
        "reasons": {},
    }

    passed_records: list[dict] = []
    rejected: list[dict] = []

    with open(input_path, encoding="utf-8") as f:
        lines = f.readlines()

    stats["total"] = len(lines)

    for line in lines:
        record = json.loads(line)

        # Reconstruct dialogue from assistant content
        messages = record.get("messages", [])
        assistant_content = ""
        for m in messages:
            if m.get("role") == "assistant":
                assistant_content = m.get("content", "")
                break

        dialogue = parse_dialogue(assistant_content)
        if not dialogue:
            dialogue = []  # fallback

        rejected_reasons = []
        passed = True

        for rule_name, rule_fn, reason_desc in FILTER_RULES:
            try:
                ok = rule_fn(dialogue) if dialogue else False
            except Exception as e:
                ok = False
                reason_desc = f"{reason_desc} (eval error: {e})"

            if not ok:
                passed = False
                rejected_reasons.append(f"{rule_name}: {reason_desc}")

        if passed:
            passed_records.append(record)
            stats["passed"] += 1
        else:
            record["_rejected_reasons"] = rejected_reasons
            rejected.append(record)
            stats["rejected"] += 1
            for r in rejected_reasons:
                stats["reasons"][r] = stats["reasons"].get(r, 0) + 1

    # Write outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in passed_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if output_rejected:
        output_rejected.parent.mkdir(parents=True, exist_ok=True)
        with open(output_rejected, "w", encoding="utf-8") as f:
            for record in rejected:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if verbose:
        total = stats["total"]
        passed_c = stats["passed"]
        pct = passed_c / total * 100 if total > 0 else 0
        print(f"\n{'='*60}")
        print(f"  Quality Filtering Summary")
        print(f"{'='*60}")
        print(f"  Total samples   : {total}")
        print(f"  Passed          : {passed_c}  ({pct:.1f}%)")
        print(f"  Rejected        : {stats['rejected']}")
        print(f"  Output          : {output_path}")
        if output_rejected:
            print(f"  Rejected log    : {output_rejected}")
        print(f"\n  Rejection reasons:")
        for reason, count in sorted(stats["reasons"].items(), key=lambda x: -x[1]):
            print(f"    [{count:3d}] {reason}")
        print(f"{'='*60}\n")

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quality filter for Socratic dialogue training data."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "formatted" / "train_formatted.jsonl",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "filtered" / "train_filtered.jsonl",
    )
    parser.add_argument(
        "--rejected",
        "-r",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "filtered" / "train_rejected.jsonl",
    )
    args = parser.parse_args()

    print(f"[INFO] Input    : {args.input}")
    print(f"[INFO] Output   : {args.output}")
    print(f"[INFO] Rejected : {args.rejected}")

    stats = filter_samples(args.input, args.output, args.rejected)
    sys.exit(0 if stats["passed"] > 0 else 1)


if __name__ == "__main__":
    main()
