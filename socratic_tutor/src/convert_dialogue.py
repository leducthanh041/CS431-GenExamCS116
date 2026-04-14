#!/usr/bin/env python3
"""
convert_dialogue.py — Convert raw Socratic dialogue JSON → Chat format for LoRA training.

Input:  socratic_tutor/dialogue/tuan*.json
        Format: [{"question_id": "...", "dialogue": [{"role": "Tutor"|"Student", "text": "..."}]}]
Output: socratic_tutor/data/formatted/train_formatted.jsonl
        Format: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

Conversion rules (from socratic_tutor_pipeline_finetune.md):
  - System: Socratic tutor behavior prompt
  - User:   Extract misconception from FIRST student turn + optional topic context
  - Assistant: Gộp toàn bộ dialogue thành 1 chuỗi assistant, giữ nguyên thứ tự
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Any

# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Bạn là một Gia sư Socratic. Bạn KHÔNG BAO GIỜ đưa đáp án trực tiếp. "
    "Nhiệm vụ của bạn là dẫn dắt học sinh tự khám phá lỗi sai của mình bằng cách đặt "
    "câu hỏi gợi mở, khai thác đúng misconception, và hướng học sinh tự sửa lỗi. "
    "Luôn duy trì thái độ kiên nhẫn, tôn trọng, và khuyến khích tư duy phản biện."
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_misconception(dialogue: list[dict]) -> str:
    """
    Extract misconception from the student's FIRST turn.
    The tutor's opening question usually frames the misconception.
    """
    student_turns = [t["text"] for t in dialogue if t["role"] == "Student"]
    if not student_turns:
        return "[No student response found]"
    return student_turns[0]


def build_assistant_content(dialogue: list[dict]) -> str:
    """
    Gộp toàn bộ dialogue thành 1 chuỗi assistant.
    Format: "Tutor: ...\nStudent: ...\nTutor: ...\nStudent: ..."
    Giữ nguyên thứ tự, không rút gọn.
    """
    lines = []
    for turn in dialogue:
        role = "Gia sư" if turn["role"] == "Tutor" else "Học sinh"
        lines.append(f"{role}: {turn['text']}")
    return "\n".join(lines)


def convert_sample(item: dict[str, Any]) -> dict[str, Any] | None:
    """
    Convert a single dialogue item to chat format.
    Returns None if validation fails.
    """
    dialogue = item.get("dialogue", [])
    if not dialogue:
        return None

    # Must start with Tutor (opening question frames the misconception)
    if dialogue[0]["role"] != "Tutor":
        return None

    # Must have at least 4 turns (2 Tutor + 2 Student minimum for meaningful dialogue)
    if len(dialogue) < 4:
        return None

    misconception = extract_misconception(dialogue)
    assistant_content = build_assistant_content(dialogue)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Học sinh có quan niệm sai về một khái niệm lập trình. Hãy dẫn dắt họ tự nhận ra lỗi sai.\n\nQuan niệm của học sinh: {misconception}"},
            {"role": "assistant", "content": assistant_content},
        ],
        "meta": {
            "question_id": item.get("question_id", "unknown"),
        },
    }


def convert_dialogues(
    dialogue_dir: Path,
    output_path: Path,
    min_turns: int = 4,
    verbose: bool = True,
) -> dict[str, int]:
    """
    Convert all dialogue JSON files in `dialogue_dir` to training format.
    Returns stats dict.
    """
    stats = {"total": 0, "converted": 0, "skipped": 0, "errors": 0}

    json_files = sorted(dialogue_dir.glob("tuan*.json"))
    if not json_files:
        print(f"[WARN] No tuan*.json files found in {dialogue_dir}")
        return stats

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for json_file in json_files:
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to read {json_file}: {e}")
            stats["errors"] += 1
            continue

        if not isinstance(data, list):
            print(f"[WARN] {json_file.name} is not a list, skipping")
            stats["errors"] += 1
            continue

        for item in data:
            stats["total"] += 1
            if len(item.get("dialogue", [])) < min_turns:
                stats["skipped"] += 1
                continue

            converted = convert_sample(item)
            if converted is None:
                stats["skipped"] += 1
                continue

            results.append(converted)
            stats["converted"] += 1

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if verbose:
        total = stats["total"]
        converted = stats["converted"]
        pct = converted / total * 100 if total > 0 else 0
        print(f"\n{'='*60}")
        print(f"  Dialogue Conversion Summary")
        print(f"{'='*60}")
        print(f"  Files processed : {len(json_files)}")
        print(f"  Total samples   : {total}")
        print(f"  Converted       : {converted}  ({pct:.1f}%)")
        print(f"  Skipped         : {stats['skipped']}")
        print(f"  Errors          : {stats['errors']}")
        print(f"  Output          : {output_path}")
        print(f"{'='*60}\n")

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert raw Socratic dialogue JSON → Chat format for LoRA training."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path(__file__).parent.parent / "dialogue",
        help="Directory containing tuan*.json dialogue files",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "formatted" / "train_formatted.jsonl",
        help="Output .jsonl file path",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=4,
        help="Minimum number of dialogue turns to keep (default: 4)",
    )
    args = parser.parse_args()

    print(f"[INFO] Input   : {args.input}")
    print(f"[INFO] Output  : {args.output}")
    print(f"[INFO] MinTurns: {args.min_turns}")

    stats = convert_dialogues(args.input, args.output, min_turns=args.min_turns)
    sys.exit(0 if stats["converted"] > 0 else 1)


if __name__ == "__main__":
    main()
