"""
prompt_config.py — MCQGen Generation Configuration Loader
=========================================================
Load configs/generation_config.yaml, merge with topic_list.json,
and compute per-topic question distribution.

Usage:
    from src.gen.prompt_config import load_generation_config, distribute_questions

    cfg = load_generation_config()
    dist = distribute_questions(cfg["generation"]["target_range"][1], cfg)
    # dist["ch07b_t01"] = 2, dist["ch02_t01"] = 1, ...
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

# NOTE: Must add parent dir (src/) to path so 'from common' works
import sys as _sys
from pathlib import Path as _Path
_pdir = str(_Path(__file__).resolve().parent.parent)  # .../CS431MCQGen/src/
if _pdir not in _sys.path:
    _sys.path.insert(0, _pdir)

# Now import after path is set
from common import Config, config

PROJECT_ROOT = Config.PROJECT_ROOT  # resolves to CS431MCQGen/ correctly


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "generation": {
        "target_range": [25, 35],
        "single_correct_ratio": 0.80,
        "two_correct_ratio": 0.60,
        "three_correct_ratio": 0.40,
        "topic_weights": {},
        "focus_topics": [],
        "focus_chapters": [],
    },
    "diversity": {
        "avoid_openings": [
            "Hãy xác định", "khi", "đâu",
            "Trong quá trình", "Khi xây dựng", "Khi huấn luyện",
            "Trong các phương pháp", "Trong các kỹ thuật", "Cho biết",
        ],
        "prefer_openings": [
            "Điều gì khiến", "Đâu là điểm khác biệt giữa",
            "Trường hợp nào sau đây minh họa đúng nhất về",
            "Điều kiện tiên quyết để... hoạt động hiệu quả là gì?",
            "Sau khi áp dụng..., kết quả mong đợi là",
            "Quan sát đoạn code sau, output nào phù hợp nhất?",
            "Nếu thay đổi tham số... thì điều gì sẽ xảy ra với",
            "Vai trò chính của... trong kiến trúc này là",
            "Nhận định nào sau đây là chính xác nhất về",
            "Mục đích chính của... là gì?",
            "Tính chất nào giúp phân biệt... với",
        ],
    },
    "explanation": {
        "include_web_sources": True,
        "max_web_sources": 3,
        "explanation_context_blocks": 5,
        "llm_model": "gpt-4o",
    },
    "retrieval": {
        "use_bm25": True,
        "use_rerank": True,
        "rerank_top_n": 30,
        "context_blocks_for_prompt": 5,
        "min_similarity": 0.30,
    },
    # Cache-busting: increment this when writing new config to force reload
    "_version": 1,
}


# ── Load generation config ────────────────────────────────────────────────────

def load_generation_config(
    config_path: Path | str | None = None,
    use_active: bool = True,
) -> dict[str, Any]:
    """
    Load generation config from YAML, merge with defaults.

    Priority (if use_active=True):
      1. configs/generation_config_active.yaml  (written by web app or CLI override)
      2. configs/generation_config.yaml        (user default)
      3. Built-in defaults

    Args:
        config_path: Explicit path to a YAML file (overrides active/default)
        use_active: If True, check for active config first
    """
    # Explicit path overrides everything
    if config_path is None and use_active:
        # Check for active config from web app / CLI
        active_path = PROJECT_ROOT / "configs" / "generation_config_active.yaml"
        if active_path.exists():
            config_path = active_path

    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "generation_config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        print(f"⚠️  Config not found: {config_path} — using defaults")
        return DEFAULT_CONFIG.copy()

    with open(config_path, encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f)

    # Deep merge with defaults
    merged = _deep_merge(DEFAULT_CONFIG, user_cfg or {})
    merged["_loaded_from"] = str(config_path)
    return merged


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ── Load topic list ──────────────────────────────────────────────────────────

def load_topic_list_with_config(config: dict[str, Any]) -> list[dict]:
    """
    Load topic_list.json, apply focus_chapters / focus_topics filter,
    and attach per-topic num_questions based on weights.
    """
    topic_path = Config.TOPIC_LIST_FILE
    if not topic_path.exists():
        raise FileNotFoundError(f"topic_list.json not found: {topic_path}")

    with open(topic_path, encoding="utf-8") as f:
        raw = json.load(f)

    focus_chapters = config["generation"].get("focus_chapters", [])
    focus_topics   = config["generation"].get("focus_topics", [])
    weights        = config["generation"].get("topic_weights", {})
    target_range   = config["generation"]["target_range"]
    total_target   = target_range[1]  # upper bound

    # Compute per-topic weights
    topic_weights = _build_topic_weights(raw, weights)

    # Filter if focus is set
    if focus_topics:
        filtered = []
        for ch in raw:
            ch_topics = [t for t in ch.get("topics", [])
                         if t["topic_id"] in focus_topics]
            if ch_topics:
                filtered.append({**ch, "topics": ch_topics})
        raw = filtered

    if focus_chapters:
        raw = [ch for ch in raw if ch["chapter_id"] in focus_chapters]

    # Distribute questions across all filtered topics
    distribution = distribute_questions(total_target, raw, topic_weights)

    # Apply distribution to topics
    all_topics = []
    for ch in raw:
        for t in ch.get("topics", []):
            t = dict(t)  # copy
            t["chapter_id"]   = ch["chapter_id"]
            t["chapter_name"] = ch["chapter_name"]
            # Override num_questions with weighted distribution
            t["num_questions"] = distribution.get(t["topic_id"], 1)
            # Attach weight for reference
            t["weight"] = topic_weights.get(t["topic_id"], 1.0)
            all_topics.append(t)

    return all_topics


def _build_topic_weights(
    chapters: list[dict],
    explicit_weights: dict[str, float],
) -> dict[str, float]:
    """Build weight map: chapter_id or topic_id → weight."""
    weights: dict[str, float] = {}
    default = 1.0

    for ch in chapters:
        ch_id = ch["chapter_id"]
        ch_w  = explicit_weights.get(ch_id, default)
        for t in ch.get("topics", []):
            t_id = t["topic_id"]
            # Topic-level weight overrides chapter weight
            weights[t_id] = explicit_weights.get(t_id, ch_w)

    return weights


# ── Distribute questions ─────────────────────────────────────────────────────

def distribute_questions(
    total_target: int,
    chapters: list[dict] | None = None,
    topic_weights: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, int]:
    """
    Weighted proportional distribution of question count across topics.

    Algorithm:
      1. Sum all weights
      2. Each topic gets floor(total × weight/total_weight)
      3. Allocate remaining slots to highest-weight topics

    Returns: {topic_id: num_questions}

    Can be called with either (chapters + weights) or (config dict).
    """
    # Load from config if not provided
    if chapters is None or topic_weights is None:
        if config is None:
            config = load_generation_config()
        chapters = load_topic_list_with_config(config)
        topic_weights = {t["topic_id"]: t["weight"] for t in chapters}

    # Build flat list: (topic_id, weight)
    topics_with_weights = []
    for ch in (chapters or []):
        for t in ch.get("topics", []):
            t_id = t["topic_id"]
            w = topic_weights.get(t_id, 1.0) if topic_weights else 1.0
            topics_with_weights.append((t_id, w))

    if not topics_with_weights:
        return {}

    total_weight = sum(w for _, w in topics_with_weights)
    if total_weight == 0:
        total_weight = 1.0

    allocated: dict[str, int] = {}
    remaining = total_target

    # First pass: proportional allocation
    for t_id, w in topics_with_weights:
        share = max(1, round(total_target * w / total_weight))
        allocated[t_id] = min(share, remaining)
        remaining -= allocated[t_id]

    # Second pass: fill remaining slots to highest-weight topics
    sorted_topics = sorted(topics_with_weights, key=lambda x: x[1], reverse=True)
    for t_id, _ in sorted_topics:
        if remaining > 0:
            allocated[t_id] += 1
            remaining -= 1
        else:
            break

    return allocated


# ── Batch context computation ───────────────────────────────────────────────

def compute_batch_context(
    num_questions: int,
    config: dict[str, Any] | None = None,
) -> dict[str, int]:
    """
    Compute per-batch mix of single vs multiple answer questions.
    Returns: {num_single, num_multi, num_two, num_three}

    Example: num_questions=5, single_correct_ratio=0.8
      → num_single=4, num_multi=1, num_two=1, num_three=0
    """
    if config is None:
        config = load_generation_config()

    ratio    = config["generation"].get("single_correct_ratio", 0.80)
    two_ratio = config["generation"].get("two_correct_ratio", 0.60)

    num_single = int(num_questions * ratio)
    num_multi  = num_questions - num_single
    num_two    = int(num_multi * two_ratio)
    num_three  = num_multi - num_two

    return {
        "num_single":   num_single,
        "num_multi":    num_multi,
        "num_two":      num_two,
        "num_three":    num_three,
        "ratio_single": num_single / num_questions if num_questions > 0 else 0,
        "ratio_multi":  num_multi  / num_questions if num_questions > 0 else 0,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test generation config")
    parser.add_argument("--config", default=None, help="Path to generation_config.yaml")
    args = parser.parse_args()

    cfg = load_generation_config(args.config)

    print("=== Generation Config ===")
    print(f"  target_range: {cfg['generation']['target_range']}")
    print(f"  single_correct_ratio: {cfg['generation']['single_correct_ratio']}")
    print(f"  topic_weights: {cfg['generation']['topic_weights']}")
    print(f"  focus_chapters: {cfg['generation']['focus_chapters']}")
    print(f"  focus_topics: {cfg['generation']['focus_topics']}")

    print("\n=== Distribution (upper bound) ===")
    dist = distribute_questions(
        total_target=cfg["generation"]["target_range"][1],
        config=cfg,
    )
    total = sum(dist.values())
    print(f"  Total questions: {total}")
    for tid, nq in sorted(dist.items()):
        print(f"    {tid}: {nq}")

    print("\n=== Batch Context (per topic with 3 questions) ===")
    bc = compute_batch_context(3, cfg)
    for k, v in bc.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
