"""
test_gen_mini.py — Gen 5-10 câu hỏi MCQ thử nghiệm để verify YouTube citation
Chỉ chạy 3 topics × 2-3 câu = ~8 câu hỏi

Submit:
  sbatch scripts/test_gen_mini.sh
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (
    Config, config,
    build_p1_gen_stem_key,
    parse_json_output,
    save_jsonl,
    init_vllm_gen, make_vllm_sampling,
    format_context_block,
)

# ─── Mini topic list: 3 topics × 2-3 câu = ~8 câu ─────────────────────────
TEST_TOPICS = [
    {"chapter_id": "ch04", "topic_id": "ch04_t01", "topic_name": "Missing Data",
     "difficulty": "G2", "num_questions": 3},
    {"chapter_id": "ch04", "topic_id": "ch04_t02", "topic_name": "Outlier Detection",
     "difficulty": "G2", "num_questions": 3},
    {"chapter_id": "ch10", "topic_id": "ch10_t01", "topic_name": "Bagging",
     "difficulty": "G2", "num_questions": 2},
]
TOTAL = sum(t["num_questions"] for t in TEST_TOPICS)
print(f"🎯 Mini test: {len(TEST_TOPICS)} topics → {TOTAL} câu hỏi")


# ─── Retrieval ────────────────────────────────────────────────────────────────

def retrieve_blocks(topic_id: str, topic_name: str, n: int = 5) -> list[dict]:
    """Query ChromaDB for top-n context blocks."""
    import chromadb
    from sentence_transformers import SentenceTransformer
    client = chromadb.PersistentClient(path=str(Config.INDEX_DIR))
    collection = client.get_or_create_collection("concept_chunks")
    # Embed query with same model used for indexing (BGE-m3 → 1024-dim)
    model = SentenceTransformer("BAAI/bge-m3")
    query_embedding = model.encode([topic_name], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=query_embedding, n_results=n)
    blocks = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        if not doc:
            continue
        blocks.append({
            "chunk_id":         meta.get("chunk_id", ""),
            "chapter_id":        meta.get("chapter_id", ""),
            "topic":            topic_name,
            "section_title":    meta.get("section_title", ""),
            "text":             doc,
            "source":           "concept_kb",
            "source_type":      meta.get("source_type", ""),
            "source_file":      meta.get("source_file", ""),
            "timestamp_start":   meta.get("timestamp_start"),
            "timestamp_end":     meta.get("timestamp_end"),
            "youtube_url":       meta.get("youtube_url", ""),
            "youtube_ts_start": meta.get("youtube_ts_start", ""),
            "youtube_ts_end":   meta.get("youtube_ts_end", ""),
        })
    return blocks


# ─── Format context ───────────────────────────────────────────────────────────

def fmt_context(blocks: list[dict]) -> str:
    """Format blocks với citation info cho prompt."""
    if not blocks:
        return ""
    lines = []
    for i, blk in enumerate(blocks, 1):
        lines.append(f"--- Context {i} ---")
        lines.append(format_context_block(blk)[:2000])
        lines.append("")
    return "\n".join(lines)


# ─── Gen ──────────────────────────────────────────────────────────────────────

def gen_for_topic(topic: dict, blocks: list[dict], llm, SamplingParams) -> list[dict]:
    topic_id   = topic["topic_id"]
    topic_name = topic["topic_name"]
    difficulty = topic.get("difficulty", "G2")
    num_q      = topic["num_questions"]

    context_str = fmt_context(blocks)
    if not context_str.strip():
        print(f"  ⚠️  No context for {topic_id}")
        return []

    results = []
    # Mix: single + multiple
    mix = []
    for i in range(num_q):
        if i == 0:
            mix.append(("single_correct", 1))
        elif i == 1:
            mix.append(("multiple_correct", 2))
        else:
            mix.append(("multiple_correct", 3))

    for seq, (q_type, count) in enumerate(mix):
        prompt = build_p1_gen_stem_key(
            topic=topic_name,
            difficulty_target=difficulty,
            concept_context_blocks=context_str,
            question_type_target=q_type,
            correct_answer_count_target=count,
        )
        messages = [{"role": "user", "content": prompt}]
        sampling = SamplingParams(
            **make_vllm_sampling(temperature=0.7, max_tokens=1024),
        )
        output = llm.chat(messages, sampling_params=sampling)
        raw = output[0].outputs[0].text
        parsed = parse_json_output(raw)

        # Attach citation metadata from used blocks
        for b in blocks[:3]:
            b_preview = {k: b.get(k) for k in
                         ["chunk_id", "youtube_url", "youtube_ts_start",
                          "source_type", "source_file", "timestamp_start"]}

        r = {
            "topic_id":   topic_id,
            "topic_name": topic_name,
            "difficulty": difficulty,
            "q_type":     q_type,
            "correct_count": count,
            "context_blocks": [
                {
                    "chunk_id":    b.get("chunk_id", ""),
                    "youtube_url":  b.get("youtube_url", ""),
                    "youtube_ts":   b.get("youtube_ts_start", ""),
                    "source_type":  b.get("source_type", ""),
                    "source_file":  b.get("source_file", ""),
                    "text_preview": b.get("text", "")[:100],
                }
                for b in blocks[:3]
            ],
            "raw_output": raw,
        }
        r.update(parsed)
        results.append(r)
        print(f"  ✅ {topic_id} q{seq+1}: {parsed.get('question_text', 'PARSE ERROR')[:80]}...")
    return results


def main():
    config.makedirs()

    print("\n📡 Retrieval...")
    all_blocks = {}
    for t in TEST_TOPICS:
        blocks = retrieve_blocks(t["topic_id"], t["topic_name"])
        video = [b for b in blocks if b.get("source_type") == "video_transcript"]
        slide = [b for b in blocks if b.get("source_type") == "slide_pdf"]
        print(f"  {t['topic_id']}: {len(blocks)} blocks "
              f"({len(video)} video, {len(slide)} slide)")
        all_blocks[t["topic_id"]] = blocks

    print("\n🤖 Loading Qwen2.5-14B-Instruct...")
    llm, SamplingParams = init_vllm_gen()
    print("  ✅ Model loaded\n")

    all_results = []
    for t in TEST_TOPICS:
        print(f"✍️  {t['topic_id']}: {t['topic_name']}...")
        blocks = all_blocks[t["topic_id"]]
        if not blocks:
            print(f"  ⚠️  skip — no context")
            continue
        results = gen_for_topic(t, blocks, llm, SamplingParams)
        all_results.extend(results)
        print()

    # Save
    out_dir = Config.OUTPUT_DIR / "test_gen_mini"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.jsonl"
    save_jsonl(all_results, out_file)

    # Summary
    print("=" * 80)
    print(f"📊 Generated: {len(all_results)} questions")
    print(f"💾 Saved: {out_file}")
    print("=" * 80)
    for i, r in enumerate(all_results, 1):
        print(f"\nQ{i} [{r['topic_name']} | {r['difficulty']} | {r['q_type']}]")
        q = r.get("question", r.get("raw_output", ""))
        print(f"  Stem: {q[:150]}")
        keys = r.get("correct_key", "?")
        print(f"  Key:  {keys}")
        for j, cb in enumerate(r["context_blocks"], 1):
            yt = cb.get("youtube_url", "")
            ts = cb.get("youtube_ts", "")
            if yt:
                print(f"  Nguồn {j}: {yt}&t={ts}s")
            else:
                src = cb.get("source_file", cb.get("chunk_id", "?"))
                print(f"  Nguồn {j}: (slide) {src}")


if __name__ == "__main__":
    main()
