"""
retrieval_hybrid.py — 5-tier Hybrid Retrieval cho MCQGen Pipeline
==================================================================
Áp dụng kỹ thuật hybrid retrieval từ tieplm project:
  Tier 1: BM25 lexical search (rank_bm25)
  Tier 2: ChromaDB vector search (BGE-m3)
  Tier 3: RRF (Reciprocal Rank Fusion, k=60)
  Tier 4: Cross-encoder reranking
  Tier 5: Metadata enrichment (youtube_url, slide, timestamps)

Sau khi có concept_chunks.jsonl đã được index vào ChromaDB,
module này build BM25 index từ JSONL và thực hiện hybrid search.

Usage:
    hr = HybridRetriever()
    hr.build_bm25_index()
    blocks = hr.retrieve("Logistic Regression", top_k=5)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from sentence_transformers import SentenceTransformer
# ── Setup ────────────────────────────────────────────────────────────────────
import sys as _sys
from pathlib import Path as _Path
# Add project root (parent of 'src/') so 'from common' and 'from gen.' work
_SRCDIR = str(_Path(__file__).resolve().parent)          # .../CS431MCQGen/src/
_PROJ   = str(_Path(__file__).resolve().parent.parent)  # .../CS431MCQGen/
for _p in (_PROJ, _SRCDIR):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from common import Config, config

PROJECT_ROOT = Config.PROJECT_ROOT  # resolves to CS431MCQGen/ correctly

# Lazy imports for heavy deps
BM25Okapi = None  # populated on first use


def _get_bm25():
    global BM25Okapi
    if BM25Okapi is None:
        from rank_bm25 import BM25Okapi as _cls
        BM25Okapi = _cls
    return BM25Okapi


def _get_chromadb():
    import chromadb
    return chromadb


def _get_embedder():
    return SentenceTransformer("BAAI/bge-m3",  device="cpu")


# ── Constants ────────────────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60  # RRF constant


# ── Chunk text index ─────────────────────────────────────────────────────────

def load_chunks_from_jsonl(jsonl_path: Path) -> list[dict]:
    """Load all chunks from concept_chunks.jsonl into memory."""
    chunks = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return chunks


# ── Main class ──────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    5-tier hybrid retriever:
      1. BM25 on concept_chunks.jsonl text
      2. ChromaDB vector search (BGE-m3)
      3. RRF fusion (BM25 + Vector)
      4. Cross-encoder rerank
      5. Metadata enrichment
    """

    def __init__(
        self,
        chroma_dir: Path | None = None,
        chunk_file: Path | None = None,
        top_k_vector: int = 150,
        top_k_bm25: int = 150,
    ):
        self.chroma_dir   = Path(chroma_dir) if chroma_dir else Config.INDEX_DIR
        self.chunk_file   = Path(chunk_file) if chunk_file else Config.CONCEPT_CHUNKS_FILE
        self.top_k_vector = top_k_vector
        self.top_k_bm25  = top_k_bm25

        # Lazy: BM25 index built on first search
        self._bm25_index: Any = None
        self._bm25_corpus: list[list[str]] = []
        self._bm25_chunk_ids: list[str] = []
        self._chunk_map: dict[str, dict] = {}   # chunk_id → full chunk dict

        # Lazy: ChromaDB collection
        self._chroma_collection: Any = None

        # Lazy: Embedding model
        self._embedder: Any = None

        # Lazy: Cross-encoder reranker
        self._reranker: Any = None

    def build_bm25_index(self) -> None:
        """Public wrapper — alias for _ensure_bm25(). Call before retrieval to warm up."""
        self._ensure_bm25()

    # ── Tier 1: BM25 index ──────────────────────────────────────────────────

    def _ensure_bm25(self) -> None:
        """Build BM25 index from JSONL chunks (idempotent)."""
        if self._bm25_index is not None:
            return

        if not self.chunk_file.exists():
            raise FileNotFoundError(
                f"concept_chunks.jsonl not found at {self.chunk_file}. "
                "Run Step 01 indexing first."
            )

        print(f"  Building BM25 index from {self.chunk_file}...")
        chunks = load_chunks_from_jsonl(self.chunk_file)

        # Build chunk map
        for c in chunks:
            self._chunk_map[c["chunk_id"]] = c

        # Tokenize corpus (simple whitespace + lowercase)
        BM25Cls = _get_bm25()
        self._bm25_corpus = [
            c["text"].lower().split() for c in chunks
        ]
        self._bm25_chunk_ids = [c["chunk_id"] for c in chunks]
        self._bm25_index = BM25Cls(self._bm25_corpus)
        print(f"  ✅ BM25 index built: {len(self._bm25_corpus)} chunks")

    def search_bm25(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Tier 1: BM25 lexical search.
        Returns: [{chunk_id, score, method}]
        """
        self._ensure_bm25()
        top_k = top_k or self.top_k_bm25

        tokenized = query.lower().split()
        scores = self._bm25_index.get_scores(tokenized)

        # Get top-K indices sorted by score desc
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in ranked:
            if scores[idx] > 0:
                results.append({
                    "chunk_id": self._bm25_chunk_ids[idx],
                    "score": float(scores[idx]),
                    "method": "bm25",
                })
        return results

    # ── Tier 2: ChromaDB vector search ────────────────────────────────────

    def _ensure_chromadb(self) -> None:
        """Load ChromaDB collection and embedder (idempotent)."""
        if self._chroma_collection is not None:
            return

        print("  Loading ChromaDB concept collection...")
        chromadb = _get_chromadb()
        print(f"  Connecting to ChromaDB at {self.chroma_dir}...")
        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        print("  Accessing 'concept_chunks' collection...")
        self._chroma_collection = client.get_or_create_collection("concept_chunks")
        print("  ✅ ChromaDB collection ready")
        self._embedder = _get_embedder()
        print("  ✅ ChromaDB + BGE-m3 loaded")

    def search_vector(
        self,
        query: str,
        top_k: int | None = None,
        chapter_filter: list[str] | None = None,
    ) -> list[dict]:
        """
        Tier 2: Vector similarity search via ChromaDB + BGE-m3.
        Returns: [{chunk_id, score, method, metadata}]
        """
        self._ensure_chromadb()
        top_k = top_k or self.top_k_vector

        # Embed query
        q_emb = self._embedder.encode([query]).tolist()[0]

        # Build ChromaDB where filter for chapter
        where_filter = None
        if chapter_filter and len(chapter_filter) == 1:
            where_filter = {"chapter_id": chapter_filter[0]}

        # Query ChromaDB
        try:
            chroma_results = self._chroma_collection.query(
                query_embeddings=[q_emb],
                n_results=top_k,
                where=where_filter,
            )
        except Exception as e:
            print(f"  ⚠️  ChromaDB query error: {e}")
            chroma_results = {"documents": [[]], "metadatas": [[]], "ids": [[]]}

        results = []
        for chunk_id, doc, meta in zip(
            chroma_results.get("ids", [[]])[0],
            chroma_results.get("documents", [[]])[0],
            chroma_results.get("metadatas", [[]])[0],
        ):
            # Distance → similarity (cosine-ish, higher = better)
            dists = chroma_results.get("distances", [[]])[0]
            scores = chroma_results.get("distances", [[]])[0]
            results.append({
                "chunk_id": chunk_id,
                "score": 1.0 - scores[len(results)] if scores else 1.0,
                "method": "vector",
                "metadata": meta,
            })
        return results

    # ── Tier 3: RRF fusion ─────────────────────────────────────────────────

    @staticmethod
    def combine_rrf(
        vector_results: list[dict],
        bm25_results: list[dict],
        k: int = RRF_K,
        final_top_k: int = 150,
    ) -> list[dict]:
        """
        Tier 3: Reciprocal Rank Fusion.
        score(d) = Σ 1 / (k + rank_in_list)
        Combines both ranked lists into a single fused ranking.
        """
        rrf_scores: dict[str, float] = {}
        result_data: dict[str, dict] = {}

        # Vector results
        for rank, result in enumerate(vector_results, start=1):
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))
            if cid not in result_data:
                result_data[cid] = result

        # BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))
            if cid not in result_data:
                result_data[cid] = result

        # Sort by RRF score desc
        sorted_ids = sorted(rrf_scores, key=lambda c: rrf_scores[c], reverse=True)[:final_top_k]

        fused = []
        for cid in sorted_ids:
            r = result_data[cid].copy()
            r["rrf_score"] = rrf_scores[cid]
            fused.append(r)
        return fused

    # ── Tier 4: Cross-encoder rerank ──────────────────────────────────────

    def _ensure_reranker(self) -> None:
        """Load cross-encoder reranker (idempotent)."""
        if self._reranker is not None:
            return
        print("  Loading cross-encoder reranker...")
        from sentence_transformers import CrossEncoder
        self._reranker = CrossEncoder(RERANKER_MODEL)
        print(f"  ✅ Cross-encoder loaded: {RERANKER_MODEL}")

    def rerank_crossencoder(
        self,
        query: str,
        combined_results: list[dict],
        top_k: int = 30,
    ) -> list[dict]:
        """
        Tier 4: Cross-encoder rerank on top-N fused results.
        Re-scores query-document pairs using a cross-encoder model.
        """
        if not combined_results:
            return []

        self._ensure_reranker()

        # Build query-document pairs from combined results
        pairs = []
        valid_results = []
        for r in combined_results:
            # Try ChromaDB metadata first, then BM25 text from chunk map
            text = ""
            if "metadata" in r and r["metadata"]:
                text = r["metadata"].get("text", "") or r["metadata"].get("document", "")
            if not text:
                # Fallback to chunk map
                c = self._chunk_map.get(r["chunk_id"], {})
                text = c.get("text", "")
            if text:
                pairs.append([query, text[:1500]])  # Truncate for speed
                valid_results.append(r)

        if not pairs:
            return combined_results[:top_k]

        # Score all pairs
        scores = self._reranker.predict(pairs, show_progress_bar=False)

        # Attach rerank score
        for r, score in zip(valid_results, scores):
            r["rerank_score"] = float(score)

        # Sort by rerank score desc
        reranked = sorted(valid_results, key=lambda r: r.get("rerank_score", 0), reverse=True)[:top_k]
        return reranked

    # ── Tier 5: Metadata enrichment ────────────────────────────────────────

    def enrich_metadata(self, results: list[dict]) -> list[dict]:
        """
        Tier 5: Attach full citation metadata to each result.
        Sources: ChromaDB metadata → chunk_map JSONL → video/slide labels.
        """
        enriched = []
        for r in results:
            chunk_id = r["chunk_id"]
            meta = r.get("metadata", {})

            # Start with ChromaDB metadata
            enriched_r = {
                "chunk_id":        chunk_id,
                "chapter_id":      meta.get("chapter_id", ""),
                "chapter_title":   meta.get("chapter_title", ""),
                "topic":           meta.get("topic", ""),
                "section_title":   meta.get("section_title", ""),
                "text":            meta.get("document", ""),
                "source":          "concept_kb",
                "source_type":     meta.get("source_type", ""),
                "source_file":     meta.get("source_file", ""),
                "youtube_url":     meta.get("youtube_url", ""),
                "youtube_ts_start": meta.get("youtube_ts_start", ""),
                "youtube_ts_end":  meta.get("youtube_ts_end", ""),
                "slide_file":      meta.get("slide_file", ""),
                # slide_start_page defaults to 0; will be upgraded from chunk_map if needed
                "slide_start_page": 0,
            }

            # Enrich from chunk_map JSONL (richer data than ChromaDB for slides)
            chunk_data = self._chunk_map.get(chunk_id, {})
            if chunk_data:
                # Priority: chunk_map overrides ChromaDB for these critical fields
                # (ChromaDB may have "" or 0 while chunk_map has real values)
                CRITICAL_KEYS = (
                    "youtube_url", "timestamp_start", "timestamp_end",
                    "youtube_timestamp_start", "youtube_timestamp_end",
                    "video_id", "sub_index",
                )
                for key in CRITICAL_KEYS:
                    if key in chunk_data:
                        current = enriched_r.get(key)
                        # Override if current is empty/zero and chunk_map has a real value
                        if not current and chunk_data[key]:
                            enriched_r[key] = chunk_data[key]

                # Other metadata: fill in if missing
                for key in ("source_file", "chapter_id", "chapter_title", "source_type",
                            "slide_file"):
                    if key in chunk_data and not enriched_r.get(key):
                        enriched_r[key] = chunk_data[key]

                # slide_start_page: ChromaDB stores 0 for slides → upgrade from page_number
                if enriched_r.get("slide_start_page", 0) == 0 and chunk_data.get("page_number"):
                    enriched_r["slide_start_page"] = chunk_data["page_number"]

                # Use richer text from JSONL if ChromaDB stripped it
                if chunk_data.get("text") and len(chunk_data["text"]) > len(enriched_r.get("text", "")):
                    enriched_r["text"] = chunk_data["text"]

            enriched.append(enriched_r)
        return enriched

    # ── Top-level retrieve ─────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        chapter_filter: list[str] | None = None,
        use_bm25: bool = True,
        use_rerank: bool = True,
        return_all_tiers: bool = False,
    ) -> list[dict]:
        """
        Full 5-tier retrieval pipeline.

        Args:
            query:          Topic / question name to search for
            top_k:          Number of final context blocks to return
            chapter_filter: Optional list of chapter_ids to filter (e.g. ["ch07b"])
            use_bm25:       Include BM25 in fusion (default True)
            use_rerank:     Apply cross-encoder rerank (default True)
            return_all_tiers: If True, return intermediate tier results too

        Returns:
            List of top_k enriched context blocks, sorted by relevance.
        """
        print(f"  🔍 Hybrid retrieval for: '{query}'")

        # ── Tier 1: BM25 ───────────────────────────────────────────────────
        bm25_results = []
        if use_bm25:
            bm25_results = self.search_bm25(query, top_k=self.top_k_bm25)
            print(f"    T1 BM25: {len(bm25_results)} results")

        # ── Tier 2: Vector ────────────────────────────────────────────────
        vector_results = self.search_vector(
            query, top_k=self.top_k_vector, chapter_filter=chapter_filter
        )
        print(f"    T2 Vector: {len(vector_results)} results")

        # ── Tier 3: RRF ───────────────────────────────────────────────────
        if use_bm25 and bm25_results:
            fused_results = self.combine_rrf(
                vector_results, bm25_results,
                k=RRF_K, final_top_k=150
            )
        else:
            fused_results = vector_results[:150]
        print(f"    T3 RRF: {len(fused_results)} fused results")

        # ── Tier 4: Rerank ────────────────────────────────────────────────
        if use_rerank:
            reranked = self.rerank_crossencoder(query, fused_results, top_k=30)
            print(f"    T4 Rerank: {len(reranked)} reranked")
        else:
            reranked = fused_results[:30]

        # ── Tier 5: Enrich ────────────────────────────────────────────────
        enriched = self.enrich_metadata(reranked)
        print(f"    T5 Enrich: {len(enriched)} enriched")

        # Return top_k
        final = enriched[:top_k]
        print(f"    ✅ Returning top-{len(final)} blocks")

        if return_all_tiers:
            return {
                "query": query,
                "bm25_results": bm25_results,
                "vector_results": vector_results,
                "fused_results": fused_results,
                "reranked_results": reranked,
                "final_results": final,
            }

        return final


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test hybrid retrieval")
    parser.add_argument("--query", default="Logistic Regression", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K results")
    parser.add_argument("--chapter", default=None, help="Filter by chapter_id")
    parser.add_argument("--no-rerank", action="store_true", help="Skip cross-encoder rerank")
    args = parser.parse_args()

    config.makedirs()

    hr = HybridRetriever()
    hr.build_bm25_index()

    chapter_filter = [args.chapter] if args.chapter else None
    blocks = hr.retrieve(
        query=args.query,
        top_k=args.top_k,
        chapter_filter=chapter_filter,
        use_rerank=not args.no_rerank,
    )

    print("\n── Results ──")
    for i, b in enumerate(blocks, 1):
        print(f"\n{i}. [{b['chunk_id']}] [{b['chapter_id']}]")
        print(f"   source_type={b['source_type']}")
        print(f"   youtube_url={b['youtube_url']}")
        print(f"   slide_file={b['slide_file']}")
        print(f"   text={b['text'][:150]}...")


if __name__ == "__main__":
    main()
