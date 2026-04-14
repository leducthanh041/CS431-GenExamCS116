"""
retrieval.py — Step 02: 5-tier Hybrid RAG Retrieval
MCQGen Pipeline: ChromaDB + BM25 → Top-K context blocks cho từng topic

Input:
  - input/topic_list.json
  - data/indexes/concept_kb/         (ChromaDB concept collection)
  - data/processed/concept_chunks.jsonl   (BM25 source)
  - configs/generation_config.yaml    (optional: topic weights, focus topics)

Output:
  - data/intermediate/02_retrieval/<topic_id>.jsonl

Retrieval pipeline (5-tier from tieplm):
  Tier 1: BM25 lexical search on concept_chunks.jsonl text
  Tier 2: ChromaDB vector search (BGE-m3 embedding)
  Tier 3: RRF fusion (Reciprocal Rank Fusion, k=60)
  Tier 4: Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
  Tier 5: Metadata enrichment (youtube_url, slide, timestamps)
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
    load_topic_list, save_jsonl,
    format_context_block,
)
from gen.retrieval_hybrid import HybridRetriever

# ── Override EXP_NAME from environment ────────────────────────────────────────
_exp_name = os.environ.get("EXP_NAME", "")
if _exp_name:
    Config.EXP_NAME = _exp_name
    Config.OUTPUT_DIR = Config.PROJECT_ROOT / "output" / Config.EXP_NAME
    Config.RETRIEVE_OUTPUT = Config.OUTPUT_DIR / "02_retrieval"
    Config.GEN_STEM_OUTPUT = Config.OUTPUT_DIR / "03_gen_stem"
    print(f"[retrieval] EXP_NAME overridden: {Config.EXP_NAME}")


# ─── Keyword expansion map (từ retrieval_pipeline.py cũ) ───────────────────

TOPIC_KEYWORDS_MAP: dict[str, list[str]] = {
    "Missing Data": [
        "missing data", "dữ liệu thiếu", "missing values", "null values",
        "imputation", "imputer", "KNNImputer", "SimpleImputer", "IterativeImputer",
        "dropna", "fillna", "handle missing",
    ],
    "Outlier Detection": [
        "outlier", "outlier detection", "ngoại lai",
        "anomaly detection", "Local Outlier Factor", "LOF", "Isolation Forest",
        "EllipticEnvelope", "IQR", "z-score",
    ],
    "Feature Extraction": [
        "feature extraction", "trích xuất đặc trưng",
        "PCA", "t-SNE", "UMAP", "SVD", "Autoencoder", "LDA",
    ],
    "Feature Transformation": [
        "feature transformation", "biến đổi đặc trưng",
        "normalization", "standardization", "scaling",
        "min-max scaling", "log transform", "binning", "encoding",
        "one-hot encoding", "label encoding",
    ],
    "Feature Selection": [
        "feature selection", "chọn lựa đặc trưng",
        "RFE", "Recursive Feature Elimination",
        "Forward Selection", "Backward Selection",
        "SelectKBest", "mutual information",
        "Lasso", "L1 regularization",
    ],
    "NumPy": ["numpy", "array", "ndarray", "vectorization", "broadcasting"],
    "Pandas": ["pandas", "dataframe", "series", "groupby", "merge"],
    "Matplotlib": ["matplotlib", "plot", "visualization", "chart"],
    "Scikit-learn": ["scikit-learn", "sklearn", "fit", "predict", "pipeline"],
    "Pipeline": ["pipeline", "sklearn pipeline", "make_pipeline"],
    "Exploratory Data Analysis": ["EDA", "exploratory", "phân tích dữ liệu"],
    "Classification Metrics": [
        "accuracy", "precision", "recall", "f1", "confusion matrix",
        "classification report", "ROC", "AUC",
    ],
    "Regression Metrics": [
        "MSE", "RMSE", "MAE", "R2", "mean squared error",
        "regression metrics",
    ],
    "Cross-validation": ["cross-validation", "k-fold", "cv", "validation set"],
    "Clustering": ["clustering", "k-means", "hierarchical", "DBSCAN", "silhouette"],
    "Dimensionality Reduction": [
        "dimensionality reduction", "PCA", "t-SNE", "UMAP", "SVD",
    ],
    "Linear Regression": [
        "linear regression", "hồi quy tuyến tính",
        "least squares", "OLS", "gradient descent",
    ],
    "Regularization": [
        "regularization", "ridge", "lasso", "l1", "l2", "chính quy",
    ],
    "Logistic Regression": [
        "logistic regression", "hồi quy logistic", "sigmoid", "odds",
    ],
    "Decision Trees": [
        "decision tree", "cây quyết định", "gini", "entropy", "information gain",
    ],
    "SVM": ["SVM", "support vector machine", "kernel", "margin", "hyperplane"],
    "Neural Networks": [
        "neural network", "mạng nơ-ron", "activation", "layer", "backpropagation",
    ],
    "CNN": ["CNN", "convolutional", "filter", "pooling", "feature map"],
    "Grid Search": ["grid search", "gridsearchcv", "hyperparameter tuning"],
    "Random Search": ["random search", "randomizedsearchcv"],
    "Bayesian Optimization": ["bayesian", "optuna", "hyperparameter optimization"],
    "Bagging": ["bagging", "bootstrap aggregating", "random forest"],
    "Boosting": ["boosting", "adaboost", "xgboost", "gradient boost", "lightgbm"],
    "Random Forest": ["random forest", "ensemble", "tree-based"],
    "Model Serving": ["model serving", "deployment", "model inference", "prediction"],
    "API": ["API", "Flask", "FastAPI", "REST API", "endpoint"],
    "Monitoring": ["monitoring", "drift detection", "performance tracking"],
}


def expand_query(topic_name: str) -> str:
    """Mở rộng query với keywords liên quan."""
    keywords = TOPIC_KEYWORDS_MAP.get(topic_name, [])
    if not keywords:
        return topic_name
    return topic_name + " " + " ".join(keywords[:5])


def run_retrieval():
    """
    Entry point cho Step 02 — 5-tier hybrid RAG retrieval cho tất cả topics.
    """
    config.makedirs()

    # ─── Load generation config (optional topic weights / focus) ────────────
    try:
        from gen.prompt_config import load_generation_config
        gen_cfg = load_generation_config()
        print("📋 Loaded generation_config.yaml (topic weights / focus topics)")
    except Exception:
        gen_cfg = {}
        print("📋 No generation_config.yaml — using topic_list.json as-is")

    # ─── Build hybrid retriever (BM25 + ChromaDB + RRF + Rerank) ────────────
    print("🔄 Building hybrid retriever (5-tier retrieval)...")
    hr = HybridRetriever(
        chroma_dir=Config.INDEX_DIR,
        chunk_file=Config.CONCEPT_CHUNKS_FILE,
        top_k_vector=150,
        top_k_bm25=150,
    )
    hr.build_bm25_index()
    print("✅ Hybrid retriever ready")

    # ─── Load topics ────────────────────────────────────────────────────────
    topics_raw = load_topic_list()
    topic_entries = []
    for ch in topics_raw:
        for t in ch.get("topics", []):
            t = dict(t)  # copy
            t["chapter_id"]   = ch["chapter_id"]
            t["chapter_name"] = ch["chapter_name"]
            topic_entries.append(t)

    # Apply topic weights / focus from gen_cfg
    focus_chapters = gen_cfg.get("generation", {}).get("focus_chapters", [])
    if focus_chapters:
        topic_entries = [t for t in topic_entries if t["chapter_id"] in focus_chapters]

    focus_topics = gen_cfg.get("generation", {}).get("focus_topics", [])
    if focus_topics:
        topic_entries = [t for t in topic_entries if t["topic_id"] in focus_topics]

    print(f"📂 Loaded {len(topic_entries)} topics")

    # ─── Hybrid retrieval cho từng topic ───────────────────────────────────
    total = 0
    for entry in topic_entries:
        topic_id   = entry["topic_id"]
        topic_name = entry["topic_name"]
        query      = expand_query(topic_name)
        chapter_id = entry.get("chapter_id", "")

        out_file = Config.RETRIEVE_OUTPUT / f"{topic_id}.jsonl"
        results = []

        # ── 5-tier hybrid retrieval ──────────────────────────────────────────
        try:
            blocks = hr.retrieve(
                query=query,
                top_k=Config.RETRIEVAL_CONCEPT_TOP_K,  # top-K for prompt
                chapter_filter=[chapter_id] if chapter_id else None,
                use_bm25=True,
                use_rerank=True,
            )
        except Exception as e:
            print(f"  ⚠️  Hybrid retrieval error for {topic_id}: {e}")
            traceback.print_exc()
            blocks = []

        # ── Build context string (same format as before) ────────────────────
        context_blocks_str = "\n".join(
            format_context_block(b) for b in blocks
        )

        record = {
            "topic_id": topic_id,
            "topic_name": topic_name,
            "query": query,
            "num_context_blocks": len(blocks),
            "context_blocks": blocks,
            "context_blocks_str": context_blocks_str,
            "assessment_reference": "",
        }
        results.append(record)
        save_jsonl(results, out_file)
        total += 1
        print(f"  ✅ {topic_id}: {len(blocks)} context blocks (hybrid, 5-tier)")

    print(f"\n✅ Hybrid retrieval done. {total} topics → {Config.RETRIEVE_OUTPUT}")


if __name__ == "__main__":
    run_retrieval()
