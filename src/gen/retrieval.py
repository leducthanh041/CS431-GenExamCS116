"""
retrieval.py — Step 02: RAG Retrieval
MCQGen Pipeline: ChromaDB → Top-K context blocks cho từng topic

Input:
  - input/topic_list.json
  - data/indexes/concept_kb/         (ChromaDB concept collection)
  - data/indexes/assessment_kb/      (ChromaDB assessment collection)

Output:
  - data/intermediate/02_retrieval/<topic_id>.jsonl
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (
    Config, config,
    load_topic_list, save_jsonl,
    format_context_block,
)


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
    Entry point cho Step 02 — RAG retrieval cho tất cả topics.
    """
    config.makedirs()

    # ─── Load ChromaDB + embedding model ──────────────────────────
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        sys.exit(1)

    print("🔄 Loading ChromaDB concept index...")
    concept_client = chromadb.PersistentClient(path=str(Config.CONCEPT_KB_DIR))
    try:
        concept_collection = concept_client.get_collection("concept_chunks")
    except Exception:
        concept_collection = concept_client.get_or_create_collection("concept_chunks")

    print("🔄 Loading ChromaDB assessment index...")
    assess_client = chromadb.PersistentClient(path=str(Config.ASSESSMENT_KB_DIR))
    try:
        assess_collection = assess_client.get_collection("assessment_items")
    except Exception:
        assess_collection = assess_client.get_or_create_collection("assessment_items")

    print("🔄 Loading BGE-m3 for query embedding...")
    embed_model = SentenceTransformer("BAAI/bge-m3")
    print("✅ ChromaDB + BGE-m3 loaded")

    # ─── Load topics ────────────────────────────────────────────────
    topics_raw = load_topic_list()
    topic_entries = []
    for ch in topics_raw:
        for t in ch.get("topics", []):
            t["chapter_id"] = ch["chapter_id"]
            topic_entries.append(t)

    print(f"📂 Loaded {len(topic_entries)} topics")

    # ─── Retrieval cho từng topic ──────────────────────────────────
    total = 0
    for entry in topic_entries:
        topic_id   = entry["topic_id"]
        topic_name = entry["topic_name"]
        query      = expand_query(topic_name)

        out_file = Config.RETRIEVE_OUTPUT / f"{topic_id}.jsonl"
        results = []

        # Embed query bằng BGE-m3 (1024-dim) — phải match với lúc index
        query_embedding = embed_model.encode([query]).tolist()

        # Retrieval từ concept KB — dùng query_embeddings thay vì query_texts
        try:
            concept_results = concept_collection.query(
                query_embeddings=query_embedding,
                n_results=Config.RETRIEVAL_CONCEPT_TOP_K,
            )
        except Exception as e:
            print(f"  ⚠️  ChromaDB query error for {topic_id}: {e}")
            concept_results = {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

        # Retrieval từ assessment KB (optional reference)
        try:
            assess_results = assess_collection.query(
                query_embeddings=query_embedding,
                n_results=Config.RETRIEVAL_ASSESSMENT_TOP_K,
            )
        except Exception:
            assess_results = {"documents": [[]], "metadatas": [[]]}

        # Build context blocks
        concept_docs = concept_results.get("documents", [[]])[0]
        concept_metas = concept_results.get("metadatas", [[]])[0]

        blocks = []
        for i, (doc, meta) in enumerate(zip(concept_docs, concept_metas)):
            if not doc:
                continue
            block = {
                "chunk_id": meta.get("chunk_id", f"{topic_id}_c{i}"),
                "chapter_id": meta.get("chapter_id", entry.get("chapter_id", "")),
                "topic": meta.get("topic", topic_name),
                "section_title": meta.get("section_title", ""),
                "text": doc,
                "source": "concept_kb",
            }
            blocks.append(block)

        if not blocks:
            print(f"  ⚠️  No context found for {topic_id}")
            continue

        # Lưu retrieval result cho topic
        record = {
            "topic_id": topic_id,
            "topic_name": topic_name,
            "query": query,
            "num_context_blocks": len(blocks),
            "context_blocks": blocks,
            "context_blocks_str": "\n".join(
                format_context_block(b) for b in blocks
            ),
            "assessment_reference": assess_results.get("documents", [[]])[0][:2],
        }
        results.append(record)

        save_jsonl(results, out_file)
        total += 1
        print(f"  ✅ {topic_id}: {len(blocks)} concept blocks retrieved")

    print(f"\n✅ Retrieval done. {total} topics → {Config.RETRIEVE_OUTPUT}")


if __name__ == "__main__":
    run_retrieval()
