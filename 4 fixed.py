"""
Fixed Step 4 pipeline.

Intended design:
1. Config A and Config B retrieve candidates from different embedding-model indexes.
2. The same cross-encoder reranker is applied to both candidate sets.
3. Metadata-aware scoring is applied only after reranking.

Important:
- Set each config's `query_model_name` to the exact model used to create that
  config's stored embeddings.
- The reranker uses the `content` field returned by vector search directly.
- Likes/upvotes are used only in the final metadata-aware scoring step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalConfig:
    label: str
    collection: Any
    query_model_name: str


RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_query_models(configs: dict[str, RetrievalConfig]) -> dict[str, SentenceTransformer]:
    """Load one query encoder per retrieval config."""
    models: dict[str, SentenceTransformer] = {}
    for key, config in configs.items():
        models[key] = SentenceTransformer(config.query_model_name)
    return models


def load_reranker() -> CrossEncoder:
    """Load the shared cross-encoder reranker."""
    return CrossEncoder(RERANKER_MODEL_NAME)


# ---------------------------------------------------------------------------
# Stage 1: Vector Retrieval
# ---------------------------------------------------------------------------


def retrieve_chromadb(
    query: str,
    collection: Any,
    query_model: SentenceTransformer,
    top_k: int = 50,
) -> tuple[list[dict[str, Any]], float]:
    """
    Retrieve candidates from one ChromaDB collection.

    The query model must match the embedding model used to build this collection.
    """
    query_vector = query_model.encode(query, convert_to_numpy=True).tolist()

    start = time.time()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    latency = time.time() - start

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    candidates: list[dict[str, Any]] = []
    for rank, (candidate_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances),
        start=1,
    ):
        candidates.append(
            {
                "rank": rank,
                "id": str(candidate_id),
                "similarity": float(1 - distance),
                "distance": float(distance),
                "title": metadata.get("title", ""),
                "category": metadata.get("category", ""),
                "subcategory": metadata.get("subcategory", ""),
                "tags": metadata.get("tags", ""),
                "difficulty": metadata.get("difficulty", ""),
                "likes": _to_number(metadata.get("likes", 0)),
                "upvotes": _to_number(metadata.get("upvotes", 0)),
                "content": document,
            }
        )

    return candidates, latency


def retrieve_both_configs(
    query: str,
    configs: dict[str, RetrievalConfig],
    query_models: dict[str, SentenceTransformer],
    candidate_k: int = 50,
) -> dict[str, dict[str, Any]]:
    """Retrieve candidate sets from all configured vector indexes."""
    output: dict[str, dict[str, Any]] = {}
    for key, config in configs.items():
        candidates, latency = retrieve_chromadb(
            query=query,
            collection=config.collection,
            query_model=query_models[key],
            top_k=candidate_k,
        )
        output[key] = {
            "label": config.label,
            "candidates": candidates,
            "latency": latency,
        }
    return output


# ---------------------------------------------------------------------------
# Stage 2: Cross-Encoder Reranking
# ---------------------------------------------------------------------------


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    reranker_model: CrossEncoder,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Rerank vector candidates once using the shared cross-encoder.

    Vector search already returns each candidate's original text in the
    `content` field, so the cross-encoder can consume it directly.
    """
    pairs = [[query.strip(), candidate["content"]] for candidate in candidates]
    scores = reranker_model.predict(pairs)

    reranked = [
        {
            "candidate": candidate,
            "rerank_score": float(score),
        }
        for candidate, score in zip(candidates, scores)
    ]
    reranked = sorted(reranked, key=lambda item: item["rerank_score"], reverse=True)

    if top_k is not None:
        return reranked[:top_k]
    return reranked


def rerank_all_configs(
    query: str,
    retrieved: dict[str, dict[str, Any]],
    reranker_model: CrossEncoder,
) -> dict[str, dict[str, Any]]:
    """Apply the same reranker to every config's retrieved candidate set."""
    output: dict[str, dict[str, Any]] = {}
    for key, result in retrieved.items():
        output[key] = {
            **result,
            "reranked": rerank_candidates(
                query=query,
                candidates=result["candidates"],
                reranker_model=reranker_model,
                top_k=None,
            ),
        }
    return output


# ---------------------------------------------------------------------------
# Stage 3: Metadata-Aware Final Scoring
# ---------------------------------------------------------------------------


def metadata_aware_rescore(
    reranked: list[dict[str, Any]],
    top_k: int = 10,
    alpha: float = 0.8,
    beta: float = 0.1,
    gamma: float = 0.1,
) -> list[dict[str, Any]]:
    """
    Apply metadata-aware scoring after cross-encoder reranking.

    combined_score =
        alpha * normalized_rerank_score
      + beta  * normalized_log_likes
      + gamma * normalized_log_upvotes
    """
    if not np.isclose(alpha + beta + gamma, 1.0):
        raise ValueError("alpha, beta, and gamma should sum to 1.0")

    rerank_scores = np.array([item["rerank_score"] for item in reranked], dtype=np.float32)
    likes = np.array([_to_number(item["candidate"].get("likes", 0)) for item in reranked], dtype=np.float32)
    upvotes = np.array([_to_number(item["candidate"].get("upvotes", 0)) for item in reranked], dtype=np.float32)

    rerank_norm = minmax_normalize(rerank_scores)
    likes_norm = minmax_normalize(np.log1p(likes))
    upvotes_norm = minmax_normalize(np.log1p(upvotes))

    combined_scores = alpha * rerank_norm + beta * likes_norm + gamma * upvotes_norm

    rescored: list[dict[str, Any]] = []
    for item, score in zip(reranked, combined_scores):
        rescored.append(
            {
                **item,
                "combined_score": float(score),
            }
        )

    return sorted(rescored, key=lambda item: item["combined_score"], reverse=True)[:top_k]


def final_score_all_configs(
    reranked_by_config: dict[str, dict[str, Any]],
    top_k: int = 10,
    alpha: float = 0.8,
    beta: float = 0.1,
    gamma: float = 0.1,
) -> dict[str, dict[str, Any]]:
    """Apply metadata-aware final scoring to every config."""
    output: dict[str, dict[str, Any]] = {}
    for key, result in reranked_by_config.items():
        output[key] = {
            **result,
            "metadata_aware": metadata_aware_rescore(
                result["reranked"],
                top_k=top_k,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            ),
        }
    return output


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def precision_at_k(results: list[dict[str, Any]], relevant_ids: set[str], k: int) -> float:
    top_ids = [_candidate_id(item) for item in results[:k]]
    hits = sum(1 for candidate_id in top_ids if candidate_id in relevant_ids)
    return hits / k


def reciprocal_rank(results: list[dict[str, Any]], relevant_ids: set[str]) -> float:
    for rank, item in enumerate(results, start=1):
        if _candidate_id(item) in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: list[dict[str, Any]], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for index, item in enumerate(results[:k]):
        if _candidate_id(item) in relevant_ids:
            dcg += 1.0 / np.log2(index + 2)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(index + 2) for index in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_fixed_pipeline(
    ground_truth: dict[str, list[str]],
    configs: dict[str, RetrievalConfig],
    query_models: dict[str, SentenceTransformer],
    reranker_model: CrossEncoder,
    candidate_k: int = 50,
    top_k: int = 10,
    alpha: float = 0.8,
    beta: float = 0.1,
    gamma: float = 0.1,
) -> pd.DataFrame:
    """Evaluate vector-only, reranked, and metadata-aware stages."""
    rows: list[dict[str, Any]] = []

    for query, relevant_ids_list in ground_truth.items():
        relevant_ids = set(map(str, relevant_ids_list))

        retrieved = retrieve_both_configs(query, configs, query_models, candidate_k=candidate_k)
        reranked = rerank_all_configs(query, retrieved, reranker_model)
        final = final_score_all_configs(
            reranked,
            top_k=top_k,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        for key, result in final.items():
            vector_results = result["candidates"][:top_k]
            reranked_results = result["reranked"][:top_k]
            metadata_results = result["metadata_aware"][:top_k]

            rows.extend(
                [
                    _metric_row(query, key, result["label"], "Vector Search", vector_results, relevant_ids, top_k),
                    _metric_row(query, key, result["label"], "Cross-Encoder Reranked", reranked_results, relevant_ids, top_k),
                    _metric_row(query, key, result["label"], "Metadata-Aware Final", metadata_results, relevant_ids, top_k),
                ]
            )

    return pd.DataFrame(rows)


def summarize_evaluation(evaluation: pd.DataFrame) -> pd.DataFrame:
    """Aggregate evaluation metrics by config and stage."""
    return (
        evaluation.groupby(["config_key", "config_label", "stage"], as_index=False)
        .agg(
            mean_p_at_k=("precision_at_k", "mean"),
            mrr=("reciprocal_rank", "mean"),
            mean_ndcg_at_k=("ndcg_at_k", "mean"),
        )
        .sort_values(["stage", "config_key"])
        .reset_index(drop=True)
    )


def run_fixed_pipeline_for_query(
    query: str,
    configs: dict[str, RetrievalConfig],
    query_models: dict[str, SentenceTransformer],
    reranker_model: CrossEncoder,
    candidate_k: int = 50,
    top_k: int = 10,
    alpha: float = 0.8,
    beta: float = 0.1,
    gamma: float = 0.1,
) -> dict[str, dict[str, Any]]:
    """Run the full fixed pipeline for one query."""
    retrieved = retrieve_both_configs(query, configs, query_models, candidate_k=candidate_k)
    reranked = rerank_all_configs(query, retrieved, reranker_model)
    return final_score_all_configs(
        reranked,
        top_k=top_k,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    min_value = float(values.min()) if len(values) else 0.0
    max_value = float(values.max()) if len(values) else 0.0

    if max_value == min_value:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_value) / (max_value - min_value)


def _candidate_id(item: dict[str, Any]) -> str:
    if "candidate" in item:
        return str(item["candidate"]["id"])
    return str(item["id"])


def _metric_row(
    query: str,
    config_key: str,
    config_label: str,
    stage: str,
    results: list[dict[str, Any]],
    relevant_ids: set[str],
    top_k: int,
) -> dict[str, Any]:
    return {
        "query": query,
        "config_key": config_key,
        "config_label": config_label,
        "stage": stage,
        "precision_at_k": precision_at_k(results, relevant_ids, top_k),
        "reciprocal_rank": reciprocal_rank(results, relevant_ids),
        "ndcg_at_k": ndcg_at_k(results, relevant_ids, top_k),
    }


def _to_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


"""
Example usage inside your notebook after config_a_chromadb and config_b_chromadb exist:

configs = {
    "A": RetrievalConfig(
        label="Config A - <model A name>",
        collection=config_a_chromadb,
        query_model_name="<exact model used to create config A embeddings>",
    ),
    "B": RetrievalConfig(
        label="Config B - <model B name>",
        collection=config_b_chromadb,
        query_model_name="<exact model used to create config B embeddings>",
    ),
}

query_models = load_query_models(configs)
reranker_model = load_reranker()

evaluation = evaluate_fixed_pipeline(
    ground_truth=GROUND_TRUTH,
    configs=configs,
    query_models=query_models,
    reranker_model=reranker_model,
    candidate_k=50,
    top_k=10,
)

summary = summarize_evaluation(evaluation)
print(evaluation)
print(summary)

single_query_results = run_fixed_pipeline_for_query(
    query="help me write a professional email to a client",
    configs=configs,
    query_models=query_models,
    reranker_model=reranker_model,
    candidate_k=50,
    top_k=10,
)
"""
