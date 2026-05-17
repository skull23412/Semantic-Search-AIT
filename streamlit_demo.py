from __future__ import annotations

import ast
import html
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


APP_DIR = Path(__file__).resolve().parent
METADATA_PATH = APP_DIR / "retrieval_metadata_enriched.csv"
EMBEDDING_CONFIGS = {
    "Config A - BGE enriched": {
        "path": APP_DIR / "embeddings_bge_enriched.npy",
        "model": "BAAI/bge-base-en-v1.5",
        "query_prefix": "",
        "passage_prefix": "",
        "note": "Step 3 vector search with enriched prompt metadata embedded by BGE.",
    },
    "Config B - E5 enriched": {
        "path": APP_DIR / "embeddings_e5_enriched.npy",
        "model": "intfloat/e5-base-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "note": "Step 3 vector search with enriched prompt metadata embedded by E5.",
    },
}
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


st.set_page_config(
    page_title="LEAF Retrieval Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        max-width: 1220px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.28rem;
    }
    .result {
        border: 1px solid rgba(49, 51, 63, 0.16);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.75rem 0;
        background: rgba(250, 250, 250, 0.76);
    }
    .result-title {
        font-weight: 700;
        font-size: 1.02rem;
        margin-bottom: 0.25rem;
    }
    .result-meta {
        color: rgba(49, 51, 63, 0.72);
        font-size: 0.88rem;
        margin-bottom: 0.6rem;
        line-height: 1.45;
    }
    .score-line {
        color: rgba(49, 51, 63, 0.84);
        font-size: 0.84rem;
        margin-top: 0.55rem;
    }
    .tag {
        display: inline-block;
        border: 1px solid rgba(49, 51, 63, 0.18);
        border-radius: 999px;
        padding: 0.12rem 0.45rem;
        margin: 0.08rem 0.12rem 0.08rem 0;
        font-size: 0.78rem;
        background: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner=False)
def load_cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL_NAME)


@st.cache_data(show_spinner=False)
def load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column, default in {
        "title": "",
        "content": "",
        "category": "",
        "subcategory": "",
        "tags": "[]",
        "difficulty": "unknown",
        "language": "",
        "target_model": "",
        "likes": 0,
        "upvotes": 0,
    }.items():
        if column not in df.columns:
            df[column] = default

    df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0).astype(int)
    df["upvotes"] = pd.to_numeric(df["upvotes"], errors="coerce").fillna(0).astype(int)
    df["tags_list"] = df["tags"].apply(parse_tags)
    df["content_preview"] = df["content"].fillna("").str.replace(r"\s+", " ", regex=True)
    return df


@st.cache_data(show_spinner=False)
def load_embeddings(path: Path) -> np.ndarray:
    embeddings = np.load(path)
    return normalize_matrix(embeddings.astype(np.float32, copy=False))


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values
    min_val = float(values.min())
    max_val = float(values.max())
    if max_val == min_val:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_val) / (max_val - min_val)


def parse_tags(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(tag) for tag in value]
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(tag) for tag in parsed]


def build_metadata_text(row: pd.Series) -> str:
    return " ".join(
        [
            str(row.get("title", "")),
            str(row.get("category", "")),
            str(row.get("subcategory", "")),
            str(row.get("tags", "")),
            str(row.get("difficulty", "")),
        ]
    )


def vector_search(
    query: str,
    embeddings: np.ndarray,
    encoder,
    query_prefix: str,
    candidate_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    query_embedding = encoder.encode(
        [query_prefix + query],
        convert_to_numpy=True,
    )[0].astype(np.float32)
    query_embedding = normalize_vector(query_embedding)
    scores = embeddings @ query_embedding
    candidate_count = min(candidate_k, len(scores))
    top_indices = np.argpartition(scores, -candidate_count)[-candidate_count:]
    ranked_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    return ranked_indices, scores[ranked_indices]


def rerank_candidates(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    cross_encoder = load_cross_encoder()
    pairs = [[query.strip(), str(candidate["content"])] for candidate in candidates]
    scores = cross_encoder.predict(pairs)

    reranked = []
    for candidate, score in zip(candidates, scores):
        item = dict(candidate)
        item["rerank_score"] = float(score)
        reranked.append(item)

    return sorted(reranked, key=lambda item: item["rerank_score"], reverse=True)[:top_k]


def metadata_semantic_scores(
    query: str,
    candidates: list[dict],
    encoder,
    query_prefix: str,
    passage_prefix: str,
) -> np.ndarray:
    metadata_texts = [passage_prefix + candidate["metadata_text"] for candidate in candidates]
    query_vec = encoder.encode(
        [query_prefix + query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    metadata_vecs = encoder.encode(
        metadata_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return cosine_similarity(query_vec, metadata_vecs).flatten()


def tfidf_keyword_scores(
    query: str,
    candidates: list[dict],
    metadata_weight: float,
    content_weight: float,
) -> np.ndarray:
    metadata_texts = [candidate["metadata_text"] for candidate in candidates]
    content_texts = [str(candidate["content"]) for candidate in candidates]
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))

    metadata_tfidf = vectorizer.fit_transform([query] + metadata_texts)
    metadata_scores = cosine_similarity(metadata_tfidf[0:1], metadata_tfidf[1:]).flatten()

    content_tfidf = vectorizer.fit_transform([query] + content_texts)
    content_scores = cosine_similarity(content_tfidf[0:1], content_tfidf[1:]).flatten()

    return (metadata_weight * metadata_scores) + (content_weight * content_scores)


def metadata_aware_rerank(
    query: str,
    candidates: list[dict],
    encoder,
    top_k: int,
    alpha: float,
    beta: float,
    gamma: float,
    metadata_keyword_weight: float,
    content_keyword_weight: float,
    query_prefix: str,
    passage_prefix: str,
) -> list[dict]:
    reranked = rerank_candidates(query, candidates, top_k=len(candidates))
    reranker_scores = np.array([item["rerank_score"] for item in reranked], dtype=np.float32)
    metadata_scores = metadata_semantic_scores(query, reranked, encoder, query_prefix, passage_prefix)
    keyword_scores = tfidf_keyword_scores(
        query,
        reranked,
        metadata_weight=metadata_keyword_weight,
        content_weight=content_keyword_weight,
    )

    reranker_norm = minmax_normalize(reranker_scores)
    metadata_norm = minmax_normalize(metadata_scores)
    keyword_norm = minmax_normalize(keyword_scores)
    final_scores = (alpha * reranker_norm) + (beta * metadata_norm) + (gamma * keyword_norm)

    for index, item in enumerate(reranked):
        item["rerank_score_norm"] = float(reranker_norm[index])
        item["metadata_score_norm"] = float(metadata_norm[index])
        item["keyword_score_norm"] = float(keyword_norm[index])
        item["final_score"] = float(final_scores[index])

    return sorted(reranked, key=lambda item: item["final_score"], reverse=True)[:top_k]


def rows_to_candidates(rows: pd.DataFrame, indices: np.ndarray, scores: np.ndarray) -> list[dict]:
    candidates = []
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        row = rows.iloc[int(idx)]
        candidates.append(
            {
                "index": int(idx),
                "rank": rank,
                "id": str(row["id"]),
                "title": str(row["title"]),
                "content": str(row["content"]),
                "category": str(row["category"]),
                "subcategory": str(row["subcategory"]),
                "tags": str(row["tags"]),
                "tags_list": row["tags_list"],
                "difficulty": str(row["difficulty"]),
                "language": str(row.get("language", "")),
                "target_model": str(row.get("target_model", "")),
                "likes": int(row["likes"]),
                "upvotes": int(row["upvotes"]),
                "vector_score": float(score),
                "metadata_text": build_metadata_text(row),
            }
        )
    return candidates


def apply_filters(metadata: pd.DataFrame, category: str | None, difficulties: list[str]) -> np.ndarray:
    mask = np.ones(len(metadata), dtype=bool)
    if category:
        mask &= metadata["category"].eq(category).to_numpy()
    if difficulties:
        mask &= metadata["difficulty"].isin(difficulties).to_numpy()
    return mask


def render_tags(tags: list[str]) -> str:
    return "".join(f'<span class="tag">{html.escape(tag)}</span>' for tag in tags[:6])


def render_result(candidate: dict, rank: int, mode: str) -> None:
    content = str(candidate["content"]).replace("\n", " ")
    if len(content) > 620:
        content = content[:620].rstrip() + "..."

    title = html.escape(str(candidate["title"]))
    prompt_id = html.escape(str(candidate["id"]))
    category = html.escape(str(candidate["category"]))
    subcategory = html.escape(str(candidate["subcategory"]))
    difficulty = html.escape(str(candidate["difficulty"]))
    language = html.escape(str(candidate.get("language", "")))
    target_model = html.escape(str(candidate.get("target_model", "")))
    content = html.escape(content)

    score_parts = [f"vector {candidate['vector_score']:.3f}"]
    if "rerank_score" in candidate:
        score_parts.append(f"reranker {candidate['rerank_score']:.3f}")
    if "final_score" in candidate:
        score_parts.append(f"final {candidate['final_score']:.3f}")

    st.markdown(
        f"""
        <div class="result">
            <div class="result-title">{rank}. {title}</div>
            <div class="result-meta">
                {prompt_id} - {category} / {subcategory} - {difficulty}
                - lang {language} - target {target_model}
                - likes {candidate['likes']} - upvotes {candidate['upvotes']}
            </div>
            <div>{content}</div>
            <div style="margin-top:0.7rem;">{render_tags(candidate['tags_list'])}</div>
            <div class="score-line">{html.escape(mode)} - {" | ".join(score_parts)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def validate_files() -> None:
    missing = []
    if not METADATA_PATH.exists():
        missing.append(METADATA_PATH.name)
    missing.extend(
        config["path"].name
        for config in EMBEDDING_CONFIGS.values()
        if not config["path"].exists()
    )
    if missing:
        st.error("Missing required files: " + ", ".join(missing))
        st.stop()


def main() -> None:
    validate_files()

    st.title("LEAF Prompt Retrieval Demo")

    with st.sidebar:
        st.subheader("Pipeline")
        config_label = st.radio("Step 3 embedding config", list(EMBEDDING_CONFIGS), index=0)
        ranking_mode = st.radio(
            "Step 4 ranking",
            [
                "Vector search only",
                "Cross-encoder reranker",
                "Metadata-aware reranker",
            ],
            index=2,
        )
        top_k = st.slider("Results shown", min_value=3, max_value=15, value=5, step=1)
        candidate_k = st.slider("Candidates retrieved", min_value=10, max_value=60, value=30, step=5)

        st.subheader("Metadata-aware weights")
        alpha = st.slider("Reranker weight", 0.0, 1.0, 0.75, 0.05)
        beta = st.slider("Metadata semantic weight", 0.0, 1.0, 0.15, 0.05)
        gamma = st.slider("Keyword weight", 0.0, 1.0, 0.10, 0.05)
        weight_sum = alpha + beta + gamma
        if weight_sum == 0:
            alpha, beta, gamma = 1.0, 0.0, 0.0
        elif abs(weight_sum - 1.0) > 0.001:
            alpha, beta, gamma = alpha / weight_sum, beta / weight_sum, gamma / weight_sum
            st.caption(f"Normalized weights: {alpha:.2f}, {beta:.2f}, {gamma:.2f}")

        metadata_keyword_weight = st.slider("Keyword metadata share", 0.0, 1.0, 0.50, 0.05)
        content_keyword_weight = 1.0 - metadata_keyword_weight

        st.subheader("Filters")
        category_filter_enabled = st.checkbox("Filter by category", value=False)
        difficulty_filter_enabled = st.checkbox("Filter by difficulty", value=False)

    config = EMBEDDING_CONFIGS[config_label]

    with st.spinner("Loading Step 3 assets..."):
        metadata = load_metadata(METADATA_PATH)
        embeddings = load_embeddings(config["path"])
        encoder = load_encoder(config["model"])

    categories = sorted(metadata["category"].dropna().unique().tolist())
    difficulties = sorted(metadata["difficulty"].dropna().unique().tolist())
    selected_category = None
    if category_filter_enabled:
        with st.sidebar:
            selected_category = st.selectbox("Category", categories)

    selected_difficulties = difficulties
    if difficulty_filter_enabled:
        with st.sidebar:
            selected_difficulties = st.multiselect("Difficulty", difficulties, default=difficulties)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Prompts", f"{len(metadata):,}")
    col_b.metric("Vector size", f"{embeddings.shape[1]}")
    col_c.metric("Step 4", ranking_mode)

    st.caption(config["note"])

    examples = [
        "create a social media marketing campaign",
        "write a response to a negative customer review",
        "write a SQL query to analyze database data",
        "help me write a professional email to a client",
    ]

    if "query" not in st.session_state:
        st.session_state.query = examples[0]

    example_cols = st.columns(len(examples))
    for index, example in enumerate(examples):
        if example_cols[index].button(example, use_container_width=True):
            st.session_state.query = example

    query = st.text_input(
        "Query",
        key="query",
        placeholder="Search for a prompt by intent, task, or topic",
    )

    if not query.strip():
        st.info("Enter a query to search the prompt corpus.")
        return

    with st.spinner("Running retrieval pipeline..."):
        mask = apply_filters(metadata, selected_category, selected_difficulties if difficulty_filter_enabled else [])
        if not mask.any():
            st.warning("No prompts match the selected filters.")
            return

        candidate_k = min(candidate_k, int(mask.sum()))
        if mask.all():
            candidate_indices, vector_scores = vector_search(
                query,
                embeddings,
                encoder,
                config["query_prefix"],
                candidate_k,
            )
            candidates = rows_to_candidates(metadata, candidate_indices, vector_scores)
        else:
            filtered_indices = np.flatnonzero(mask)
            filtered_embeddings = embeddings[filtered_indices]
            local_indices, vector_scores = vector_search(
                query,
                filtered_embeddings,
                encoder,
                config["query_prefix"],
                candidate_k,
            )
            candidate_indices = filtered_indices[local_indices]
            candidates = rows_to_candidates(metadata, candidate_indices, vector_scores)

        if ranking_mode == "Vector search only":
            results = candidates[:top_k]
        elif ranking_mode == "Cross-encoder reranker":
            results = rerank_candidates(query, candidates, top_k=top_k)
        else:
            results = metadata_aware_rerank(
                query,
                candidates,
                encoder,
                top_k=top_k,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                metadata_keyword_weight=metadata_keyword_weight,
                content_keyword_weight=content_keyword_weight,
                query_prefix=config["query_prefix"],
                passage_prefix=config["passage_prefix"],
            )

    st.subheader("Results")
    st.caption(f"{config_label} - {ranking_mode}")

    for rank, candidate in enumerate(results, start=1):
        render_result(candidate, rank, ranking_mode)


if __name__ == "__main__":
    main()
