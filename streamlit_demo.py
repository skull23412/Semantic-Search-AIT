from __future__ import annotations

import ast
import html
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
METADATA_PATH = APP_DIR / "embedding_metadata_minilm.csv"
EMBEDDING_PATHS = {
    "Config A: content only": APP_DIR / "embeddings_config_a_minilm.npy",
    "Config B: title + category + tags + content": APP_DIR / "embeddings_config_b_minilm.npy",
}
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


st.set_page_config(
    page_title="LEAF Semantic Search Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.35rem;
    }
    .result {
        border: 1px solid rgba(49, 51, 63, 0.16);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.75rem 0;
        background: rgba(250, 250, 250, 0.7);
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
def load_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


@st.cache_data(show_spinner=False)
def load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
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


def search(query: str, embeddings: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    model = load_encoder()
    query_embedding = model.encode([query], convert_to_numpy=True)[0].astype(np.float32)
    query_embedding = normalize_vector(query_embedding)
    scores = embeddings @ query_embedding
    candidate_count = min(top_k, len(scores))
    top_indices = np.argpartition(scores, -candidate_count)[-candidate_count:]
    ranked_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    return ranked_indices, scores[ranked_indices]


def render_tags(tags: list[str]) -> str:
    return "".join(f'<span class="tag">{html.escape(tag)}</span>' for tag in tags[:6])


def render_result(row: pd.Series, score: float, rank: int) -> None:
    content = row["content_preview"]
    if len(content) > 560:
        content = content[:560].rstrip() + "..."
    title = html.escape(str(row["title"]))
    prompt_id = html.escape(str(row["id"]))
    category = html.escape(str(row["category"]))
    subcategory = html.escape(str(row["subcategory"]))
    content = html.escape(content)

    st.markdown(
        f"""
        <div class="result">
            <div class="result-title">{rank}. {title}</div>
            <div class="result-meta">
                {prompt_id} - {category} / {subcategory} - score {score:.3f}
            </div>
            <div>{content}</div>
            <div style="margin-top:0.7rem;">{render_tags(row['tags_list'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def validate_files() -> None:
    missing = [str(METADATA_PATH.name)] if not METADATA_PATH.exists() else []
    missing.extend(str(path.name) for path in EMBEDDING_PATHS.values() if not path.exists())
    if missing:
        st.error("Missing required files: " + ", ".join(missing))
        st.stop()


def main() -> None:
    validate_files()

    st.title("LEAF Semantic Search Demo")

    with st.sidebar:
        st.subheader("Search Setup")
        config_label = st.radio("Embedding input", list(EMBEDDING_PATHS), index=1)
        top_k = st.slider("Results", min_value=3, max_value=15, value=5, step=1)
        category_filter_enabled = st.checkbox("Filter by category", value=False)

    with st.spinner("Loading metadata and embeddings..."):
        metadata = load_metadata(METADATA_PATH)
        embeddings = load_embeddings(EMBEDDING_PATHS[config_label])

    categories = sorted(metadata["category"].dropna().unique().tolist())
    selected_category = None
    if category_filter_enabled:
        with st.sidebar:
            selected_category = st.selectbox("Category", categories)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Prompts", f"{len(metadata):,}")
    col_b.metric("Vector size", f"{embeddings.shape[1]}")
    col_c.metric("Model", "MiniLM-L6-v2")

    examples = [
        "write a cold outreach email for a SaaS product",
        "review a vendor contract for legal risks",
        "help me structure customer support training materials",
        "generate SQL customer segmentation with window functions",
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

    with st.spinner("Searching..."):
        if selected_category:
            mask = metadata["category"].eq(selected_category).to_numpy()
            filtered_indices = np.flatnonzero(mask)
            filtered_embeddings = embeddings[filtered_indices]
            local_indices, scores = search(query, filtered_embeddings, top_k)
            result_indices = filtered_indices[local_indices]
        else:
            result_indices, scores = search(query, embeddings, top_k)

    st.subheader("Results")
    st.caption(config_label)

    for rank, (idx, score) in enumerate(zip(result_indices, scores), start=1):
        render_result(metadata.iloc[int(idx)], float(score), rank)


if __name__ == "__main__":
    main()
