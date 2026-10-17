
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "cleaned_questions.csv"
CLUSTER_PATH = BASE_DIR / "data" / "clustered_questions.csv"
EMBEDDINGS_PATH = BASE_DIR / "data" / "semantic_embeddings.npy"

st.set_page_config(
    page_title="Déjà Code",
    page_icon="🔎",
    layout="wide",
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["QCode", "Title", "clean_statement"]).copy()
    df = df.reset_index(drop=True)

    if CLUSTER_PATH.exists():
        clustered = pd.read_csv(CLUSTER_PATH)
        if "cluster" in clustered.columns and len(clustered) == len(df):
            df["cluster"] = clustered["cluster"].values

    return df


@st.cache_resource
def build_search_index(texts):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


@st.cache_resource
def load_semantic_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_semantic_embeddings(texts):
    model = load_semantic_model()

    if EMBEDDINGS_PATH.exists():
        cached = np.load(EMBEDDINGS_PATH)
        if cached.shape[0] == len(texts):
            return model, cached

    embeddings = model.encode(
        list(texts),
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    np.save(EMBEDDINGS_PATH, embeddings)
    return model, embeddings


def parse_tags(value):
    try:
        tags = ast.literal_eval(str(value))
        if isinstance(tags, list):
            return ", ".join(str(tag) for tag in tags)
    except (ValueError, SyntaxError):
        pass

    return str(value) if pd.notna(value) else "Not available"


def problem_url(value):
    link = str(value).strip()
    if link.startswith(("https://", "http://")):
        return link
    if link and link.lower() != "nan":
        return "https://www.codechef.com" + (
            link if link.startswith("/") else "/" + link
        )
    return None


st.title("🔎 Déjà Code")
st.subheader("Discover programming problems by meaning and concepts")
st.write(
    "Search competitive programming problems by describing the task, "
    "explore related challenges, and discover groups of similar problems."
)

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset not found. Check that data/cleaned_questions.csv exists.")
    st.stop()

search_text = (
    df["Title"].fillna("").astype(str)
    + " "
    + df["clean_statement"].fillna("").astype(str)
).tolist()

vectorizer, problem_vectors = build_search_index(tuple(search_text))

tab_search, tab_clusters, tab_about = st.tabs(
    ["🔍 Search Problems", "🧩 Explore Clusters", "ℹ️ About"]
)

with tab_search:
    st.markdown("### Find problems related to your idea")

    query = st.text_area(
        "Describe a programming problem or algorithm idea",
        placeholder=(
            "Example: Find the shortest path between two vertices "
            "in a weighted graph"
        ),
        height=120,
    )

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        search_mode = st.selectbox(
            "Search method",
            ["Semantic", "Hybrid", "TF-IDF"],
            help=(
                "Semantic compares meaning; TF-IDF compares word overlap; "
                "Hybrid combines both."
            ),
        )

    with col2:
        top_k = st.slider("Number of results", 3, 10, 5)

    with col3:
        difficulties = (
            sorted(df["level"].dropna().astype(str).unique())
            if "level" in df.columns
            else []
        )
        selected_difficulties = st.multiselect(
            "Filter by difficulty",
            difficulties,
            default=difficulties,
        )

    with st.expander("Try an example query"):
        examples = [
            "Find the shortest path between vertices in a weighted graph",
            "Find the lowest common ancestor of two nodes in a tree",
            "Sort an array of integers efficiently",
            "Count the divisors of a number",
            "Use dynamic programming to maximize profit",
        ]
        example = st.selectbox("Example problem idea", examples)
        if st.button("Use example"):
            st.session_state["example_query"] = example
            st.rerun()

    if "example_query" in st.session_state:
        query = st.session_state.pop("example_query")
        st.info(f"Example loaded: {query}")

    if st.button("Find similar problems", type="primary"):
        if not query.strip():
            st.warning("Enter a problem description first.")
        else:
            with st.spinner(f"Searching using {search_mode} retrieval..."):
                tfidf_scores = cosine_similarity(
                    vectorizer.transform([query]), problem_vectors
                ).ravel()

                semantic_scores = None

                if search_mode in ["Semantic", "Hybrid"]:
                    try:
                        model, embeddings = load_semantic_embeddings(
                            tuple(search_text)
                        )
                        query_embedding = model.encode(
                            [query],
                            normalize_embeddings=True,
                            show_progress_bar=False,
                        )
                        semantic_scores = (
                            embeddings @ query_embedding[0]
                        )
                    except Exception as exc:
                        st.error(
                            "Semantic search could not load. "
                            "Check the model installation and internet access."
                        )
                        st.exception(exc)
                        st.stop()

                if search_mode == "TF-IDF":
                    scores = tfidf_scores
                elif search_mode == "Semantic":
                    scores = semantic_scores
                else:
                    # Reciprocal Rank Fusion: combine the two ranked lists.
                    tfidf_order = np.argsort(-tfidf_scores)
                    semantic_order = np.argsort(-semantic_scores)

                    tfidf_ranks = np.empty(len(df), dtype=int)
                    semantic_ranks = np.empty(len(df), dtype=int)

                    tfidf_ranks[tfidf_order] = np.arange(1, len(df) + 1)
                    semantic_ranks[semantic_order] = np.arange(1, len(df) + 1)

                    rrf_k = 60
                    scores = (
                        1 / (rrf_k + tfidf_ranks)
                        + 1 / (rrf_k + semantic_ranks)
                    )

                candidates = df.copy()
                candidates["similarity"] = scores
                candidates["tfidf_score"] = tfidf_scores

                if semantic_scores is not None:
                    candidates["semantic_score"] = semantic_scores

                if "level" in candidates.columns:
                    candidates = candidates[
                        candidates["level"].astype(str).isin(
                            selected_difficulties
                        )
                    ]

                candidates = candidates.sort_values(
                    "similarity", ascending=False
                ).head(top_k)

            if candidates.empty:
                st.info("No matching problems found. Try another query or filter.")
            else:
                st.success(
                    f"Showing {len(candidates)} results using {search_mode} search."
                )

                for rank, (_, row) in enumerate(candidates.iterrows(), 1):
                    with st.container(border=True):
                        st.markdown(f"### {rank}. {row['Title']}")

                        if search_mode == "TF-IDF":
                            score_label = "TF-IDF cosine similarity"
                        elif search_mode == "Semantic":
                            score_label = "Semantic cosine similarity"
                        else:
                            score_label = "RRF fusion score"

                        st.caption(
                            f"Problem ID: {row['QCode']} · "
                            f"{score_label}: {row['similarity']:.4f}"
                        )

                        info_cols = st.columns(3)
                        with info_cols[0]:
                            st.write(
                                f"**Difficulty:** "
                                f"{row.get('level', 'Not available')}"
                            )
                        with info_cols[1]:
                            st.write(f"**TF-IDF:** {row['tfidf_score']:.3f}")
                        with info_cols[2]:
                            if "semantic_score" in row:
                                st.write(
                                    f"**Semantic:** "
                                    f"{row['semantic_score']:.3f}"
                                )

                        st.write(f"**Tags:** {parse_tags(row.get('Tags', ''))}")

                        statement = str(row["clean_statement"])
                        st.write(
                            statement[:650] + ("..." if len(statement) > 650 else "")
                        )

                        url = problem_url(row.get("link", ""))
                        if url:
                            st.markdown(f"[Open original problem]({url})")


with tab_clusters:
    st.markdown("### Explore groups of related problems")

    if "cluster" not in df.columns:
        st.info("Run `python clustering.py` first to generate cluster assignments.")
    else:
        cluster_ids = sorted(df["cluster"].dropna().unique())
        selected_cluster = st.selectbox("Choose a cluster", cluster_ids)

        members = df[df["cluster"] == selected_cluster]
        st.metric("Problems in this cluster", len(members))

        display_columns = ["Title", "QCode"]
        if "level" in members.columns:
            display_columns.append("level")
        if "Tags" in members.columns:
            display_columns.append("Tags")

        st.dataframe(
            members[display_columns].head(100),
            use_container_width=True,
            hide_index=True,
        )

        plot_path = BASE_DIR / "data" / "clusters.png"
        if plot_path.exists():
            st.markdown("### Visualize problem clusters")
            st.image(
                str(plot_path),
                caption="K-Means clusters projected using Truncated SVD",
                use_container_width=True,
            )

        st.caption(
            "Cluster IDs are machine-generated groups, not verified "
            "algorithm categories. Inspect titles and tags to interpret them."
        )


with tab_about:
    st.markdown("### How Déjà Code works")
    st.markdown(
        """
        1. **Preprocessing:** cleans problem statements and removes duplicates.
        2. **TF-IDF baseline:** represents titles and statements using word and
           two-word phrase weights.
        3. **Sentence Transformers:** encodes each problem into a 384-dimensional
           vector intended to capture semantic similarity.
        4. **Cosine similarity:** ranks problems against the user's query.
        5. **Hybrid retrieval:** combines lexical and semantic scores.
        6. **K-Means:** groups problems into eight clusters.
        7. **Truncated SVD:** projects text vectors into two dimensions for
           cluster visualization.
        """
    )

    st.info(
        "Similarity scores indicate relative model similarity, not the "
        "probability that two problems use the same algorithm. Evaluate "
        "retrieval quality with manually judged relevant results."
    )
    st.write(f"**Dataset size:** {len(df):,} problems")
    st.write("**Semantic model:** all-MiniLM-L6-v2")
