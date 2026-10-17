
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path("data/cleaned_questions.csv")
CACHE_PATH = Path("data/semantic_embeddings.npy")
OUTPUT_PATH = Path("data/manual_evaluation.csv")

df = pd.read_csv(DATA_PATH)

documents = (
    "Problem title: " + df["Title"].fillna("").astype(str)
    + "\nProblem statement: "
    + df["clean_statement"].fillna("").astype(str)
).tolist()

print("Building TF-IDF index...")
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=30000,
    ngram_range=(1, 2),
    sublinear_tf=True
)
tfidf_matrix = vectorizer.fit_transform(documents)

print("Loading semantic embeddings...")
embeddings = np.load(CACHE_PATH)

queries = [
    "Find the shortest path between vertices in a weighted graph",
    "Sort an array of integers efficiently",
    "Use dynamic programming to maximize profit",
    "Find the lowest common ancestor of two nodes in a tree",
    "Count divisors of a number",
]

records = []

for query in queries:
    tfidf_scores = cosine_similarity(
        vectorizer.transform([query]), tfidf_matrix
    ).flatten()

    # Model embeddings are already normalized.
    # Normalize the query embedding before taking dot products.
    from sentence_transformers import SentenceTransformer

    # Load the model only once, below the import.
    if "model" not in globals():
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    query_embedding = model.encode(
        [query], normalize_embeddings=True
    )[0]
    semantic_scores = embeddings @ query_embedding

    rankings = {
        "TF-IDF": tfidf_scores.argsort()[::-1][:5],
        "Semantic": semantic_scores.argsort()[::-1][:5],
    }

    for method, indices in rankings.items():
        for rank, idx in enumerate(indices, start=1):
            row = df.iloc[idx]
            records.append({
                "query": query,
                "method": method,
                "rank": rank,
                "QCode": row.get("QCode", ""),
                "title": row["Title"],
                "tags": row.get("Tags", ""),
                "link": row.get("link", ""),
                "relevant": ""
            })

pd.DataFrame(records).to_csv(OUTPUT_PATH, index=False)

print(f"\nEvaluation template saved to: {OUTPUT_PATH}")
print(f"Total results to judge: {len(records)}")
print("Open the CSV in Excel and fill the relevant column with 1 or 0.")
