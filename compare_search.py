
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path("data/cleaned_questions.csv")
CACHE_PATH = Path("data/semantic_embeddings.npy")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

df = pd.read_csv(DATA_PATH)

documents = (
    "Problem title: " + df["Title"].fillna("").astype(str)
    + "\nProblem statement: "
    + df["clean_statement"].fillna("").astype(str)
).tolist()

# Method 1: TF-IDF lexical search
print("Building TF-IDF index...")
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=30000,
    ngram_range=(1, 2),
    sublinear_tf=True
)
tfidf_matrix = vectorizer.fit_transform(documents)

# Method 2: Cached semantic embeddings
print("Loading semantic embeddings...")
embeddings = np.load(CACHE_PATH)

model = SentenceTransformer(MODEL_NAME)


def search_tfidf(query, top_k=5):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    return scores.argsort()[::-1][:top_k]


def search_semantic(query, top_k=5):
    query_vector = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ query_vector
    return scores.argsort()[::-1][:top_k]


queries = [
    "Find the shortest path between vertices in a weighted graph",
    "Sort an array of integers efficiently",
    "Use dynamic programming to maximize profit",
    "Find the lowest common ancestor of two nodes in a tree",
    "Count divisors of a number",
]

for query in queries:
    print("\n" + "=" * 75)
    print("QUERY:", query)

    for method_name, search_function in [
        ("TF-IDF", search_tfidf),
        ("SEMANTIC", search_semantic),
    ]:
        print(f"\n--- {method_name}: TOP 5 ---")
        indices = search_function(query)

        for rank, idx in enumerate(indices, start=1):
            row = df.iloc[idx]
            print(f"{rank}. {row['Title']} | Tags: {row.get('Tags', '')}")
