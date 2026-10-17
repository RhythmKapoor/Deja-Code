
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

DATA_PATH = Path("data/cleaned_questions.csv")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Use both the title and cleaned problem statement.
titles = df["Title"].fillna("").astype(str)
statements = df["clean_statement"].fillna("").astype(str)

documents = (
    "Problem title: " + titles
    + "\nProblem statement: " + statements
).tolist()

print(f"Loaded {len(documents)} problems.")
print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

CACHE_PATH = Path("data/semantic_embeddings.npy")

if CACHE_PATH.exists():
    print("Loading cached embeddings...")
    embeddings = np.load(CACHE_PATH)
else:
    print("Generating embeddings for the dataset...")
    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    np.save(CACHE_PATH, embeddings)
    print("Embeddings saved to", CACHE_PATH)

print("Embeddings generated:", embeddings.shape)



def search_problems(query, top_k=5):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    # Since vectors are normalized, dot product equals cosine similarity.
    scores = embeddings @ query_embedding[0]
    top_indices = scores.argsort()[::-1][:top_k]

    results = df.iloc[top_indices].copy()
    results["similarity"] = scores[top_indices]

    return results


if __name__ == "__main__":
    query = input("\nEnter a programming problem or algorithm idea: ").strip()

    if not query:
        print("Please enter a non-empty query.")
    else:
        results = search_problems(query, top_k=5)

        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            print(f"\n{rank}. {row['Title']}")
            print(f"   Similarity: {row['similarity']:.4f}")
            print(f"   Difficulty: {row.get('level', 'Unknown')}")
            print(f"   Tags: {row.get('Tags', 'Unknown')}")
            print(f"   Link: {row.get('link', 'N/A')}")
            print(
                "   Statement: "
                + str(row["clean_statement"])[:250].replace("\n", " ")
                + "..."
            )
