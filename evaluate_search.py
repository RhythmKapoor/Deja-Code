
import ast
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "data/cleaned_questions.csv"
K = 5


def parse_tags(value):
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return {str(tag).strip().lower() for tag in parsed}
    except (ValueError, SyntaxError):
        pass
    return set()


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Title", "clean_statement"]).copy()
    df = df.reset_index(drop=True)

    df["tags_set"] = df["Tags"].apply(parse_tags)

    # Each test query has a statement and an expected tag keyword.
    test_queries = [
        ("Find the shortest path between vertices in a graph", "graph"),
        ("Sort an array of integers", "sorting"),
        ("Find a solution using dynamic programming", "dynamic"),
        ("Find properties of a tree and its vertices", "tree"),
        ("Solve a number theory problem involving divisors", "number")
    ]

    # Fit only on the reference collection.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(
        df["Title"].fillna("") + " " + df["clean_statement"].fillna("")
    )

    precision_scores = []
    recall_scores = []

    for query, expected_tag in test_queries:
        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(query_vector, X).ravel()

        ranked = scores.argsort()[::-1]
        relevant = [
            i for i, tags in enumerate(df["tags_set"])
            if any(expected_tag in tag for tag in tags)
        ]

        relevant_set = set(relevant)
        top_k = [
            i for i in ranked
            if scores[i] > 0
        ][:K]

        hits = sum(i in relevant_set for i in top_k)

        precision = hits / K
        recall = hits / len(relevant) if relevant else 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)

        print(f"\nQuery: {query}")
        print(f"Expected tag keyword: {expected_tag}")
        print(f"Relevant dataset items by tag: {len(relevant)}")
        print(f"Relevant items in top {K}: {hits}")
        print(f"Precision@{K}: {precision:.3f}")
        print(f"Recall@{K}: {recall:.3f}")
        print("Retrieved problems:")

        for i in top_k:
            print(f"  - {df.iloc[i]['Title']} (score={scores[i]:.3f})")

    print("\n=== OVERALL RESULTS ===")
    print(f"Mean Precision@{K}: {sum(precision_scores)/len(precision_scores):.3f}")
    print(f"Mean Recall@{K}: {sum(recall_scores)/len(recall_scores):.3f}")
    print("Evaluation uses tags as imperfect relevance labels.")


if __name__ == "__main__":
    main()
