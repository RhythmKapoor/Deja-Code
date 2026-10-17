
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "data/cleaned_questions.csv"


def main():
    print("Loading cleaned programming problems...")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["QCode", "Title", "clean_statement"]).copy()

    # Normalize required fields
    for column in ["QCode", "Title", "clean_statement"]:
        df[column] = df[column].astype(str).str.strip()

    df = df[
        (df["QCode"] != "") &
        (df["Title"] != "") &
        (df["clean_statement"] != "")
    ].reset_index(drop=True)

    print(f"Loaded {len(df)} problems.")

    # Convert statements into TF-IDF vectors
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    
    search_text = (
        df["Title"].fillna("").astype(str)
        + " "
        + df["clean_statement"].fillna("").astype(str)
    )

    problem_vectors = vectorizer.fit_transform(search_text)


    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print("Search engine ready!")

    while True:
        query = input(
            "\nEnter a programming problem (or type 'exit'): "
        ).strip()

        if query.lower() == "exit":
            print("Search session ended.")
            break

        if not query:
            print("Please enter a non-empty query.")
            continue

        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(
            query_vector, problem_vectors
        ).ravel()

        ranked_indices = scores.argsort()[::-1]
        ranked_indices = [
            i for i in ranked_indices if scores[i] > 0
        ][:5]

        if not ranked_indices:
            print("No meaningful word overlap found.")
            continue

        print("\nTOP 5 MATCHING PROBLEMS")
        print("=" * 45)

        for rank, index in enumerate(ranked_indices, start=1):
            row = df.iloc[index]

            print(f"\n{rank}. {row['Title']}")
            print(f"   ID: {row['QCode']}")
            print(f"   Similarity: {scores[index]:.3f}")

            if "level" in df.columns:
                print(f"   Difficulty: {row['level']}")

            if "Tags" in df.columns:
                print(f"   Tags: {row['Tags']}")

            if "link" in df.columns:
                link = str(row["link"])
                if link.startswith("http://") or link.startswith("https://"):
                    print(f"   Link: {link}")
                elif link and link.lower() != "nan":
                    print(f"   Link: https://www.codechef.com{link}")


if __name__ == "__main__":
    main()
