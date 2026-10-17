
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

DATA_PATH = "data/cleaned_questions.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Title", "clean_statement"]).copy()
    df = df.reset_index(drop=True)

    # Represent titles and statements numerically
    text = (
        df["Title"].fillna("").astype(str)
        + " "
        + df["clean_statement"].fillna("").astype(str)
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(text)

    # Group problems into 8 clusters
    model = KMeans(n_clusters=8, random_state=42, n_init=10)
    df["cluster"] = model.fit_predict(X)

    df.to_csv("data/clustered_questions.csv", index=False)

    print("Clustering complete!")
    print("\nCluster sizes:")
    print(df["cluster"].value_counts().sort_index())
    
    print("\nRepresentative problems in each cluster:")

    for cluster_id in sorted(df["cluster"].unique()):
        cluster_rows = df[df["cluster"] == cluster_id]

        print(f"\nCLUSTER {cluster_id} ({len(cluster_rows)} problems)")

        for title in cluster_rows["Title"].head(8):
            print(f" - {title}")


    # Reduce the vectors to two dimensions for visualization
    reducer = TruncatedSVD(n_components=2, random_state=42)
    points = reducer.fit_transform(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        points[:, 0],
        points[:, 1],
        c=df["cluster"],
        s=12,
        alpha=0.7
    )
    plt.title("Déjà Code: Problem Clusters")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig("data/clusters.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
