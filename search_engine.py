
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

problems = [
    {
        "title": "Two Sum",
        "statement": "Find two numbers in an array whose sum equals a target."
    },
    {
        "title": "Pair with Given Sum",
        "statement": "Find a pair of array elements that add up to a given value."
    },
    {
        "title": "Shortest Path",
        "statement": "Find the shortest path between two nodes in a graph."
    },
    {
        "title": "Array Sorting",
        "statement": "Sort an array of integers in ascending order."
    },
    {
        "title": "Maximum Subarray",
        "statement": "Find the contiguous subarray with the maximum possible sum."
    }
]

statements = [problem["statement"] for problem in problems]

vectorizer = TfidfVectorizer(stop_words="english")
problem_vectors = vectorizer.fit_transform(statements)


def search_similar_problems(query, top_k=3):
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(query_vector, problem_vectors)[0]

    ranked_indices = scores.argsort()[::-1]

    for index in ranked_indices[:top_k]:
        print(
            f"{problems[index]['title']} "
            f"| Similarity: {scores[index]:.3f}"
        )


if __name__ == "__main__":
    query = input("Enter a programming problem: ")
    print("\nMost similar problems:")
    search_similar_problems(query)
