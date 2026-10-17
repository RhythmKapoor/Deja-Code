
from pathlib import Path

import numpy as np
import pandas as pd

path = Path("data/manual_evaluation.csv")
df = pd.read_csv(path)

df["relevant"] = pd.to_numeric(df["relevant"], errors="coerce")

if df["relevant"].isna().any():
    raise ValueError(
        "Some relevance labels are missing. Fill every relevant cell with 1 or 0."
    )

if not df["relevant"].isin([0, 1]).all():
    raise ValueError("Relevance labels must be 0 or 1.")

def ndcg_at_5(relevances):
    relevances = np.asarray(relevances, dtype=float)
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    dcg = np.sum((2**relevances - 1) / discounts)

    ideal = np.sort(relevances)[::-1]
    idcg = np.sum((2**ideal - 1) / discounts)

    return dcg / idcg if idcg > 0 else 0.0

summary = []

for (query, method), group in df.groupby(["query", "method"]):
    group = group.sort_values("rank")
    labels = group["relevant"].to_numpy()

    summary.append({
        "query": query,
        "method": method,
        "Precision@5": labels.sum() / 5,
        "nDCG@5": ndcg_at_5(labels),
    })

results = pd.DataFrame(summary)

print("\nPer-query results:")
print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\nMean results by method:")
print(
    results.groupby("method")[["Precision@5", "nDCG@5"]]
    .mean()
    .to_string(float_format=lambda x: f"{x:.3f}")
)
