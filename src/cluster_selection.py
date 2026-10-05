"""
cluster_selection.py
Justifies the choice of N_CLUSTERS used in model.py by running K-Means
across a range of k values and reporting:
1. Inertia (for the elbow method)
2. Silhouette score (higher = better-separated clusters)

Run this once, look at the printed table + saved chart, and use the
result to set N_CLUSTERS in model.py. Keep the chart for your
technical deep-dive slides — "why 4 clusters" is a question you will
be asked.

Run with:
    python src/cluster_selection.py
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    from .data import load_raw_data, clean_data, compute_rfm_features, build_preprocessor, NUMERIC_FEATURES
except ImportError:
    from data import load_raw_data, clean_data, compute_rfm_features, build_preprocessor, NUMERIC_FEATURES

K_RANGE = range(2, 9)
OUTPUT_CHART_PATH = "cluster_selection.png"


def evaluate_k_range(numeric_data, k_range=K_RANGE):
    """
    Returns a list of dicts: {k, inertia, silhouette} for each k tested.
    Inertia always decreases as k grows — look for the "elbow" where it
    stops dropping sharply. Silhouette score should be interpreted
    alongside it: pick the k near the elbow that also has a
    reasonably high silhouette score (closer to 1 is better,
    0 means overlapping clusters, negative means likely mislabeled).
    """
    results = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(numeric_data)
        sil_score = silhouette_score(numeric_data, labels)
        results.append({"k": k, "inertia": kmeans.inertia_, "silhouette": sil_score})
    return results


def plot_results(results, output_path=OUTPUT_CHART_PATH):
    ks = [r["k"] for r in results]
    inertias = [r["inertia"] for r in results]
    silhouettes = [r["silhouette"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(ks, inertias, marker="o")
    ax1.set_title("Elbow method")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia")

    ax2.plot(ks, silhouettes, marker="o", color="orange")
    ax2.set_title("Silhouette score")
    ax2.set_xlabel("Number of clusters (k)")
    ax2.set_ylabel("Silhouette score")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Chart saved to {output_path}")


def recommend_k(results) -> int:
    """
    Simple heuristic: among k values where silhouette score is within
    5% of the maximum observed, pick the smallest k (simpler model,
    easier to explain, still well-separated). Review this manually —
    don't treat the heuristic as the final answer for your slides.
    """
    max_sil = max(r["silhouette"] for r in results)
    candidates = [r["k"] for r in results if r["silhouette"] >= max_sil * 0.95]
    return min(candidates)


if __name__ == "__main__":
    raw = load_raw_data()
    df = clean_data(raw)
    df = compute_rfm_features(df)

    preprocessor = build_preprocessor()
    preprocessor.fit(df[NUMERIC_FEATURES + [c for c in df.columns if c not in NUMERIC_FEATURES and c != "Churn"]]) \
        if False else None  # placeholder guard; fit numeric scaler directly below

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    numeric_data = scaler.fit_transform(df[NUMERIC_FEATURES])

    results = evaluate_k_range(numeric_data)

    print(f"{'k':>3} | {'inertia':>12} | {'silhouette':>10}")
    for r in results:
        print(f"{r['k']:>3} | {r['inertia']:>12.1f} | {r['silhouette']:>10.3f}")

    plot_results(results)
    best_k = recommend_k(results)
    print(f"\nRecommended k: {best_k} (update N_CLUSTERS in model.py if different)")
