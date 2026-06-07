"""Clustering model training and evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


@dataclass(frozen=True)
class ModelResult:
    """Stores fitted clustering output."""

    name: str
    labels: np.ndarray
    model: Any
    metrics: dict[str, float]


def evaluate_clusters(features: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:
    """Calculate standard clustering quality metrics."""
    valid = [label for label in set(labels) if label != -1]
    if len(valid) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    mask = labels != -1
    x = features.loc[mask]
    y = labels[mask]
    return {
        "silhouette": float(silhouette_score(x, y)),
        "davies_bouldin": float(davies_bouldin_score(x, y)),
        "calinski_harabasz": float(calinski_harabasz_score(x, y)),
    }


def kmeans_diagnostics(features: pd.DataFrame, k_range: range = range(2, 11), random_state: int = 42) -> pd.DataFrame:
    """Calculate inertia and silhouette scores for candidate K values."""
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = model.fit_predict(features)
        rows.append({"k": k, "inertia": float(model.inertia_), "silhouette": float(silhouette_score(features, labels))})
    return pd.DataFrame(rows)


def fit_kmeans(features: pd.DataFrame, n_clusters: int, random_state: int = 42) -> ModelResult:
    """Fit K-Means."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = model.fit_predict(features)
    return ModelResult("K-Means", labels, model, evaluate_clusters(features, labels))


def fit_hierarchical(features: pd.DataFrame, n_clusters: int) -> ModelResult:
    """Fit agglomerative hierarchical clustering."""
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(features)
    return ModelResult("Hierarchical", labels, model, evaluate_clusters(features, labels))


def hierarchical_linkage_sample(features: pd.DataFrame, sample_size: int = 500, random_state: int = 42):
    """Create a sampled linkage matrix for dendrogram visualization."""
    return linkage(features.sample(min(sample_size, len(features)), random_state=random_state), method="ward")


def fit_dbscan(features: pd.DataFrame, eps: float = 2.4, min_samples: int = 18) -> ModelResult:
    """Fit DBSCAN as a density-based benchmark."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(features)
    return ModelResult("DBSCAN", labels, model, evaluate_clusters(features, labels))


def choose_best_model(results: list[ModelResult]) -> ModelResult:
    """Select the model with the highest valid silhouette score."""
    valid = [result for result in results if not np.isnan(result.metrics["silhouette"])]
    if not valid:
        raise ValueError("No valid clustering model produced at least two clusters.")
    return max(valid, key=lambda result: result.metrics["silhouette"])


def metrics_table(results: list[ModelResult]) -> pd.DataFrame:
    """Create a model comparison table."""
    rows = []
    for result in results:
        rows.append({
            "model": result.name,
            "n_clusters": int(len(set(result.labels)) - (1 if -1 in result.labels else 0)),
            "noise_points": int((result.labels == -1).sum()),
            **result.metrics,
        })
    return pd.DataFrame(rows).sort_values("silhouette", ascending=False)


def pca_projection(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Project standardized features into two dimensions."""
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(features)
    return pd.DataFrame({
        "PCA1": components[:, 0], "PCA2": components[:, 1], "Cluster": labels,
        "Explained_Variance_PC1": pca.explained_variance_ratio_[0],
        "Explained_Variance_PC2": pca.explained_variance_ratio_[1],
    }, index=features.index)
