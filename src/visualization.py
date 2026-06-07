"""Visualization helpers for EDA, clustering diagnostics, and segment reporting."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_samples

sns.set_theme(style="whitegrid", palette="Set2")


def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_eda_charts(data: pd.DataFrame, output_dir: str | Path) -> None:
    """Create and save exploratory analysis charts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for column in ["Age", "Income", "Annual_Spending"]:
        plt.figure(figsize=(8, 5))
        sns.histplot(data[column], kde=True, bins=35)
        plt.title(column.replace("_", " "))
        _save(output / f"{column.lower()}_distribution.png")
    for column in ["Gender", "Product_Category_Preference"]:
        plt.figure(figsize=(9, 5))
        sns.countplot(data=data, y=column, order=data[column].value_counts().index)
        plt.title(column.replace("_", " "))
        _save(output / f"{column.lower()}_bar.png")
    plt.figure(figsize=(11, 8))
    sns.heatmap(data.select_dtypes("number").corr(), cmap="vlag", center=0)
    plt.title("Correlation Heatmap")
    _save(output / "correlation_heatmap.png")
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=data, x="Gender", y="Annual_Spending")
    plt.title("Annual Spending by Gender")
    _save(output / "spending_boxplot.png")
    plt.figure(figsize=(10, 5))
    sns.violinplot(data=data, x="Location", y="Income", inner="quartile")
    plt.title("Income by Location")
    _save(output / "income_by_location_violin.png")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data.sample(min(1500, len(data)), random_state=42), x="Income", y="Annual_Spending", hue="Product_Category_Preference", s=24, alpha=0.7)
    plt.title("Income vs Spending")
    _save(output / "income_vs_spending.png")
    grid = sns.pairplot(data[["Age", "Income", "Annual_Spending", "Purchase_Frequency", "Last_Purchase_Days"]].sample(min(900, len(data)), random_state=42), corner=True, plot_kws={"s": 12, "alpha": 0.55})
    grid.fig.suptitle("Pair Plot of Core Numeric Features", y=1.02)
    grid.savefig(output / "pair_plot.png", dpi=160)
    plt.close("all")


def save_model_charts(diagnostics: pd.DataFrame, linkage_matrix, features: pd.DataFrame, labels, pca_data: pd.DataFrame, output_dir: str | Path) -> None:
    """Save clustering diagnostic charts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=diagnostics, x="k", y="inertia", marker="o")
    plt.title("K-Means Elbow Curve")
    _save(output / "kmeans_elbow_curve.png")
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=diagnostics, x="k", y="silhouette", marker="o")
    plt.title("K-Means Silhouette by K")
    _save(output / "kmeans_silhouette_scores.png")
    sample = features.sample(min(1200, len(features)), random_state=42)
    sample_labels = pd.Series(labels, index=features.index).loc[sample.index].to_numpy()
    sil = pd.DataFrame({"Cluster": sample_labels, "Silhouette": silhouette_samples(sample, sample_labels)})
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=sil, x="Cluster", y="Silhouette")
    plt.title("Silhouette Analysis by Cluster")
    _save(output / "silhouette_analysis.png")
    plt.figure(figsize=(12, 5))
    dendrogram(linkage_matrix, truncate_mode="lastp", p=30, leaf_rotation=45, leaf_font_size=9)
    plt.title("Hierarchical Clustering Dendrogram")
    _save(output / "hierarchical_dendrogram.png")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_data, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", s=28, alpha=0.75)
    plt.title("PCA 2D Cluster Visualization")
    _save(output / "pca_cluster_plot.png")


def save_segment_charts(segmented: pd.DataFrame, output_dir: str | Path) -> None:
    """Save charts focused on final customer segments."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    sns.countplot(data=segmented, y="Assigned_Segment", order=segmented["Assigned_Segment"].value_counts().index)
    plt.title("Cluster Distribution")
    _save(output / "cluster_distribution.png")
    for y_value, filename in [("Annual_Spending", "spending_by_segment"), ("Income", "income_by_segment"), ("Purchase_Frequency", "purchase_frequency_by_segment")]:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=segmented, x="Assigned_Segment", y=y_value)
        plt.xticks(rotation=25, ha="right")
        plt.title(f"{y_value.replace('_', ' ')} by Segment")
        _save(output / f"{filename}.png")
    plt.figure(figsize=(10, 5))
    sns.violinplot(data=segmented, x="Assigned_Segment", y="Age", inner="quartile")
    plt.xticks(rotation=25, ha="right")
    plt.title("Age by Segment")
    _save(output / "age_by_segment_violin.png")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=segmented, x="Assigned_Segment", y="Customer_Value_Score", errorbar=None)
    plt.xticks(rotation=25, ha="right")
    plt.title("Customer Value Score by Segment")
    _save(output / "customer_value_score_by_segment.png")
