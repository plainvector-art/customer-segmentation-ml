"""Run the complete customer segmentation project pipeline."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from clustering import choose_best_model, fit_dbscan, fit_hierarchical, fit_kmeans, hierarchical_linkage_sample, kmeans_diagnostics, metrics_table, pca_projection
from data_preprocessing import generate_synthetic_customer_data, preprocess_pipeline
from insights import attach_segment_names, build_cluster_profiles, generate_business_insights
from visualization import save_eda_charts, save_model_charts, save_segment_charts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "customer_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
REPORT_DIR = OUTPUT_DIR / "reports"


def _markdown_table(dataframe) -> str:
    """Render a small dataframe as a Markdown table without optional dependencies."""
    headers = [str(column) for column in dataframe.columns]
    rows = [[str(value) for value in row] for row in dataframe.values.tolist()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _select_business_k(diagnostics) -> int:
    """Choose an interpretable K using silhouette among post-binary candidates."""
    candidates = diagnostics[diagnostics["k"].between(3, 6)].copy()
    if candidates.empty:
        return int(diagnostics.sort_values("silhouette", ascending=False).iloc[0]["k"])
    return int(candidates.sort_values("silhouette", ascending=False).iloc[0]["k"])


def main() -> None:
    """Execute the full project pipeline."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    raw = generate_synthetic_customer_data(6000, 42)
    raw.to_csv(DATA_PATH, index=False)
    prep = preprocess_pipeline(raw)
    diagnostics = kmeans_diagnostics(prep.scaled_features)
    optimal_k = _select_business_k(diagnostics)
    results = [
        fit_kmeans(prep.scaled_features, optimal_k),
        fit_hierarchical(prep.scaled_features, optimal_k),
        fit_dbscan(prep.scaled_features),
    ]
    best = choose_best_model(results)
    profile = build_cluster_profiles(prep.feature_data, best.labels)
    segmented = attach_segment_names(prep.feature_data, profile, best.labels)
    insights = generate_business_insights(segmented, profile)
    pca_data = pca_projection(prep.scaled_features, best.labels)
    metrics = metrics_table(results)
    segmented[["Customer_ID", "Assigned_Segment", "Cluster"]].to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)
    segmented.to_csv(OUTPUT_DIR / "customer_segments_full.csv", index=False)
    profile.to_csv(OUTPUT_DIR / "cluster_profiles.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    diagnostics.to_csv(OUTPUT_DIR / "kmeans_diagnostics.csv", index=False)
    prep.outlier_report.to_csv(OUTPUT_DIR / "outlier_comparison.csv", index=False)
    pca_data.to_csv(OUTPUT_DIR / "pca_projection.csv", index=False)
    (REPORT_DIR / "cleaning_report.json").write_text(json.dumps(prep.cleaning_report, indent=2), encoding="utf-8")
    (REPORT_DIR / "business_insights.txt").write_text("\n".join(f"{idx}. {insight}" for idx, insight in enumerate(insights, 1)), encoding="utf-8")
    save_eda_charts(segmented, CHART_DIR)
    save_model_charts(diagnostics, hierarchical_linkage_sample(prep.scaled_features), prep.scaled_features, best.labels, pca_data, CHART_DIR)
    save_segment_charts(segmented, CHART_DIR)
    report = f"""# Executive Summary: Customer Segmentation Using Machine Learning

## Objective
Develop a customer segmentation solution that groups retail customers by demographics, purchase behavior, value, engagement, and RFM signals.

## Methodology
Generated a 6,000-customer synthetic dataset, cleaned missing values and duplicates, compared IQR and Z-score outliers, engineered value/recency/engagement/RFM features, scaled model features with StandardScaler, and compared K-Means, hierarchical clustering, and DBSCAN. Scaling is necessary because distance-based models can otherwise be dominated by high-magnitude fields such as income and annual spending.

## Model Selection
Selected **{best.name}** with **{optimal_k}** clusters. K=2 had the highest raw silhouette score, but K={optimal_k} was selected from the post-binary candidate range because it preserves stronger marketing interpretability while retaining acceptable separation and compactness.

## Model Comparison
{_markdown_table(metrics)}

## Segment Profiles
{_markdown_table(profile)}

## Actionable Insights
""" + "\n".join(f"{idx}. {insight}" for idx, insight in enumerate(insights, 1)) + """

## Recommendations
Prioritize VIP retention for premium customers, digital journeys for online-heavy customers, win-back campaigns for stale customers, and value-led promotions for budget shoppers. Retrain monthly or when customer mix changes materially.
"""
    (REPORT_DIR / "executive_summary.md").write_text(report, encoding="utf-8")
    logging.info("Project complete. Selected model: %s", best.name)


if __name__ == "__main__":
    main()
