"""Data generation, cleaning, outlier analysis, and feature engineering."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)
NUMERIC_COLUMNS = ["Age", "Income", "Annual_Spending", "Purchase_Frequency", "Average_Order_Value", "Last_Purchase_Days", "Online_Purchases", "Store_Purchases"]
CATEGORICAL_COLUMNS = ["Gender", "Marital_Status", "Education", "Occupation", "Location", "Product_Category_Preference"]
MODEL_FEATURES = NUMERIC_COLUMNS + ["Customer_Value_Score", "Purchase_Intensity", "Recency_Score", "Engagement_Score", "RFM_Score"]


@dataclass(frozen=True)
class PreprocessingResult:
    """Container for model-ready data and preprocessing metadata."""

    cleaned_data: pd.DataFrame
    feature_data: pd.DataFrame
    scaled_features: pd.DataFrame
    scaler: StandardScaler
    cleaning_report: dict[str, Any]
    outlier_report: pd.DataFrame


def generate_synthetic_customer_data(n_records: int = 6000, random_state: int = 42) -> pd.DataFrame:
    """Generate a realistic retail customer dataset with five latent customer types."""
    rng = np.random.default_rng(random_state)
    mix = rng.choice(["premium", "mainstream", "budget", "at_risk", "digital"], n_records, p=[0.18, 0.30, 0.22, 0.15, 0.15])
    params = {
        "premium": (42, 10, 135000, 30000, 9500, 2200, 38, 9, 28, 20),
        "mainstream": (36, 11, 78000, 18000, 4600, 1200, 24, 7, 74, 45),
        "budget": (31, 9, 43000, 12000, 1900, 650, 13, 5, 118, 55),
        "at_risk": (48, 12, 62000, 21000, 1300, 700, 7, 4, 245, 90),
        "digital": (29, 7, 90000, 22000, 6200, 1500, 31, 8, 38, 15),
    }
    cats = {
        "premium": ["Electronics", "Home", "Fashion", "Beauty"],
        "mainstream": ["Fashion", "Home", "Groceries", "Sports"],
        "budget": ["Groceries", "Fashion", "Beauty", "Sports"],
        "at_risk": ["Home", "Groceries", "Fashion", "Books"],
        "digital": ["Electronics", "Fashion", "Beauty", "Books"],
    }
    rows = []
    for i, segment in enumerate(mix, 1):
        age_mu, age_sd, inc_mu, inc_sd, spend_mu, spend_sd, freq_mu, freq_sd, rec_mu, rec_sd = params[segment]
        age = int(np.clip(rng.normal(age_mu, age_sd), 18, 75))
        income = float(np.clip(rng.normal(inc_mu, inc_sd), 18000, 240000))
        spending = float(np.clip(rng.normal(spend_mu, spend_sd), 120, 22000))
        frequency = int(np.clip(rng.normal(freq_mu, freq_sd), 1, 80))
        aov = float(np.clip(spending / max(frequency, 1) * rng.normal(1.0, 0.12), 15, 1500))
        recency = int(np.clip(rng.normal(rec_mu, rec_sd), 1, 365))
        online_share = {"premium": 0.45, "mainstream": 0.48, "budget": 0.38, "at_risk": 0.25, "digital": 0.78}[segment]
        online = int(np.clip(round(frequency * rng.normal(online_share, 0.10)), 0, frequency))
        store = int(max(frequency - online + rng.integers(-2, 3), 0))
        rows.append({
            "Customer_ID": f"CUST-{i:05d}", "Age": age,
            "Gender": rng.choice(["Female", "Male", "Non-binary"], p=[0.51, 0.47, 0.02]),
            "Income": round(income, 2),
            "Marital_Status": rng.choice(["Single", "Married", "Divorced", "Widowed"], p=[0.34, 0.49, 0.13, 0.04]),
            "Education": rng.choice(["High School", "Bachelor", "Master", "PhD"], p=[0.27, 0.46, 0.22, 0.05]),
            "Occupation": rng.choice(["Professional", "Manager", "Entrepreneur", "Student", "Retired", "Service", "Technician"]),
            "Location": rng.choice(["Urban", "Suburban", "Rural"], p=[0.55, 0.34, 0.11]),
            "Annual_Spending": round(spending, 2), "Purchase_Frequency": frequency,
            "Average_Order_Value": round(aov, 2),
            "Product_Category_Preference": rng.choice(cats[segment]),
            "Last_Purchase_Days": recency, "Online_Purchases": online, "Store_Purchases": store,
        })
    data = pd.DataFrame(rows)
    for column in ["Income", "Education", "Average_Order_Value", "Product_Category_Preference"]:
        data.loc[rng.choice(data.index, int(n_records * 0.01), replace=False), column] = np.nan
    return pd.concat([data, data.sample(35, random_state=random_state)], ignore_index=True)


def load_data(path: str | Path) -> pd.DataFrame:
    """Load customer data from CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)


def detect_outliers_iqr(data: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    """Count outliers with the interquartile range method."""
    result = {}
    for column in columns:
        q1, q3 = data[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        result[column] = int(((data[column] < q1 - 1.5 * iqr) | (data[column] > q3 + 1.5 * iqr)).sum())
    return result


def detect_outliers_zscore(data: pd.DataFrame, columns: list[str], threshold: float = 3.0) -> dict[str, int]:
    """Count outliers using absolute z-scores."""
    result = {}
    for column in columns:
        std = data[column].std(ddof=0)
        result[column] = 0 if not std else int((((data[column] - data[column].mean()) / std).abs() > threshold).sum())
    return result


def clean_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Clean missing values, remove duplicates, and compare outlier methods."""
    cleaned = data.copy()
    missing_before = cleaned.isna().sum().to_dict()
    duplicates = int(cleaned.duplicated().sum())
    for column in NUMERIC_COLUMNS:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())
    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].fillna(cleaned[column].mode(dropna=True).iloc[0])
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    iqr = detect_outliers_iqr(cleaned, NUMERIC_COLUMNS)
    zscore = detect_outliers_zscore(cleaned, NUMERIC_COLUMNS)
    report = {
        "rows_before": int(len(data)), "rows_after": int(len(cleaned)),
        "duplicates_removed": duplicates, "missing_values_before": missing_before,
        "missing_values_after": cleaned.isna().sum().to_dict(),
    }
    outliers = pd.DataFrame({"feature": NUMERIC_COLUMNS, "iqr_outliers": [iqr[c] for c in NUMERIC_COLUMNS], "zscore_outliers": [zscore[c] for c in NUMERIC_COLUMNS]})
    LOGGER.info("Cleaning report: %s", report)
    return cleaned, report, outliers


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create customer value, engagement, recency, and RFM features."""
    featured = data.copy()
    spend_rank = featured["Annual_Spending"].rank(pct=True)
    freq_rank = featured["Purchase_Frequency"].rank(pct=True)
    aov_rank = featured["Average_Order_Value"].rank(pct=True)
    recency_rank = (366 - featured["Last_Purchase_Days"]).rank(pct=True)
    total_purchases = featured["Online_Purchases"] + featured["Store_Purchases"]
    featured["Customer_Value_Score"] = (0.50 * spend_rank + 0.30 * freq_rank + 0.20 * aov_rank) * 100
    featured["Purchase_Intensity"] = featured["Annual_Spending"] / featured["Purchase_Frequency"].clip(lower=1)
    featured["Recency_Score"] = recency_rank * 100
    featured["Engagement_Score"] = (0.45 * freq_rank + 0.35 * recency_rank + 0.20 * total_purchases.rank(pct=True)) * 100
    featured["RFM_Recency"] = pd.qcut(featured["Last_Purchase_Days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    featured["RFM_Frequency"] = pd.qcut(featured["Purchase_Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    featured["RFM_Monetary"] = pd.qcut(featured["Annual_Spending"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    featured["RFM_Score"] = featured[["RFM_Recency", "RFM_Frequency", "RFM_Monetary"]].sum(axis=1)
    featured["RFM_Segment"] = pd.cut(featured["RFM_Score"], [2, 6, 9, 12, 15], labels=["Low Value", "Developing", "Loyal", "Champions"], include_lowest=True).astype(str)
    return featured


def scale_model_features(data: pd.DataFrame, features: list[str] | None = None) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize numeric model features for distance-based clustering."""
    selected = features or MODEL_FEATURES
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(data[selected]), columns=selected, index=data.index)
    return scaled, scaler


def preprocess_pipeline(data: pd.DataFrame) -> PreprocessingResult:
    """Run all preprocessing steps."""
    cleaned, report, outliers = clean_data(data)
    featured = add_engineered_features(cleaned)
    scaled, scaler = scale_model_features(featured)
    return PreprocessingResult(cleaned, featured, scaled, scaler, report, outliers)
