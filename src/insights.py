"""Segment naming, profiling, and business insight generation."""
from __future__ import annotations

import pandas as pd


def assign_segment_names(profile: pd.DataFrame) -> dict[int, str]:
    """Assign readable segment names based on relative cluster metrics."""
    names = {}
    spend75 = profile["Avg_Annual_Spending"].quantile(0.75)
    spend35 = profile["Avg_Annual_Spending"].quantile(0.35)
    freq25 = profile["Avg_Purchase_Frequency"].quantile(0.25)
    freq75 = profile["Avg_Purchase_Frequency"].quantile(0.75)
    rec65 = profile["Avg_Last_Purchase_Days"].quantile(0.65)
    online70 = profile["Avg_Online_Purchases"].quantile(0.70)
    for _, row in profile.iterrows():
        cluster = int(row["Cluster"])
        if row["Avg_Annual_Spending"] >= spend75 and row["Avg_Purchase_Frequency"] >= freq75:
            names[cluster] = "Premium Loyalists"
        elif row["Avg_Last_Purchase_Days"] >= rec65 and row["Avg_Purchase_Frequency"] <= freq25:
            names[cluster] = "At-Risk Customers"
        elif row["Avg_Annual_Spending"] <= spend35:
            names[cluster] = "Budget Shoppers"
        elif row["Avg_Online_Purchases"] >= online70:
            names[cluster] = "Digital Enthusiasts"
        else:
            names[cluster] = "Occasional Buyers"
    return names


def _strategy(name: str) -> str:
    strategies = {
        "Premium Loyalists": "VIP rewards, early access, premium bundles, and referral incentives.",
        "Budget Shoppers": "Targeted discounts, value bundles, and price-led promotional campaigns.",
        "At-Risk Customers": "Win-back journeys, replenishment reminders, and limited-time reactivation offers.",
        "Digital Enthusiasts": "App-first campaigns, personalized recommendations, and online exclusive drops.",
    }
    return strategies.get(name, "Nurture with category recommendations, seasonal offers, and frequency-building campaigns.")


def build_cluster_profiles(data: pd.DataFrame, labels) -> pd.DataFrame:
    """Summarize customer characteristics by cluster."""
    profiled = data.copy()
    profiled["Cluster"] = labels
    profile = profiled.groupby("Cluster").agg(
        Segment_Size=("Customer_ID", "count"),
        Avg_Age=("Age", "mean"),
        Avg_Income=("Income", "mean"),
        Avg_Annual_Spending=("Annual_Spending", "mean"),
        Avg_Purchase_Frequency=("Purchase_Frequency", "mean"),
        Avg_Average_Order_Value=("Average_Order_Value", "mean"),
        Avg_Last_Purchase_Days=("Last_Purchase_Days", "mean"),
        Avg_Online_Purchases=("Online_Purchases", "mean"),
        Avg_Store_Purchases=("Store_Purchases", "mean"),
        Avg_Customer_Value_Score=("Customer_Value_Score", "mean"),
        Top_Category=("Product_Category_Preference", lambda s: s.mode().iloc[0]),
        Top_RFM_Segment=("RFM_Segment", lambda s: s.mode().iloc[0]),
    ).reset_index()
    names = assign_segment_names(profile)
    profile["Segment_Name"] = profile["Cluster"].map(names)
    profile["Recommended_Strategy"] = profile["Segment_Name"].map(_strategy)
    numeric = profile.select_dtypes("number").columns
    profile[numeric] = profile[numeric].round(2)
    return profile[[
        "Cluster", "Segment_Name", "Segment_Size", "Avg_Age", "Avg_Income",
        "Avg_Annual_Spending", "Avg_Purchase_Frequency", "Avg_Average_Order_Value",
        "Avg_Last_Purchase_Days", "Top_Category", "Top_RFM_Segment", "Recommended_Strategy",
    ]]


def attach_segment_names(data: pd.DataFrame, profile: pd.DataFrame, labels) -> pd.DataFrame:
    """Add cluster IDs and readable segment names."""
    segmented = data.copy()
    segmented["Cluster"] = labels
    segmented["Assigned_Segment"] = segmented["Cluster"].map(dict(zip(profile["Cluster"], profile["Segment_Name"])))
    return segmented


def generate_business_insights(segmented: pd.DataFrame, profile: pd.DataFrame) -> list[str]:
    """Generate executive-ready, data-driven insights."""
    total = segmented["Annual_Spending"].sum()
    segment_revenue = segmented.groupby("Assigned_Segment")["Annual_Spending"].sum().sort_values(ascending=False)
    top_revenue_segment = segment_revenue.index[0]
    age_revenue = segmented.assign(Age_Band=pd.cut(segmented["Age"], [17, 24, 34, 44, 54, 65, 100])).groupby("Age_Band", observed=False)["Annual_Spending"].sum()
    top_age_band = age_revenue.idxmax()
    top_spend = profile.sort_values("Avg_Annual_Spending", ascending=False).iloc[0]
    frequent = profile.sort_values("Avg_Purchase_Frequency", ascending=False).iloc[0]
    stale = profile.sort_values("Avg_Last_Purchase_Days", ascending=False).iloc[0]
    largest = profile.sort_values("Segment_Size", ascending=False).iloc[0]
    digital = segmented.groupby("Assigned_Segment")["Online_Purchases"].mean().sort_values(ascending=False).index[0]
    aov = profile.sort_values("Avg_Average_Order_Value", ascending=False).iloc[0]
    low_frequency = profile.sort_values("Avg_Purchase_Frequency").iloc[0]
    rfm = pd.crosstab(segmented["Assigned_Segment"], segmented["RFM_Segment"], normalize="index")
    champion_segment = rfm.get("Champions", pd.Series(dtype=float)).sort_values(ascending=False).index[0]
    return [
        f"{top_revenue_segment} generates the largest revenue share at {segment_revenue.iloc[0] / total * 100:.1f}% of annual spending.",
        f"Customers aged {top_age_band} contribute the highest revenue share at {age_revenue.max() / total * 100:.1f}%.",
        f"{top_spend['Segment_Name']} has the highest average annual spending at USD {top_spend['Avg_Annual_Spending']:,.0f}.",
        f"{frequent['Segment_Name']} purchases most often, averaging {frequent['Avg_Purchase_Frequency']:.1f} purchases per year.",
        f"{stale['Segment_Name']} has the longest average recency window at {stale['Avg_Last_Purchase_Days']:.0f} days and should receive win-back messaging.",
        f"{largest['Segment_Name']} is the largest audience, representing {largest['Segment_Size'] / len(segmented) * 100:.1f}% of customers.",
        f"{digital} is the strongest online segment and is best suited for app, email, and web personalization campaigns.",
        f"{aov['Segment_Name']} has the highest average order value at USD {aov['Avg_Average_Order_Value']:,.0f}, making it attractive for bundles and premium assortments.",
        f"{low_frequency['Segment_Name']} has the lowest purchase frequency, so frequency-building offers can lift revenue without changing acquisition spend.",
        f"RFM comparison shows {champion_segment} has the highest concentration of Champion customers, validating it as a priority retention segment.",
    ]
