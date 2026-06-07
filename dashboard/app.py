"""Interactive Plotly Dash dashboard for customer segmentation outputs."""
from __future__ import annotations

from pathlib import Path

import dash
import pandas as pd
import plotly.express as px
from dash import Input, Output, dcc, html

ROOT = Path(__file__).resolve().parents[1]
customers = pd.read_csv(ROOT / "outputs" / "customer_segments_full.csv")
segments = sorted(customers["Assigned_Segment"].dropna().unique())
app = dash.Dash(__name__)
app.title = "Customer Segmentation Dashboard"
app.layout = html.Div([
    html.H1("Customer Segmentation Dashboard"),
    html.Label("Customer segment"),
    dcc.Dropdown(id="segment-filter", options=[{"label": s, "value": s} for s in segments], value=segments, multi=True),
    html.Div(id="kpi-cards"),
    dcc.Graph(id="spending-chart"),
    dcc.Graph(id="demographic-chart"),
    dcc.Graph(id="cluster-chart"),
    dcc.Graph(id="category-chart"),
], style={"fontFamily": "Arial, sans-serif", "margin": "24px"})


@app.callback(
    Output("kpi-cards", "children"),
    Output("spending-chart", "figure"),
    Output("demographic-chart", "figure"),
    Output("cluster-chart", "figure"),
    Output("category-chart", "figure"),
    Input("segment-filter", "value"),
)
def update_dashboard(selected):
    """Refresh dashboard views for selected segments."""
    filtered = customers[customers["Assigned_Segment"].isin(selected)] if selected else customers
    revenue = filtered["Annual_Spending"].sum()
    avg_spend = filtered["Annual_Spending"].mean()
    avg_frequency = filtered["Purchase_Frequency"].mean()
    card = {"padding": "16px", "border": "1px solid #ddd", "borderRadius": "6px", "minWidth": "180px"}
    cards = html.Div([
        html.Div([html.H3(f"{len(filtered):,}"), html.P("Customers")], style=card),
        html.Div([html.H3(f"USD {revenue:,.0f}"), html.P("Annual Spending")], style=card),
        html.Div([html.H3(f"USD {avg_spend:,.0f}"), html.P("Avg Spending")], style=card),
        html.Div([html.H3(f"{avg_frequency:.1f}"), html.P("Avg Frequency")], style=card),
    ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"})
    spending = px.box(filtered, x="Assigned_Segment", y="Annual_Spending", color="Assigned_Segment", title="Spending by Segment")
    demographics = px.histogram(filtered, x="Age", color="Assigned_Segment", nbins=30, title="Age Distribution")
    cluster = px.scatter(filtered, x="Income", y="Annual_Spending", color="Assigned_Segment", size="Purchase_Frequency", hover_data=["Customer_ID", "Product_Category_Preference"], title="Income vs Spending Cluster View")
    category = px.bar(filtered.groupby(["Assigned_Segment", "Product_Category_Preference"], as_index=False)["Customer_ID"].count(), x="Product_Category_Preference", y="Customer_ID", color="Assigned_Segment", title="Product Preference by Segment", labels={"Customer_ID": "Customers"})
    return cards, spending, demographics, cluster, category


if __name__ == "__main__":
    app.run_server(debug=True)
