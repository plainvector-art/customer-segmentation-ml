# Customer Segmentation Using Machine Learning

## Project Overview
This project builds an end-to-end customer segmentation solution for a retail business. It groups customers using demographic attributes, purchase behavior, engineered value metrics, and RFM analysis so marketing teams can design targeted campaigns.

## Business Problem
A retail company has customer transaction and demographic data but lacks a reliable way to group customers for personalized marketing. The solution identifies meaningful customer segments, discovers high-value customers, and supports targeted promotions and retention strategies.

## Dataset Description
The project generates a realistic synthetic dataset with 6,000 base customers and intentional data quality issues for cleaning demonstrations.

Fields include demographics (`Customer_ID`, `Age`, `Gender`, `Income`, `Marital_Status`, `Education`, `Occupation`, `Location`) and purchase behavior (`Annual_Spending`, `Purchase_Frequency`, `Average_Order_Value`, `Product_Category_Preference`, `Last_Purchase_Days`, `Online_Purchases`, `Store_Purchases`).

## Installation Steps
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run
```bash
python src/run_project.py
```

Launch the dashboard after outputs are created:

```bash
python dashboard/app.py
```

Open the Dash URL printed in the terminal, usually `http://127.0.0.1:8050`.

## Methodology
1. Generate realistic customer data.
2. Detect and handle missing values.
3. Identify and remove duplicate records.
4. Compare IQR and Z-score outlier detection.
5. Perform demographic, purchasing, product, and correlation EDA.
6. Engineer customer value, purchase intensity, recency, engagement, and RFM features.
7. Scale features with `StandardScaler` because distance-based clustering is sensitive to feature magnitude.
8. Train and compare K-Means, hierarchical clustering, and DBSCAN.
9. Evaluate models with silhouette score, Davies-Bouldin index, and Calinski-Harabasz score.
10. Profile segments and generate business recommendations.

## Results
Outputs include `customer_segments.csv`, `customer_segments_full.csv`, `cluster_profiles.csv`, `model_comparison.csv`, `outlier_comparison.csv`, `reports/executive_summary.md`, and `reports/business_insights.txt`.

## Visual Examples
Charts are saved in `outputs/charts/`: histograms, bar charts, box plots, violin plots, scatter plots, pair plot, correlation heatmap, elbow curve, silhouette analysis, dendrogram, PCA plot, and segment comparison charts.

## Business Impact
The project helps teams identify high-value loyal customers, discover at-risk customers, separate price-sensitive shoppers from premium customers, build digital-first experiences, and improve campaign ROI through differentiated messaging.

## Project Structure
```text
Customer-Segmentation/
├── data/customer_data.csv
├── dashboard/app.py
├── notebooks/Customer_Segmentation.ipynb
├── src/data_preprocessing.py
├── src/clustering.py
├── src/visualization.py
├── src/insights.py
├── src/run_project.py
├── outputs/charts/
├── outputs/reports/
├── README.md
├── requirements.txt
└── presentation_summary.md
```

## Code Quality
The code follows a modular architecture with type hints, docstrings, logging, reusable functions, and file-loading error handling.
