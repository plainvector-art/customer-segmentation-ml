# Executive Summary: Customer Segmentation Using Machine Learning

## Objective
Develop a customer segmentation solution that groups retail customers by demographics, purchase behavior, value, engagement, and RFM signals.

## Methodology
Generated a 6,000-customer synthetic dataset, cleaned missing values and duplicates, compared IQR and Z-score outliers, engineered value/recency/engagement/RFM features, scaled model features with StandardScaler, and compared K-Means, hierarchical clustering, and DBSCAN. Scaling is necessary because distance-based models can otherwise be dominated by high-magnitude fields such as income and annual spending.

## Model Selection
Selected **K-Means** with **4** clusters. K=2 had the highest raw silhouette score, but K=4 was selected from the post-binary candidate range because it preserves stronger marketing interpretability while retaining acceptable separation and compactness.

## Model Comparison
| model | n_clusters | noise_points | silhouette | davies_bouldin | calinski_harabasz |
| --- | --- | --- | --- | --- | --- |
| K-Means | 4 | 0 | 0.2924476590872519 | 1.1276100498179953 | 3058.6016047520147 |
| Hierarchical | 4 | 0 | 0.2713980185920208 | 1.0542741620712963 | 2662.118288179992 |
| DBSCAN | 1 | 18 | nan | nan | nan |

## Segment Profiles
| Cluster | Segment_Name | Segment_Size | Avg_Age | Avg_Income | Avg_Annual_Spending | Avg_Purchase_Frequency | Avg_Average_Order_Value | Avg_Last_Purchase_Days | Top_Category | Top_RFM_Segment | Recommended_Strategy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | At-Risk Customers | 161 | 40.98 | 62338.95 | 2405.76 | 2.28 | 1134.9 | 196.07 | Fashion | Low Value | Win-back journeys, replenishment reminders, and limited-time reactivation offers. |
| 1 | Occasional Buyers | 2535 | 33.3 | 80282.62 | 5029.27 | 25.31 | 215.52 | 59.8 | Fashion | Loyal | Nurture with category recommendations, seasonal offers, and frequency-building campaigns. |
| 2 | Budget Shoppers | 2105 | 37.43 | 52729.47 | 1790.42 | 10.78 | 195.66 | 168.23 | Fashion | Low Value | Targeted discounts, value bundles, and price-led promotional campaigns. |
| 3 | Premium Loyalists | 1199 | 40.74 | 130877.07 | 9313.21 | 38.57 | 255.19 | 28.34 | Fashion | Champions | VIP rewards, early access, premium bundles, and referral incentives. |

## Actionable Insights
1. Occasional Buyers generates the largest revenue share at 45.4% of annual spending.
2. Customers aged (34, 44] contribute the highest revenue share at 30.9%.
3. Premium Loyalists has the highest average annual spending at USD 9,313.
4. Premium Loyalists purchases most often, averaging 38.6 purchases per year.
5. At-Risk Customers has the longest average recency window at 196 days and should receive win-back messaging.
6. Occasional Buyers is the largest audience, representing 42.2% of customers.
7. Premium Loyalists is the strongest online segment and is best suited for app, email, and web personalization campaigns.
8. At-Risk Customers has the highest average order value at USD 1,135, making it attractive for bundles and premium assortments.
9. At-Risk Customers has the lowest purchase frequency, so frequency-building offers can lift revenue without changing acquisition spend.
10. RFM comparison shows Premium Loyalists has the highest concentration of Champion customers, validating it as a priority retention segment.

## Recommendations
Prioritize VIP retention for premium customers, digital journeys for online-heavy customers, win-back campaigns for stale customers, and value-led promotions for budget shoppers. Retrain monthly or when customer mix changes materially.
