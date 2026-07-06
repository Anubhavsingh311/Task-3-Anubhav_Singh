# Customer Segmentation Blueprint
### From Unlabeled Data to Actionable Personas Using Unsupervised Learning

**DecodeLabs Data Science Internship — Project 3**  
**Author:** Anubhav Singh · [@Anubhavsingh311](https://github.com/Anubhavsingh311)

---

## Objective

Modern enterprise databases generate raw transactional signals containing hidden behavioral patterns invisible to the naked eye. This project builds a complete unsupervised learning pipeline that converts 20 engineered retail features into mathematically isolated, strategically actionable customer segments — each translated into a named business persona with a marketing playbook.

---

## Pipeline Architecture (IPO Framework)

```
Raw Data (14 cols)
       │
       ▼
Feature Engineering ──► 20 Dimensions
       │
       ▼
  SCALE ──► StandardScaler  (z = (x − μ) / σ)
       │
       ▼
COMPRESS ──► PCA             (95% variance → 13 components)
       │
       ▼
 CLUSTER ──► K-Means++       (K proven by Elbow + Silhouette)
       │
       ▼
TRANSLATE ──► Business Personas  (centroids inverse-transformed)
```

---

## Dataset

- **Source:** Retail transactional order data
- **Records:** 1,200 orders
- **Raw Columns:** 14 (OrderID, Date, CustomerID, Product, Quantity, UnitPrice, ShippingAddress, PaymentMethod, OrderStatus, TrackingNumber, ItemsInCart, CouponCode, ReferralSource, TotalPrice)
- **Engineered to:** 20 features (behavioral, temporal, encoded categoricals)

---

## Methodology

### Phase 1 — Exploratory Data Analysis
- Shape, types, missing values, duplicates
- Histograms and boxplots for all numerical features
- Correlation heatmap across all 20 engineered features

### Phase 2 — Preprocessing & Feature Engineering
20 features derived from 14 raw columns:

| Category | Features |
|----------|----------|
| Original numeric | Quantity, UnitPrice, ItemsInCart, TotalPrice |
| Derived behavioral | revenue_per_item, price_gap, discount_ratio, cart_to_order_ratio |
| Binary flags | has_coupon, is_weekend, is_high_value, price_tier |
| Temporal | order_month, order_dayofweek, order_quarter, order_year |
| Encoded categoricals | payment_encoded, status_encoded, referral_encoded, product_encoded |

**Standardization:** `z = (x − μ) / σ` via `StandardScaler` — required because K-Means uses Euclidean distance (`d(p,q) = √Σ(pᵢ−qᵢ)²`) and is sensitive to feature magnitude.

### Phase 3 — Principal Component Analysis (PCA)
- **Problem:** Curse of Dimensionality — in D > 20 spaces, all points become equidistant, breaking Euclidean distance
- **Solution:** PCA finds orthogonal axes of maximum variance (Eigenvalue equation: Σv = λv)
- **Threshold:** 95% cumulative explained variance → **13 Principal Components**
- **Visualization:** 2D scatter on PC1 vs PC2

### Phase 4 — Choosing K (Two Diagnostic Gatekeepers)

**Gatekeeper 1 — Elbow Method (WCSS):**
- WCSS = Σ‖xᵢ − μₖ‖² plotted for K = 2 to 10
- Elbow detected at **K = 5** via KneeLocator algorithm

**Gatekeeper 2 — Silhouette Score:**
- `s(i) = (b(i) − a(i)) / max(a(i), b(i))`
- Scores computed for K = 2 to 10
- Confirms cluster structure quality

**Final K = 5** — Elbow method identifies the structural inflection point where adding clusters yields diminishing behavioral insight on this transactional dataset.

### Phase 5 — K-Means Clustering
- Algorithm: `KMeans(n_clusters=5, init='k-means++', n_init=20, random_state=42)`
- K-Means++ initialization prevents poor centroid placement
- 20 independent runs ensure global convergence

### Phase 6 & 7 — Visualization & Cluster Analysis
- Final scatter plot: PC1 vs PC2, colored by cluster, centroids marked
- Mean profile per cluster: TotalPrice, UnitPrice, Quantity, ItemsInCart, revenue_per_item, coupon usage
- Bar chart comparisons across all 5 segments

---

## Results — The 5 Customer Personas

Centroids are inverse-transformed back to original feature space (Cₒᵣᵢ = (Cₛ꜀ₐₗₑ꜀ · σ) + μ) to produce interpretable human-centric metrics.

| Cluster | Persona | Size | Avg Order | Key Signal |
|---------|---------|------|-----------|------------|
| 0 | **Budget Shoppers** | 261 (21.8%) | $555 | Highest coupon use (78%), prefer low-cost products |
| 1 | **Selective Spenders** | 43 (3.6%) | $1,974 | High-value bulk buyers, social proof driven |
| 2 | **Casual Browsers** | 341 (28.4%) | $573 | Largest segment, high cart ratio, conversion risk |
| 3 | **Premium Power Buyers** | 274 (22.8%) | $2,242 | 99% high-value orders, top revenue segment |
| 4 | **Focused Minimalists** | 281 (23.4%) | $802 | Single-item precision buyers, high return rate |

### Strategic Actions

**Budget Shoppers (C0):** Flash sales, volume discounts, loyalty reward points, triggered email campaigns with coupon codes.

**Selective Spenders (C1):** Referral programs, bundle deals, early access to new arrivals, VIP tier enrollment.

**Casual Browsers (C2):** Retargeting campaigns, abandoned cart emails, social proof (reviews/ratings), limited-time offers.

**Premium Power Buyers (C3):** Dedicated account management, exclusive loyalty tier, early product launches, enterprise warranty packages.

**Focused Minimalists (C4):** Detailed product specs, comparison guides, hassle-free returns, post-purchase follow-up, accessory upsells.

---

## Visualizations

| File | Description |
|------|-------------|
| `images/missing_values.png` | Missing value bar chart |
| `images/distributions.png` | Histograms + boxplots for numerical features |
| `images/correlation_heatmap.png` | 20×20 feature correlation heatmap |
| `images/scaling_comparison.png` | Before vs after StandardScaler |
| `images/explained_variance.png` | Individual + cumulative PCA variance |
| `images/pca_scatter.png` | PC1 vs PC2 scatter (pre-clustering) |
| `images/elbow_method.png` | WCSS elbow curve, K=2 to 10 |
| `images/silhouette_score.png` | Silhouette scores, K=2 to 10 |
| `images/customer_segments.png` | Final cluster plot with centroids |
| `images/cluster_profiles.png` | Comparative bar charts per cluster |
| `images/persona_map.png` | Final labeled persona scatter map |

---

## Repository Structure

```
Customer-Segmentation-Unsupervised-ML/
│
├── Project3.ipynb                  ← Full analysis notebook (executed)
├── Dataset_for_Data_Analytics_Sheet1.csv
├── segmented_customers.csv         ← CustomerID + Cluster + Persona labels
├── requirements.txt
├── .gitignore
├── README.md
│
└── images/
    ├── missing_values.png
    ├── distributions.png
    ├── correlation_heatmap.png
    ├── scaling_comparison.png
    ├── explained_variance.png
    ├── pca_scatter.png
    ├── elbow_method.png
    ├── silhouette_score.png
    ├── customer_segments.png
    ├── cluster_profiles.png
    └── persona_map.png
```

---

## Setup & Reproduction

```bash
# 1. Clone the repository
git clone https://github.com/Anubhavsingh311/Customer-Segmentation-Unsupervised-ML.git
cd Customer-Segmentation-Unsupervised-ML

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook Project3.ipynb
```

---

## Tools & Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.11+ | Core runtime |
| pandas | 2.x | Data manipulation |
| numpy | 1.x | Numerical operations |
| scikit-learn | 1.x | PCA, KMeans, StandardScaler, silhouette_score |
| matplotlib | 3.x | Visualizations |
| seaborn | 0.x | Heatmaps and styled plots |
| kneed | 0.x | Automatic elbow detection |

---

## Key Concepts Demonstrated

- Unsupervised learning pipeline design (Scale → Compress → Cluster → Translate)
- PCA for dimensionality reduction and curse of dimensionality mitigation
- K-Means++ clustering with mathematical K validation
- Elbow Method and Silhouette Score as diagnostic gatekeepers
- Centroid inverse-transformation for business persona extraction
- Feature engineering from raw transactional data

---

*DecodeLabs Data Science Internship · www.decodelabs.tech*
