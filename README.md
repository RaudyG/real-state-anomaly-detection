# Real Estate Anomaly Detection — Overpricing in Common Homes

An unsupervised anomaly detection project that identifies suspiciously overpriced standard-size homes in the US real estate market using an Isolation Forest model.

---

## Overview

| | |
|---|---|
| **Dataset** | Realtor.com listings — US real estate market |
| **Technique** | Isolation Forest (unsupervised anomaly detection) |
| **Focus** | Overpriced small/medium homes (≤3 beds, ≤2 baths) |
| **Contamination** | 0.5% — top anomalies only |
| **Stack** | Python · Scikit-learn · Pandas |

---

## Problem Statement

Standard price-per-size comparisons often miss overpricing in common homes because they benchmark against the full market, including luxury properties. This project builds price ratio features (`price_per_sqft`, `price_per_bath`) that expose anomalous pricing patterns regardless of property size — then uses Isolation Forest to isolate the most statistically unusual cases.

This approach is directly transferable to financial fraud detection and credit risk, where anomalies in transaction ratios signal irregular behavior.

---

## Methodology

### 1. Feature engineering
Two ratio features were derived to capture relative overpricing:

- `price_per_sqft` — price divided by house size in square feet
- `price_per_bath` — price divided by number of bathrooms

These ratios normalize for size and expose properties that are expensive relative to what they offer.

### 2. Model configuration
```python
IsolationForest(
    n_estimators=150,    # More trees for finer detection
    contamination=0.005, # Top 0.5% most anomalous records
    random_state=42
)
```

### 3. Post-prediction filter
Anomalies were filtered to **standard-size homes only** (≤3 beds, ≤2 baths) to focus on overpriced common properties — the segment most relevant to average buyers and most susceptible to pricing irregularities.

---

## Results

| Metric | Value |
|---|---|
| Total records processed | 526,514 |
| Anomalies detected (0.5% contamination) | ~2,632 |
| Anomalies in standard-size homes (≤3 beds, ≤2 baths) | **281** |
| Effective anomaly rate in common homes | **0.05%** of full dataset |

The model identified 281 common homes with statistically anomalous price-to-feature ratios. The filter for standard-size properties makes the detection far more selective than the contamination parameter alone — surfacing only the most extreme cases in the segment most relevant to average buyers.

### Top 5 most anomalous cases

| Price | Beds | Baths | Size (sqft) | Price/sqft | City | Anomaly Score |
|---|---|---|---|---|---|---|
| $14,995,000 | 3 | 1 | 1,404 | $10,680 | White Sulphur Springs | -0.0778 |
| $9,154,500 | 3 | 2 | 1,200 | $7,629 | Parmelee | -0.0770 |
| $11,500,000 | 3 | 1 | 773 | $14,877 | Olney | -0.0745 |
| $6,400,000 | 2 | 1 | 672 | $9,524 | Helmville | -0.0739 |
| $16,800,000 | 3 | 2 | 1,976 | $8,502 | Venango | -0.0727 |

**Interpretation:** The most extreme case is a 3-bedroom, 1-bathroom home in White Sulphur Springs priced at $14.9M — $10,680 per square foot. For context, Manhattan averages ~$1,500/sqft. These are not luxury homes by size; they are common homes with anomalous pricing behavior, which is exactly what ratio-based feature engineering is designed to expose.

The `anomaly_score` is the output of `decision_function()` — the more negative the value, the further the record is from the normal distribution of the training data.

---

## Dataset

This project uses the **Realtor.com Real Estate Dataset** available on Kaggle:

🔗 [Download from Kaggle](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset)

Download the file and rename it to `realtor-data.csv` before running the script. The CSV uses `;` as separator.

**Required columns:** `price`, `bed`, `bath`, `acre_lot`, `house_size`, `city`, `state`

---

## Running the project

```bash
pip install pandas numpy scikit-learn
python IsolationForest.py
```

---

## Key insight

Isolation Forest is particularly well-suited for this problem because:
- It does not require labeled data (no need to define what "overpriced" means upfront)
- It scales well to large datasets (`n_jobs=-1` enables parallel processing)
- The `contamination` parameter gives explicit control over how selective the detection is

The same logic applies directly to **financial transaction monitoring**, where ratio-based features (transaction amount vs. account average, frequency vs. historical baseline) are standard practice.

---

## Next Steps

- Add geographic clustering to compare anomaly rates by state and city
- Incorporate time-based features if listing dates are available
- Apply SHAP values to explain why each property was flagged
- Benchmark against DBSCAN and Local Outlier Factor (LOF)

---

## Author

**Raudy Garcia** — Data Analyst at DGII · Dominican Republic  
[LinkedIn](https://www.linkedin.com/in/raudy-garcia-25424721a) · [GitHub](https://github.com/RaudyG)
