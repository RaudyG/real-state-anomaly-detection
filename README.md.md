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

The model surfaces properties with anomalous price-to-feature ratios in the common home segment. Output includes:

- `anomaly_score` — the lower the score, the more anomalous the record
- `price_per_sqft` — key signal for overpricing relative to size
- City and state context for geographic analysis

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
