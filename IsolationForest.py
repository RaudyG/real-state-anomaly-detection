import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# --- 1. Load dataset ---
print("Loading Realtor dataset...")
df = pd.read_csv("realtor-data.csv", sep=';', on_bad_lines='skip')
df.columns = df.columns.str.strip()

# Keep only relevant columns and drop nulls
df_clean = df[['price', 'bed', 'bath', 'acre_lot', 'house_size', 'city', 'state']].dropna().copy()
print(f"Records after cleaning: {df_clean.shape[0]}")

# --- 2. Feature engineering ---
# Price ratios expose overpricing regardless of property size
df_clean['price_per_sqft'] = df_clean['price'] / df_clean['house_size']
df_clean['price_per_bath'] = df_clean['price'] / df_clean['bath']
print("Ratio features created: 'price_per_sqft', 'price_per_bath'")

# --- 3. Define training features ---
features = ['price', 'bed', 'bath', 'acre_lot', 'house_size', 'price_per_sqft', 'price_per_bath']
X = df_clean[features]

# --- 4. Train Isolation Forest ---
print("\nTraining Isolation Forest...")
model = IsolationForest(
    n_estimators=150,     # More trees for finer anomaly detection
    contamination=0.005,  # Target top 0.5% most anomalous records
    random_state=42,
    n_jobs=-1
)
model.fit(X)

# --- 5. Score and label records ---
df_clean['anomaly_score'] = model.decision_function(X)
df_clean['is_anomaly'] = model.predict(X)

# --- 6. Filter: anomalies in standard-size homes only ---
# Focus on small/medium homes (<=3 beds, <=2 baths) to surface overpriced common properties
anomalies = df_clean[
    (df_clean['is_anomaly'] == -1) &
    (df_clean['bed'] <= 3) &
    (df_clean['bath'] <= 2)
]

# --- 7. Display results ---
print("\nTOP 5 OVERPRICED COMMON HOMES (sorted by anomaly score):")
print("-" * 85)
display_cols = ['price', 'bed', 'bath', 'house_size', 'price_per_sqft', 'city', 'anomaly_score']
print(anomalies[display_cols].sort_values(by='anomaly_score').head(5).to_string(index=False))

print(f"\nTotal anomalous common homes detected: {anomalies.shape[0]}")
print(f"Anomaly rate: {anomalies.shape[0] / df_clean.shape[0] * 100:.2f}% of cleaned dataset")
