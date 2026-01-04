"""
Quick comparison: Mean vs Max aggregation for ward-level flood risk
"""
import pandas as pd
import joblib
import numpy as np

# Load model and data
model = joblib.load('flood_risk_model.pkl')
df = pd.read_csv('FinalTrainingData3.csv')

# Load feature columns
import json
with open('feature_columns.json', 'r') as f:
    feature_cols = json.load(f)

def derive_risk_category(flood_count):
    if flood_count == 0:
        return 'No Risk'
    elif flood_count <= 2:
        return 'Low'
    elif flood_count <= 4:
        return 'Moderate'
    elif flood_count <= 6:
        return 'High'
    else:
        return 'Severe'

# Get 2025 data
latest_data = df[df['year'] == 2025].copy()
X_latest = latest_data[feature_cols].values
predictions = model.predict(X_latest)
predictions = np.maximum(0, predictions)
latest_data['predicted_flood_count'] = predictions

print("="*80)
print("WARD RISK AGGREGATION COMPARISON")
print("="*80)

# Method 1: MEAN (current - wrong)
ward_mean = latest_data.groupby('Ward_No').agg({
    'WardName': 'first',
    'predicted_flood_count': 'mean'
}).reset_index()
ward_mean['risk_category'] = ward_mean['predicted_flood_count'].apply(derive_risk_category)

# Method 2: MAX (corrected)
ward_max = latest_data.groupby('Ward_No').agg({
    'WardName': 'first',
    'predicted_flood_count': 'max'
}).reset_index()
ward_max['risk_category'] = ward_max['predicted_flood_count'].apply(derive_risk_category)

print("\nMETHOD 1: MEAN Aggregation (Current - Misleading)")
print("-" * 80)
mean_dist = ward_mean['risk_category'].value_counts().sort_index()
for cat, count in mean_dist.items():
    print(f"  {cat:12s}: {count:3d} wards")

print("\nMETHOD 2: MAX Aggregation (Corrected - Shows Peak Risk)")
print("-" * 80)
max_dist = ward_max['risk_category'].value_counts().sort_index()
for cat, count in max_dist.items():
    print(f"  {cat:12s}: {count:3d} wards")

print("\n" + "="*80)
print("TOP 10 HOTSPOTS COMPARISON")
print("="*80)

top_mean = ward_mean.nlargest(10, 'predicted_flood_count')
top_max = ward_max.nlargest(10, 'predicted_flood_count')

print("\nUsing MEAN (hides monsoon peaks):")
print("-" * 80)
for idx, row in top_mean.iterrows():
    print(f"  Ward {row['Ward_No']:3.0f} {row['WardName']:25s}: {row['predicted_flood_count']:.2f} ({row['risk_category']})")

print("\nUsing MAX (shows actual worst-case risk):")
print("-" * 80)
for idx, row in top_max.iterrows():
    print(f"  Ward {row['Ward_No']:3.0f} {row['WardName']:25s}: {row['predicted_flood_count']:.2f} ({row['risk_category']})")

print("\n" + "="*80)
print("CONCLUSION:")
print("  - MAX aggregation is correct for risk assessment (worst-case scenario)")
print("  - MEAN aggregation artificially lowers risk scores by averaging dry months")
print("="*80)
