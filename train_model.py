"""
Delhi Flood Risk Prediction System
===================================

This system uses Random Forest regression to predict ward-level flood counts
based on rainfall, drainage, terrain, and land-use features.

Author: Senior ML Engineer
Purpose: Mapping Water-Logging Hotspots of Delhi
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# ============================
# 1. DATA LOADING & PREPROCESSING
# ============================

print("="*70)
print("DELHI FLOOD RISK PREDICTION SYSTEM")
print("="*70)
print("\n[1/6] Loading data...")

# Load training data
df = pd.read_csv('FinalTrainingData3.csv')
print(f"[OK] Loaded {len(df):,} records from FinalTrainingData3.csv")
print(f"  - Time range: {df['year'].min()} to {df['year'].max()}")
print(f"  - Unique wards: {df['Ward_No'].nunique()}")
print(f"  - Total features: {len(df.columns)}")

# ============================
# 2. FEATURE ENGINEERING
# ============================

print("\n[2/6] Preparing features...")

# Identify features to drop (metadata only, not useful for ML)
features_to_drop = ['Ward_No', 'WardName', 'zone_name']
metadata_cols = [col for col in features_to_drop if col in df.columns]

# Identify feature columns (all except target and metadata)
target_col = 'flood_count'
feature_cols = [col for col in df.columns 
                if col not in metadata_cols + [target_col]]

print(f"[OK] Dropped metadata columns: {metadata_cols}")
print(f"[OK] Using {len(feature_cols)} features for training:")
print(f"  {', '.join(feature_cols)}")

# Check for missing values
missing_count = df[feature_cols + [target_col]].isnull().sum().sum()
if missing_count > 0:
    print(f"\n[WARNING] {missing_count} missing values detected")
    # Fill missing values with median for numeric columns
    for col in feature_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    print("[OK] Missing values filled with column medians")
else:
    print("[OK] No missing values detected")

# ============================
# 3. TIME-BASED TRAIN/VAL/TEST SPLIT
# ============================

print("\n[3/6] Creating time-based splits...")

# Split data by year (time-aware, no shuffling)
train_data = df[df['year'] <= 2022].copy()
val_data = df[df['year'] == 2023].copy()
test_data = df[df['year'] >= 2024].copy()

print(f"[OK] Train set (2015-2022): {len(train_data):,} records")
print(f"[OK] Validation set (2023): {len(val_data):,} records")
print(f"[OK] Test set (2024-2025): {len(test_data):,} records")

# Prepare feature matrices and target vectors
X_train = train_data[feature_cols].values
y_train = train_data[target_col].values

X_val = val_data[feature_cols].values
y_val = val_data[target_col].values

X_test = test_data[feature_cols].values
y_test = test_data[target_col].values

print(f"[OK] Feature matrix shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

# ============================
# 4. MODEL TRAINING
# ============================

print("\n[4/6] Training Random Forest Regressor...")

# Random Forest Regressor with optimized hyperparameters
# (For hackathon: could also try XGBoost/LightGBM for better performance)
model = RandomForestRegressor(
    n_estimators=200,           # Number of trees
    max_depth=15,                # Maximum depth of trees
    min_samples_split=10,        # Minimum samples to split a node
    min_samples_leaf=4,          # Minimum samples in leaf node
    max_features='sqrt',         # Number of features for best split
    n_jobs=-1,                   # Use all CPU cores
    random_state=42,
    verbose=0
)

# Train the model
print("  Training in progress...")
model.fit(X_train, y_train)
print("[OK] Model training completed!")

# NOTE: The model trained above is for *evaluation* (honest time-based split).
# We will later retrain a final production model on all available data (including 2025)
# before saving flood_risk_model.pkl for deployment.

# ============================
# 5. MODEL EVALUATION
# ============================

print("\n[5/6] Evaluating model performance...")

def evaluate_model(y_true, y_pred, dataset_name):
    """Calculate and display evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n  {dataset_name}:")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAE:  {mae:.4f}")
    print(f"    R²:   {r2:.4f}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

# Predictions on all datasets
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Ensure non-negative predictions (flood count can't be negative)
y_train_pred = np.maximum(0, y_train_pred)
y_val_pred = np.maximum(0, y_val_pred)
y_test_pred = np.maximum(0, y_test_pred)

# Evaluate on all sets
metrics_train = evaluate_model(y_train, y_train_pred, "Training Set")
metrics_val = evaluate_model(y_val, y_val_pred, "Validation Set")
metrics_test = evaluate_model(y_test, y_test_pred, "Test Set")

# Feature importance analysis
print("\n  Top 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"    {row['feature']:30s} {row['importance']:.4f}")

# ============================
# 6. SAVE MODEL & GENERATE PREDICTIONS
# ============================

print("\n[6/6] Saving model and generating predictions...")

# ============================
# TRAIN FINAL MODEL (PRODUCTION)
# ============================

print("\n  Training final production model on all data through 2025...")

train_full = df[df['year'] <= 2025].copy()
X_full = train_full[feature_cols].values
y_full = train_full[target_col].values

final_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42,
    verbose=0
)
final_model.fit(X_full, y_full)
print("[OK] Final model trained (includes 2025)")

# Save the trained model
joblib.dump(final_model, 'flood_risk_model.pkl')
print("[OK] Model saved to flood_risk_model.pkl")

# Save feature columns for inference
with open('feature_columns.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)
print("[OK] Feature columns saved to feature_columns.json")

# ============================
# DERIVE RISK SCORES
# ============================

def derive_risk_category(flood_count):
    """
    Convert predicted flood count to categorical risk level
    
    Args:
        flood_count: Predicted number of flood incidents
        
    Returns:
        Risk category: 'No Risk', 'Low', 'Moderate', 'High', or 'Severe'
    """
    if flood_count == 0:
        return 'No Risk'
    elif flood_count <= 0.2:
        return 'Low'
    elif flood_count <= 0.6:
        return 'Moderate'
    elif flood_count <= 1.2:
        return 'High'
    else:
        return 'Severe'

# Get predictions for the most recent data (2025)
latest_data = df[df['year'] == 2025].copy()
print(f"\n[OK] Generating predictions for {len(latest_data):,} ward-months in 2025...")

if len(latest_data) > 0:
    X_latest = latest_data[feature_cols].values
    latest_predictions = final_model.predict(X_latest)
    latest_predictions = np.maximum(0, latest_predictions)  # Ensure non-negative
    
    latest_data['predicted_flood_count'] = latest_predictions
    latest_data['risk_category'] = latest_data['predicted_flood_count'].apply(derive_risk_category)
    
    month_angle = np.arctan2(latest_data['month_sin'], latest_data['month_cos'])
    month_angle = np.mod(month_angle, 2 * np.pi)
    latest_data['month'] = (np.round(month_angle / (2 * np.pi) * 12).astype(int) % 12) + 1

    latest_data['is_monsoon'] = latest_data['month'].isin([6, 7, 8, 9])

    monthly_predictions = latest_data[[
        'Ward_No', 'WardName', 'zone_name', 'month', 'is_monsoon',
        'predicted_flood_count', 'max_rainfall_3day_mm', 'avg_monsoon_rainfall_mm'
    ]].copy()

    monthly_predictions['risk_category'] = monthly_predictions['predicted_flood_count'].apply(derive_risk_category)
    monthly_predictions['risk_score'] = monthly_predictions['predicted_flood_count'].round(2)
    monthly_predictions['risk_index'] = (
        monthly_predictions.groupby('month')['predicted_flood_count']
        .transform(lambda s: s.rank(method='average', pct=True) * 100)
    ).round(1)
    monthly_predictions['risk_category_scaled'] = pd.cut(
        monthly_predictions['risk_index'],
        bins=[-0.01, 60, 85, 95, 100.01],
        labels=['Low', 'Moderate', 'High', 'Severe'],
        right=True
    ).astype(str)
    monthly_predictions.loc[monthly_predictions['predicted_flood_count'] == 0, 'risk_category_scaled'] = 'No Risk'

    monthly_predictions.to_csv('monthly_flood_predictions_2025.csv', index=False)
    print("[OK] Monthly predictions saved to monthly_flood_predictions_2025.csv")
    
    print("\n  Sanity checks (2025 - Monthly):")
    print(f"    Unique months present: {sorted(monthly_predictions['month'].unique().tolist())}")
    print(f"    Monsoon ward-months: {int(monthly_predictions['is_monsoon'].sum())} / {len(monthly_predictions):,}")
    
    print("\n  Prediction summary by month (mean / max):")
    month_summary = monthly_predictions.groupby('month')['predicted_flood_count'].agg(['mean', 'max']).round(3)
    print(month_summary)
    
    print("\n  Risk distribution by month (scaled percentiles):")
    month_risk = pd.crosstab(monthly_predictions['month'], monthly_predictions['risk_category_scaled'])
    print(month_risk)
    
    print("\n  Monsoon vs Non-monsoon prediction summary (mean / max):")
    monsoon_summary = monthly_predictions.groupby('is_monsoon')['predicted_flood_count'].agg(['mean', 'max']).round(3)
    print(monsoon_summary)
    
    print("\n  Top 5 wards per monsoon month (by risk_score):")
    for m in [6, 7, 8, 9]:
        m_df = monthly_predictions[monthly_predictions['month'] == m].nlargest(5, 'risk_score')
        if len(m_df) == 0:
            continue
        print(f"\n    Month {m}:")
        for _, row in m_df.iterrows():
            print(f"      Ward {row['Ward_No']:3.0f} ({row['WardName']:25s}): {row['risk_score']:.2f} ({row['risk_category']})")

    print("\n  Risk Category Distribution (2025 - Monthly):")
    risk_dist = monthly_predictions['risk_category'].value_counts().sort_index()
    for category, count in risk_dist.items():
        print(f"    {category:12s}: {count:3d} ward-months")

    print("\n  Top 10 Flood Hotspots (Highest Risk - Overall Max Monthly):")
    top_hotspots = monthly_predictions.nlargest(10, 'risk_score')
    for idx, row in top_hotspots.iterrows():
        print(f"    Ward {row['Ward_No']:3.0f} ({row['WardName']:25s}) Month {row['month']:2.0f}: Risk Score = {row['risk_score']:.2f} ({row['risk_category']})")

else:
    print("[WARNING] No data found for 2025")

# ============================
# SAVE EVALUATION METRICS
# ============================

metrics_summary = {
    'model_type': 'Random Forest Regressor',
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'hyperparameters': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 4
    },
    'metrics': {
        'train': metrics_train,
        'validation': metrics_val,
        'test': metrics_test
    },
    'feature_importance': feature_importance.head(15).to_dict('records')
}

with open('model_metrics.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)
print("\n[OK] Model metrics saved to model_metrics.json")

# ============================
# VISUALIZATION (Optional)
# ============================

try:
    # Create visualizations directory
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Feature importance plot
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(15)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Importance Score')
    plt.title('Top 15 Most Important Features for Flood Prediction')
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Prediction vs Actual (Test Set)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Flood Count')
    plt.ylabel('Predicted Flood Count')
    plt.title(f'Prediction vs Actual (Test Set 2024-2025)\nR² = {metrics_test["r2"]:.4f}')
    plt.tight_layout()
    plt.savefig('visualizations/prediction_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("[OK] Visualizations saved to visualizations/ directory")
    
except Exception as e:
    print(f"[WARNING] Could not create visualizations: {str(e)}")

print("\n" + "="*70)
print("MODEL TRAINING COMPLETE!")
print("="*70)
print("\nNext steps:")
print("1. Review model_metrics.json for detailed performance metrics")
print("2. Check monthly_flood_predictions_2025.csv for month-wise risk scores")
print("3. Open index.html in a browser to view the interactive flood risk map")
print("="*70)
