"""
Quick inference script for making predictions on new data
"""

import joblib
import pandas as pd
import numpy as np
import json
import sys

def derive_risk_category(flood_count):
    """Convert predicted flood count to risk category"""
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

def predict_flood_risk(input_csv, output_csv='predictions.csv'):
    """
    Make flood risk predictions on new data
    
    Args:
        input_csv: Path to CSV with same features as training data
        output_csv: Path to save predictions
    """
    print("="*70)
    print("FLOOD RISK PREDICTION - INFERENCE MODE")
    print("="*70)
    
    # Load model
    print("\n[1/4] Loading trained model...")
    try:
        model = joblib.load('flood_risk_model.pkl')
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print("❌ Error: flood_risk_model.pkl not found!")
        print("   Please run train_model.py first to train the model.")
        sys.exit(1)
    
    # Load feature columns
    print("\n[2/4] Loading feature configuration...")
    try:
        with open('feature_columns.json', 'r') as f:
            feature_cols = json.load(f)
        print(f"✓ Loaded {len(feature_cols)} feature columns")
    except FileNotFoundError:
        print("❌ Error: feature_columns.json not found!")
        sys.exit(1)
    
    # Load input data
    print(f"\n[3/4] Loading input data from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
        print(f"✓ Loaded {len(df):,} records")
    except FileNotFoundError:
        print(f"❌ Error: {input_csv} not found!")
        sys.exit(1)
    
    # Check for required columns
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: Missing required columns: {missing_cols}")
        sys.exit(1)
    
    # Prepare features
    X = df[feature_cols].values
    
    # Make predictions
    print("\n[4/4] Making predictions...")
    predictions = model.predict(X)
    predictions = np.maximum(0, predictions)  # Ensure non-negative
    
    # Add predictions to dataframe
    df['predicted_flood_count'] = predictions
    df['risk_category'] = df['predicted_flood_count'].apply(derive_risk_category)
    df['risk_score'] = df['predicted_flood_count'].round(2)
    
    # Save results
    df.to_csv(output_csv, index=False)
    print(f"✓ Predictions saved to {output_csv}")
    
    # Display summary
    print("\n" + "="*70)
    print("PREDICTION SUMMARY")
    print("="*70)
    print(f"\nTotal records processed: {len(df):,}")
    print("\nRisk distribution:")
    risk_counts = df['risk_category'].value_counts().sort_index()
    for category, count in risk_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {category:12s}: {count:5d} ({pct:5.1f}%)")
    
    print("\nTop 10 Highest Risk:")
    top_10 = df.nlargest(10, 'risk_score')
    for idx, row in top_10.iterrows():
        ward_info = f"Ward {row.get('Ward_No', 'N/A')}" if 'Ward_No' in row.columns else f"Row {idx}"
        print(f"  {ward_info}: Risk Score = {row['risk_score']:.2f} ({row['risk_category']})")
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <input_csv> [output_csv]")
        print("\nExample:")
        print("  python predict.py FinalTrainingData3.csv predictions.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
    
    predict_flood_risk(input_file, output_file)
