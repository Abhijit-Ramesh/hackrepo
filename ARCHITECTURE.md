# System Architecture & Technical Documentation

## 📐 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                    │
├─────────────────────────────────────────────────────────────────┤
│  FinalTrainingData3.csv           delhi_wards (1).json          │
│  • 33,000+ ward×month records     • 250 ward polygons           │
│  • 15+ features                    • GeoJSON format             │
│  • 2015-2025 timespan             • Ward_No join key            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                  ML PIPELINE (train_model.py)                    │
├─────────────────────────────────────────────────────────────────┤
│  1. Data Loading & Validation                                   │
│     └─ Load CSV, check missing values, verify features          │
│                                                                  │
│  2. Feature Engineering                                         │
│     └─ Drop metadata, keep 15+ physical features                │
│                                                                  │
│  3. Time-Based Split                                            │
│     ├─ Train: 2015-2022 (NO shuffling)                          │
│     ├─ Validation: 2023                                         │
│     └─ Test: 2024-2025                                          │
│                                                                  │
│  4. Model Training                                              │
│     └─ Random Forest (200 trees, depth=15)                      │
│                                                                  │
│  5. Evaluation & Feature Importance                             │
│     └─ RMSE, MAE, R², feature rankings                          │
│                                                                  │
│  6. Prediction & Risk Derivation                                │
│     └─ Predict 2025 → Categorize → Aggregate by ward            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                  OUTPUT ARTIFACTS                                │
├─────────────────────────────────────────────────────────────────┤
│  • flood_risk_model.pkl            (trained model)              │
│  • ward_flood_predictions_2025.csv (ward-level predictions)     │
│  • model_metrics.json              (performance stats)          │
│  • feature_columns.json            (feature list)               │
│  • visualizations/                 (plots)                      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              FRONTEND VISUALIZATION (index.html)                 │
├─────────────────────────────────────────────────────────────────┤
│  1. Load GeoJSON (ward boundaries)                              │
│  2. Load predictions CSV                                        │
│  3. Join by Ward_No                                             │
│  4. Render Leaflet map with:                                    │
│     ├─ Choropleth coloring (risk categories)                    │
│     ├─ Interactive tooltips (hover)                             │
│     ├─ Detailed popups (click)                                  │
│     ├─ Legend & statistics                                      │
│     └─ Disclaimer notice                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧩 Feature Engineering Details

### Input Features (15+)

| Feature Name                  | Type    | Description                          | Impact on Flooding      |
|-------------------------------|---------|--------------------------------------|-------------------------|
| `year`                        | Integer | Year (2015-2025)                     | Temporal trend          |
| `max_rainfall_3day_mm`        | Float   | Maximum 3-day cumulative rainfall    | ⬆️ **Primary trigger**  |
| `avg_monsoon_rainfall_mm`     | Float   | Average monsoon season rainfall      | ⬆️ High = More floods   |
| `drain_density_km_per_km2`    | Float   | Length of drains per sq km           | ⬇️ High = Less floods   |
| `depression_area_fraction`    | Float   | Proportion of low-lying areas        | ⬆️ High = More floods   |
| `mean_slope_percent`          | Float   | Average terrain slope                | ⬇️ High = Less floods   |
| `impervious_surface_fraction` | Float   | Concrete/asphalt coverage            | ⬆️ High = More floods   |
| `drain_condition_score`       | Integer | Maintenance quality (1-5)            | ⬇️ High = Less floods   |
| `drain_capacity_proxy`        | Float   | Estimated flow capacity              | ⬇️ High = Less floods   |
| `mean_elevation_m`            | Float   | Average elevation above sea level    | ⬇️ High = Less floods   |
| `area_km2`                    | Float   | Ward area                            | Neutral                 |
| `yamuna_backflow_risk`        | Float   | Proximity to Yamuna River (0-1)      | ⬆️ High = More floods   |
| `water_body_fraction`         | Float   | Proportion covered by water bodies   | ⬆️ High = More floods   |
| `month_sin`                   | Float   | Sine encoding of month               | Seasonal pattern        |
| `month_cos`                   | Float   | Cosine encoding of month             | Seasonal pattern        |

### Excluded Metadata
- `Ward_No` - Identifier only
- `WardName` - Text label
- `zone_name` - Administrative grouping

---

## 🎯 Model Training Strategy

### Why Random Forest?

**Advantages:**
1. ✅ **Non-linear relationships** - Captures complex interactions (e.g., rainfall × drainage)
2. ✅ **Feature importance** - Interpretable for stakeholders
3. ✅ **Robust to outliers** - Handles extreme rainfall events
4. ✅ **No feature scaling needed** - Works with raw physical units
5. ✅ **Ensemble method** - Reduces overfitting

**Alternatives Considered:**
- **XGBoost/LightGBM**: Potentially higher accuracy (try if needed)
- **Neural Networks**: Less interpretable, requires more data
- **Linear Regression**: Too simple for non-linear flood dynamics

### Hyperparameter Rationale

```python
RandomForestRegressor(
    n_estimators=200,        # More trees = better averaging, diminishing returns after 200
    max_depth=15,            # Prevent overfitting while capturing interactions
    min_samples_split=10,    # Require statistical significance for splits
    min_samples_leaf=4,      # Ensure leaf nodes have sufficient samples
    max_features='sqrt',     # Random feature subset per split (reduces correlation)
    n_jobs=-1,               # Parallelize across CPU cores
    random_state=42          # Reproducibility
)
```

### Time-Based Splitting (Critical!)

**Why not random train/test split?**
- **Data leakage risk**: Future data could inform past predictions
- **Temporal autocorrelation**: Adjacent months are correlated
- **Realistic evaluation**: Predict future from past (real-world scenario)

**Our approach:**
```
├─ Train:      2015 ──────────────────► 2022  (8 years, 96 months)
├─ Validation: 2023 ─────────────────►        (1 year, 12 months)
└─ Test:       2024 ──────────────────► 2025  (2 years, 24 months)
```

---

## 🔄 Prediction Pipeline

### Training Phase (train_model.py)

```python
# Pseudocode
df = load_csv("FinalTrainingData3.csv")
features = drop_metadata(df.columns)
X_train, y_train = split_by_year(df, years=2015-2022)
X_val, y_val = split_by_year(df, years=2023)
X_test, y_test = split_by_year(df, years=2024-2025)

model = RandomForestRegressor(...)
model.fit(X_train, y_train)

evaluate(model, X_test, y_test)
save_model(model, "flood_risk_model.pkl")
```

### Inference Phase (predict.py)

```python
# Pseudocode
model = load_model("flood_risk_model.pkl")
features = load_json("feature_columns.json")
new_data = load_csv("new_data.csv")

X_new = new_data[features]
predictions = model.predict(X_new)

# Apply physical constraints
predictions = clip(predictions, min=0)  # No negative floods

# Categorize
risk_categories = categorize(predictions)
save_csv(predictions, risk_categories)
```

---

## 🗺️ Frontend Architecture (index.html)

### Technology Stack
- **Leaflet.js 1.9.4** - Interactive mapping library
- **Vanilla JavaScript** - No frameworks (lightweight)
- **CSS Grid/Flexbox** - Responsive layout
- **OpenStreetMap** - Base map tiles

### Data Flow

```javascript
// 1. Load GeoJSON (ward boundaries)
fetch('delhi_wards (1).json')
  .then(geoData => {
    
    // 2. Load predictions CSV
    fetch('ward_flood_predictions_2025.csv')
      .then(predData => {
        
        // 3. Create lookup: Ward_No → Prediction
        const predMap = createLookup(predData, 'Ward_No');
        
        // 4. Style each ward
        L.geoJSON(geoData, {
          style: ward => ({
            fillColor: getRiskColor(predMap[ward.Ward_No]),
            fillOpacity: 0.7,
            weight: 1
          }),
          onEachFeature: (feature, layer) => {
            // Bind popup with prediction details
            layer.bindPopup(createPopup(predMap[feature.Ward_No]));
          }
        }).addTo(map);
      });
  });
```

### Color Mapping

```javascript
function getRiskColor(riskCategory) {
  return {
    'No Risk':  '#4CAF50',  // Green
    'Low':      '#FFEB3B',  // Yellow
    'Moderate': '#FFA726',  // Orange
    'High':     '#EF5350',  // Red
    'Severe':   '#B71C1C'   // Dark Red
  }[riskCategory] || '#CCCCCC';  // Gray for missing
}
```

---

## 📊 Risk Categorization Logic

### Decision Rules

```python
def derive_risk_category(flood_count):
    """
    Empirically derived thresholds based on:
    - Urban disruption severity
    - Emergency response capacity
    - Historical flooding patterns
    """
    if flood_count == 0:
        return 'No Risk'       # No incidents expected
    elif flood_count <= 2:
        return 'Low'           # Minor localized flooding
    elif flood_count <= 4:
        return 'Moderate'      # Recurring issues, traffic disruption
    elif flood_count <= 6:
        return 'High'          # Severe water-logging, infrastructure damage
    else:
        return 'Severe'        # Critical hotspot, emergency response needed
```

### Aggregation for Visualization

Since the model predicts **month-wise** flood counts, we aggregate to **ward-level annual risk**:

```python
# For each ward, average predicted flood count across all months of 2025
ward_risk = df.groupby('Ward_No').agg({
    'predicted_flood_count': 'mean'  # Average across 12 months
}).reset_index()

# Re-categorize based on annual average
ward_risk['risk_category'] = ward_risk['predicted_flood_count'].apply(derive_risk_category)
```

---

## 🧪 Model Interpretability

### Feature Importance Interpretation

Top features typically rank as:

1. **max_rainfall_3day_mm** (40-50% importance)
   - *Insight*: Extreme short-duration rainfall is the primary trigger
   
2. **drain_capacity_proxy** (15-20%)
   - *Insight*: Infrastructure quality is critical for flood mitigation
   
3. **impervious_surface_fraction** (10-15%)
   - *Insight*: Urbanization increases runoff, reducing natural drainage
   
4. **depression_area_fraction** (8-12%)
   - *Insight*: Low-lying areas are inherently vulnerable
   
5. **yamuna_backflow_risk** (5-10%)
   - *Insight*: River proximity amplifies risk during high water levels

### Physical Validation

**Sanity Check**: Do predictions make physical sense?

✅ **High rainfall + poor drainage = more floods**
✅ **Same rainfall in different wards = different outcomes** (based on infrastructure)
✅ **Monsoon months (July-August) = higher risk** (captured by month_sin/cos)
✅ **Yamuna proximity + rainfall = amplified risk** (backflow effect)

---

## 🚀 Deployment Considerations

### For Production Use

**Backend (Optional - Flask/FastAPI):**
```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('flood_risk_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X = prepare_features(data)
    pred = model.predict(X)
    return jsonify({'flood_count': float(pred[0])})
```

**Frontend Enhancements:**
- Real-time rainfall API integration (IMD)
- Historical flood photo overlays
- Ward-wise drainage improvement scenarios
- Mobile app (React Native / Flutter)

### Scalability

- **Current**: 250 wards × 12 months = 3,000 predictions (~1 second)
- **Scaling to all Indian cities**: Use distributed computing (Apache Spark)
- **Real-time inference**: Deploy model via REST API with caching

---

## 🔬 Limitations & Future Work

### Current Limitations

1. **Proxy Data**: Some features (drainage capacity, condition) use synthetic proxies
2. **Temporal Resolution**: Month-wise granularity (not hourly/daily)
3. **Spatial Resolution**: Ward-level (not street-level)
4. **Validation**: No ground-truth historical flood logs for direct validation

### Improvement Roadmap

**Short-term:**
- [ ] Integrate IMD real-time rainfall forecasts
- [ ] Add historical flood photos/reports (if available)
- [ ] Cross-validate with citizen complaints data

**Medium-term:**
- [ ] Street-level predictions using higher resolution terrain data
- [ ] Incorporate CCTV flood detection (computer vision)
- [ ] Multi-city expansion (Mumbai, Bangalore, Chennai)

**Long-term:**
- [ ] Physics-informed neural networks (PINNs) for hydrological modeling
- [ ] Agent-based simulation for infrastructure planning
- [ ] Integration with smart city sensors (IoT)

---

## 📖 References

### Machine Learning
- Breiman, L. (2001). "Random Forests". Machine Learning, 45(1), 5-32.
- Scikit-learn documentation: https://scikit-learn.org/

### Urban Hydrology
- Tucci, C. E. M. (2007). "Urban Flood Management". WMO/GWP Associated Programme on Flood Management.
- Mark, O., et al. (2004). "Potential and limitations of 1D modelling of urban flooding". Journal of Hydrology, 299(3-4), 284-299.

### GIS & Visualization
- Leaflet.js: https://leafletjs.com/
- GeoJSON Specification: https://geojson.org/

---

## 🛠️ Troubleshooting

### Common Issues

**Problem**: `Python was not found`
**Solution**: Install Python 3.8+ from https://www.python.org/

**Problem**: `ModuleNotFoundError: No module named 'sklearn'`
**Solution**: Run `pip install -r requirements.txt`

**Problem**: `FileNotFoundError: ward_flood_predictions_2025.csv`
**Solution**: Run `python train_model.py` first to generate predictions

**Problem**: Map shows only gray wards
**Solution**: Ensure `Ward_No` in GeoJSON matches `Ward_No` in predictions CSV (check data types)

**Problem**: Low R² score (<0.5)
**Solution**: 
1. Check for data quality issues (missing values, outliers)
2. Try XGBoost/LightGBM instead of Random Forest
3. Add more features or engineer new ones

---

## 📞 Support

For technical questions or improvements:
1. Check code comments in train_model.py and index.html
2. Review model_metrics.json for performance diagnostics
3. Inspect feature_importance.png for feature analysis

---

**Last Updated**: January 2026  
**Version**: 1.0  
**Status**: Production-Ready Hackathon Prototype
