# JAL DRISHTI

**Ward-level Water-Logging Hotspot Visualization using Machine Learning**

JAL DRISHTI is a prototype that identifies ward-level water-logging hotspots using a Random Forest regression model trained on rainfall, drainage, terrain, and land-use proxy features.

---

## 🎯 Project Overview

Delhi experiences recurring water-logging that disrupts traffic, damages infrastructure, and affects citizens. This system provides:

- **Ward-level flood risk assessment** (250 wards)
- **ML-driven predictions** using Random Forest
- **Interactive GIS visualization** with Leaflet.js
- **Actionable insights** for urban planning authorities

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Modern web browser (Chrome, Firefox, Edge)

### Installation

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **(Optional) Train the model (regenerates model + feature list):**

   ```bash
   python train_model.py
   ```

   This will:

   - Load and preprocess `FinalTrainingData3.csv`
   - Train Random Forest on 2015-2022 data
   - Validate on 2023, test on 2024-2025
   - Generate `monthly_flood_predictions_2025.csv`
   - Save model to `flood_risk_model.pkl`

3. **Run the web app (FastAPI serves the upload page + map):**

   ```bash
   python -m uvicorn main:app --host 127.0.0.1 --port 8001
   ```

4. **Open in your browser:**

   - Upload page: `http://127.0.0.1:8001/`
   - Map page: `http://127.0.0.1:8001/map.html`

---

## 📁 Project Structure

```
project/
│
├── FinalTrainingData3.csv              # Training dataset (ward × month)
├── delhi_wards (1).json                # GeoJSON ward boundaries
│
├── train_model.py                      # ML training pipeline
├── main.py                             # FastAPI server (static + /api endpoints)
├── index.html                          # CSV upload page (redirects to map.html)
├── map.html                            # Interactive map + analytics
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── flood_risk_model.pkl               # Trained model (generated)
├── monthly_flood_predictions_2025.csv  # Default predictions used by the map
├── model_metrics.json                 # Performance metrics (generated)
├── feature_columns.json               # Feature list (generated)
│
└── visualizations/                    # Model performance plots
    ├── feature_importance.png
    └── prediction_scatter.png
```

---

## 🧠 Machine Learning Methodology

### Target Variable

- **`flood_count`**: Expected number of water-logging incidents per ward per month (regression task)

### Features (15+)

All features except metadata (`Ward_No`, `WardName`, `zone_name`) are used:

**Rainfall Features:**

- `max_rainfall_3day_mm` - Primary flood trigger
- `avg_monsoon_rainfall_mm` - Seasonal context
- `month_sin`, `month_cos` - Temporal encoding

**Drainage Infrastructure:**

- `drain_density_km_per_km2` - Drainage network coverage
- `drain_condition_score` - Maintenance quality (1-5)
- `drain_capacity_proxy` - Flow capacity estimate

**Terrain & Land Use:**

- `depression_area_fraction` - Low-lying vulnerability
- `mean_slope_percent` - Water runoff rate
- `mean_elevation_m` - Absolute elevation
- `impervious_surface_fraction` - Concrete/asphalt coverage
- `water_body_fraction` - Natural water bodies
- `yamuna_backflow_risk` - River proximity risk

**Administrative:**

- `year` - Temporal trend
- `area_km2` - Ward size

### Model Architecture

- **Algorithm**: Random Forest Regressor
- **Hyperparameters**:
  - `n_estimators`: 200 trees
  - `max_depth`: 15
  - `min_samples_split`: 10
  - `min_samples_leaf`: 4

### Data Split (Time-Aware)

- **Train**: 2015-2022 (8 years)
- **Validation**: 2023 (1 year)
- **Test**: 2024-2025 (2 years)

**No random shuffling** - maintains temporal consistency.

### Risk Score Derivation

Predicted `flood_count` → Categorical risk:

| Flood Count | Risk Category |
| ----------- | ------------- |
| 0           | **No Risk**   |
| 0 - 0.2     | **Low**       |
| 0.2 - 0.6   | **Moderate**  |
| 0.6 - 1.2   | **High**      |
| 1.2+        | **Severe**    |

---

## 📊 Model Performance

After training, check `model_metrics.json` for:

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)
- **Feature Importance Rankings**

Expected performance:

- R² > 0.7 on test set
- Top features: `max_rainfall_3day_mm`, `drain_capacity_proxy`, `impervious_surface_fraction`

---

## 🗺️ Web App (Upload + Map + Analytics)

The app is served by FastAPI (`main.py`) and consists of:

- **Upload page** (`/` → `index.html`): upload a CSV to `/api/predict`, stores the returned predictions in browser storage, then redirects to the map.
- **Map page** (`/map.html`): interactive Leaflet map with ward polygons and a range-based risk view.

### Map Features

- Choropleth coloring by risk category (ward-level)
- Click popups showing:
  - Risk level badge
  - Selected range label
  - MAX + AVG predicted flood count across the selected range
  - Rainfall proxy fields (if present)
  - Zone information

### Range Selection (single month or window)

- The UI uses a **range-only selector**.
- If you choose the same month for start and end, it behaves like a single-month view.
- For a multi-month selection, the map shows **worst-case risk per ward** using:

  - `predicted_flood_count_max` across the selected months (used for coloring)

### Panels / Analytics

- **Flood Risk Analysis panel** (left of the map): legend + key stats + model info.
  - Collapsible. When collapsed, it is removed from layout and the **map expands to full width**.
- **Insights & Infographics** (below): operational summary, recommended actions, hotspot list.
  - Includes compact Chart.js charts (risk mix + trend) sized to avoid excessive scrolling.

**Color Scheme:**

- 🟢 Green: No Risk
- 🟡 Yellow: Low
- 🟠 Orange: Moderate
- 🔴 Red: High
- 🔴 Dark Red: Severe

---

## 🔬 Key Assumptions & Constraints

### Physically Informed Logic

1. **Rainfall is the primary trigger** - floods cannot occur without rainfall
2. **Infrastructure matters** - same rainfall produces different outcomes based on:
   - Drainage capacity and condition
   - Terrain (slope, elevation, depressions)
   - Land use (impervious surfaces)
3. **Yamuna proximity** - wards near the river flood at lower thresholds (backflow risk)
4. **Seasonality** - month is encoded using `month_sin`/`month_cos`

### Data Granularity

- **Unit**: Ward × Month (not yearly aggregates)
- **Time range**: 2015-2025
- **Spatial coverage**: All 250 Delhi wards

### Model Limitations

⚠️ **Disclaimer**: This system uses physically informed machine learning and public climatological proxies to approximate urban flood risk where ward-level historical records are unavailable. Intended for:

- Hackathon demonstration
- Risk visualization
- Early-warning exploration

**Not suitable for**:

- Real-time emergency response
- Infrastructure investment decisions (without validation)
- Legal/compliance purposes

---

## 📈 Example Workflow

```python
# 1. Train the model
$ python train_model.py

# Output:
# ✓ Loaded 33,000 records
# ✓ Training Random Forest...
# ✓ Test R² = 0.78
# ✓ Top feature: max_rainfall_3day_mm (importance: 0.42)
# ✓ Saved monthly_flood_predictions_2025.csv

# 2. View predictions
$ cat monthly_flood_predictions_2025.csv | head

# Ward_No,WardName,zone_name,predicted_flood_count,risk_category,risk_score
# 1,WARD_A,Zone_X,0.00,No Risk,0.00
# 2,WARD_B,Zone_X,0.12,Low,0.12
# 3,WARD_C,Zone_Y,0.45,Moderate,0.45
# 4,WARD_D,Zone_Y,0.95,High,0.95
# 5,WARD_E,Zone_Z,1.80,Severe,1.80
# ...

# 3. Run the web app
$ python -m uvicorn main:app --host 127.0.0.1 --port 8001

# 4. Open
# Upload page: http://127.0.0.1:8001/
# Map page:    http://127.0.0.1:8001/map.html
```

---

## 🔌 API

- **GET `/api/template`**
  - Downloads a CSV header template for upload.
- **POST `/api/predict`**
  - Upload a future-year inputs CSV.
  - Returns a CSV including (at minimum):
    - `predicted_flood_count`
    - `risk_category`
    - `risk_score`
    - `risk_index`
    - `risk_category_scaled`

### Month encoding

If `month_sin`/`month_cos` are not present in the upload, the backend computes them from `month`.

---

## 🛠️ Advanced Usage

### Using XGBoost/LightGBM Instead

For better performance, modify [train_model.py](train_model.py):

```python
# Replace Random Forest with XGBoost
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    random_state=42
)
```

### Making Predictions on New Data

```python
import joblib
import pandas as pd
import json

# Load model and feature columns
model = joblib.load('flood_risk_model.pkl')
with open('feature_columns.json') as f:
    features = json.load(f)

# Prepare new data (must have all required features)
new_data = pd.DataFrame({...})  # Your data here
X_new = new_data[features].values

# Predict
predictions = model.predict(X_new)
predictions = np.maximum(0, predictions)  # Ensure non-negative
```

---

## 📦 Dependencies

See [requirements.txt](requirements.txt) for full list. Key packages:

- **scikit-learn** (1.3+) - Random Forest
- **pandas** (2.0+) - Data manipulation
- **numpy** (1.24+) - Numerical operations
- **matplotlib** (3.7+) - Visualization
- **seaborn** (0.12+) - Statistical plots

Optional (for advanced models):

- **xgboost** - Gradient boosting
- **lightgbm** - Fast gradient boosting

---

## 🏆 Demo Notes

This project is designed for immediate demo:

✅ **Complete end-to-end pipeline**
✅ **Professional visualization**
✅ **Clear documentation**
✅ **Reproducible results**
✅ **Production-quality code**
✅ **Physical interpretability**

**Presentation Tips:**

1. Start with **Upload → Map** (end-to-end flow)
2. Use the **range selector** to show worst-case ward risk for a window
3. Collapse the left panel to show the map expanding to full width
4. Use the **Hotspots + Risk Mix + Trend** cards for a quick operational briefing

---

## 📝 Citations & Data Sources

- **Ward Boundaries**: Municipal Corporation of Delhi (MCD) GeoJSON
- **Rainfall Data**: India Meteorological Department (IMD) proxies
- **Drainage Infrastructure**: Public GIS datasets + synthetic proxies
- **Terrain Data**: SRTM Digital Elevation Model

---

## 👥 Author

**Senior Machine Learning Engineer**  
Urban Flood-Risk Systems Architect

---

## 📄 License

This project is developed for educational and hackathon purposes.

---

## 🔗 Next Steps

1. **Validate with historical flood records** (if available)
2. **Integrate real-time rainfall forecasts** (IMD API)
3. **Deploy as web service** (FastAPI)
4. **Add mobile-responsive design**
5. **Include drainage improvement scenarios** (what-if analysis)

---

**For questions or improvements, check the code comments or modify as needed!** 🚀
