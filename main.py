from __future__ import annotations

import io
import json
import math
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles


app = FastAPI(title="Delhi Flood Risk Predictor")


def derive_risk_category(flood_count: float) -> str:
    if flood_count == 0:
        return "No Risk"
    if flood_count <= 0.2:
        return "Low"
    if flood_count <= 0.6:
        return "Moderate"
    if flood_count <= 1.2:
        return "High"
    return "Severe"


def compute_month_sin_cos(month_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    month_numeric = pd.to_numeric(month_series, errors="coerce")
    if month_numeric.isna().any():
        raise HTTPException(status_code=400, detail="Invalid month values. Month must be integers 1-12.")
    if ((month_numeric < 1) | (month_numeric > 12)).any():
        raise HTTPException(status_code=400, detail="Month out of range. Month must be between 1 and 12.")

    theta = 2 * math.pi * ((month_numeric.astype(int) - 1) / 12.0)
    return np.sin(theta), np.cos(theta)


def load_feature_columns() -> List[str]:
    try:
        with open("feature_columns.json", "r") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail="feature_columns.json not found. Train the model first.") from e


def load_model():
    try:
        return joblib.load("flood_risk_model.pkl")
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail="flood_risk_model.pkl not found. Train the model first.") from e


@app.get("/api/template", response_class=PlainTextResponse)
def download_template() -> str:
    feature_cols = load_feature_columns()

    base_cols = ["Ward_No", "WardName", "zone_name", "year", "month"]

    cols = base_cols.copy()
    for c in feature_cols:
        if c in {"month_sin", "month_cos"}:
            continue
        if c not in cols:
            cols.append(c)

    return ",".join(cols) + "\n"


@app.post("/api/predict")
def predict(file: UploadFile = File(...)) -> Response:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    content = file.file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {str(e)}") from e

    required_base = ["Ward_No", "month"]
    missing_base = [c for c in required_base if c not in df.columns]
    if missing_base:
        raise HTTPException(status_code=400, detail=f"Missing required columns for mapping: {missing_base}")

    feature_cols = load_feature_columns()

    if "month_sin" not in df.columns or "month_cos" not in df.columns:
        month_sin, month_cos = compute_month_sin_cos(df["month"])
        df["month_sin"] = month_sin
        df["month_cos"] = month_cos

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required model feature columns",
                "missing": missing_features,
            },
        )

    model = load_model()

    X = df[feature_cols].values
    preds = model.predict(X)
    preds = np.maximum(0, preds)

    out = df.copy()
    out["predicted_flood_count"] = preds
    out["risk_category"] = out["predicted_flood_count"].apply(derive_risk_category)
    out["risk_score"] = out["predicted_flood_count"].round(2)

    out["is_monsoon"] = pd.to_numeric(out["month"], errors="coerce").astype(int).isin([6, 7, 8, 9])

    if "max_rainfall_3day_mm" not in out.columns:
        out["max_rainfall_3day_mm"] = 0.0
    if "avg_monsoon_rainfall_mm" not in out.columns:
        out["avg_monsoon_rainfall_mm"] = 0.0

    out["risk_index"] = (
        out.groupby("month")["predicted_flood_count"]
        .transform(lambda s: s.rank(method="average", pct=True) * 100)
    ).round(1)

    out["risk_category_scaled"] = pd.cut(
        out["risk_index"],
        bins=[-0.01, 60, 85, 95, 100.01],
        labels=["Low", "Moderate", "High", "Severe"],
        right=True,
    ).astype(str)
    out.loc[out["predicted_flood_count"] == 0, "risk_category_scaled"] = "No Risk"

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    return Response(content=csv_bytes, media_type="text/csv")


app.mount("/", StaticFiles(directory=".", html=True), name="static")
