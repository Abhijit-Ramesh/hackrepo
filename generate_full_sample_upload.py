import numpy as np
import pandas as pd


def month_from_sin_cos(month_sin: pd.Series, month_cos: pd.Series) -> pd.Series:
    angle = np.arctan2(month_sin.astype(float), month_cos.astype(float))
    angle = np.mod(angle, 2 * np.pi)
    month = (np.round(angle / (2 * np.pi) * 12).astype(int) % 12) + 1
    return month


def main():
    df = pd.read_csv('FinalTrainingData3.csv')

    base = df[df['year'] == 2025].copy()
    if len(base) == 0:
        raise SystemExit('No rows found for year == 2025 in FinalTrainingData3.csv')

    base['month'] = month_from_sin_cos(base['month_sin'], base['month_cos'])

    keep_cols = [
        'Ward_No', 'WardName', 'zone_name',
        'year', 'month',
        'max_rainfall_3day_mm', 'avg_monsoon_rainfall_mm',
        'drain_density_km_per_km2', 'depression_area_fraction',
        'mean_slope_percent', 'impervious_surface_fraction',
        'drain_condition_score', 'drain_capacity_proxy',
        'mean_elevation_m', 'area_km2', 'yamuna_backflow_risk',
        'water_body_fraction'
    ]

    missing = [c for c in keep_cols if c not in base.columns]
    if missing:
        raise SystemExit(f'Missing expected columns in training data: {missing}')

    out = base[keep_cols].copy()

    # Change year to next year (example: 2026)
    out['year'] = 2026

    # Ensure month is 1..12 integers
    out['month'] = out['month'].astype(int)

    # Stable ordering: Ward_No then month
    out = out.sort_values(['Ward_No', 'month']).reset_index(drop=True)

    out_path = 'sample_upload_inputs_2026_full.csv'
    out.to_csv(out_path, index=False)
    print(f'Wrote {len(out):,} rows to {out_path}')


if __name__ == '__main__':
    main()
