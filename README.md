# Shvaas

Hyperlocal air quality prediction using machine learning and environmental data.

Shvaas is an applied ML project that predicts air pollution at a hyperlocal level by blending historical air quality readings with weather and environmental signals. It began as a learn-by-doing exercise on messy real-world data (PM2.5 in Los Angeles) and has grown into an exploration of how local atmospheric regimes shape model performance.

## Why this matters
- Persistence baselines (using only past pollutant values) are strong but miss the “why” behind changes or gaps between monitors.
- Weather and transport features may improve spatial understanding—but not uniformly. Some sites benefit; others get noisier.
- The project focuses on spatially aware evaluation to see where models help, where they hurt, and why.

## Objectives
- Build a strong PM2.5 baseline from historical EPA sensor data.
- Integrate NOAA weather/wind to capture transport and dispersion.
- Evaluate by site, not just global averages; surface regime-specific behavior.
- Move toward regime-aware predictions, risk indexing, and uncertainty-aware outputs.

## Data
- EPA AQS hourly air quality measurements.
- NOAA Global Hourly weather datasets.
- Station metadata (lat/lon).
- Engineered features: temporal cycles (hour/day), weather (temp, humidity, pressure, wind), spatial station mapping.
Details on expected files: `docs/data_overview.md`.

## Modeling approach
- Current experiments: Random Forest regression baselines; persistence vs weather-enhanced models.
- Feature engineering for temporal and environmental signals.
- Site-level evaluation with per-site plots to reveal heterogeneous effects.
- Key insight so far: weather helps selectively; regimes differ across space.

## Pipeline (repo-aware, run from project root)
1) Aggregate (optional): `python src/data_ingestion/aggregates_wind_weather.py` → `data/raw/weather_hourly_jan2025_noaa_v1.csv`.
2) Decode weather: `python src/data_ingestion/decode_weather_data.py` → `data/interim/weather_hourly_jan2025_decoded_v1.csv`.
3) Clean AQI sites: `python src/data_ingestion/clean_aqi_sites.py` → `data/interim/la_sites.csv` + figure.
4) Filter PM hourly: `python src/data_ingestion/clean_hourly_pm25.py` → `data/interim/la_pm25FF_hrly_0125.csv`.
5) Merge PM sources: `python src/data_ingestion/combined_pm25_data.py` → `data/interim/la_pm25_0925_combined.csv` + figure.
6) Clip negatives + time feats: `python src/features/initial_rfr_data_cleaning.py` → `data/interim/la_pm25(nonegative)_0125_combined.csv`.
7) Map PM sites to weather + weights: `python src/features/pm_site_to_weather_knn.py`; `python src/visualization/temp.py`.
8) Fuse PM + weather: `python src/visualization/extended_temp.py` → `data/processed/pm25_with_weather_idw_k5_jan2025.csv`.
9) Modeling: `python src/models/initial_rfr.py`, `src/models/new_rfr.py`, `src/models/link.py` → figures in `reports/figures/`.

## Repo layout
```
src/                # pipeline scripts (ingestion → features → models → viz)
data/raw            # original inputs (git-ignored)
data/interim        # cleaned/intermediate tables
data/processed      # modeling-ready datasets
reports/figures     # generated plots
docs/               # data & pipeline guides
requirements.txt
```

## Tech stack
Python · pandas · NumPy · scikit-learn · matplotlib · tqdm

## Status
Active research/experimentation; documenting learnings while iterating toward regime-aware, risk-oriented air quality predictions.
