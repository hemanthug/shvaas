# Data Overview

Data lives under `data/` and is grouped by lifecycle stage. Large binaries are ignored by git; keep a local copy or replace with slimmer samples for CI.

## Raw (`data/raw/`)
- `aqs_sites.csv` — EPA AQI site metadata (nationwide).
- `la_sites.csv` — subset exported by scripts after cleaning; kept here if you prefer to cache it early.
- `hourly_88101_2025.csv` — hourly PM2.5 FRM/FEM observations (Jan 2025, large).
- `hourly_88502_2025.csv` — hourly PM2.5 non-FRM/FEM observations (Jan 2025, large).
- `weather_hourly_jan2025_noaa_v1.csv` — NOAA hourly weather (aggregated across station files).
- `noaa_station_files/` — optional folder for per-station CSVs used by `aggregates_wind_weather.py`.

## Interim (`data/interim/`)
- `la_pm25FF_hrly_0125.csv` — LA County FRM/FEM PM2.5 slice (Jan 1–25).
- `la_pm25nFF_hrly_0125.csv` — LA County non-FRM/FEM PM2.5 slice (Jan 1–25).
- `la_pm25_0125_combined.csv` — merged FRM + non-FRM (Jan 1–25).
- `la_pm25_0925_combined.csv` — merged FRM + non-FRM (Sep 1–25).
- `la_pm25(nonegative)_0125_combined.csv` — negative values clipped to zero with time features.
- `weather_hourly_jan2025_decoded_v1.csv` — decoded NOAA weather with engineered columns.
- `pm_site_to_weather_knn_k5.csv` — KNN mapping of PM sites to nearest weather stations.
- `pm_site_to_weather_weights_k5.csv` — distance-based weights for interpolation.
- `la_sites.csv` — cleaned AQI site metadata (LA County only).
- `*.png` plots written by data_ingestion scripts (kept in `reports/figures/`).

## Processed (`data/processed/`)
- `pm25_with_weather_idw_k5_jan2025.csv` — modeling-ready PM+weather dataset (hourly, interpolated by station weights).

### Conventions
- All timestamps are localized to `timestamp_local`/`date` columns and floored to hourly where applicable.
- IDs: `site_id` for PM monitors, `station_id` for weather stations.
- Coordinate columns are numeric (`latitude`, `longitude`).
