# Pipeline Details

This describes what each stage expects and produces. Run commands from the repo root so relative paths resolve.

## 1. Weather aggregation (optional)
`python src/data_ingestion/aggregates_wind_weather.py`
- **Input:** Per-station NOAA CSVs placed in `data/raw/noaa_station_files/`.
- **Output:** `data/raw/weather_hourly_jan2025_noaa_v1.csv` (hourly, deduped, limited to Jan 2025).

## 2. Weather decoding
`python src/data_ingestion/decode_weather_data.py`
- **Input:** `data/raw/weather_hourly_jan2025_noaa_v1.csv`
- **Output:** `data/interim/weather_hourly_jan2025_decoded_v1.csv`
- **Features:** wind components (`u_wind`, `v_wind`), temperature spread, SLP anomaly, log visibility.

## 3. AQI sites cleanup
`python src/data_ingestion/clean_aqi_sites.py`
- **Input:** `data/raw/aqs_sites.csv`
- **Output:** `data/interim/la_sites.csv` + `reports/figures/la_aqi_monitoring_sites.png`
- **Notes:** Filters to CA/LA (state 06, county 037), standardizes IDs, removes missing coords.

## 4. PM2.5 hourly filtering
`python src/data_ingestion/clean_hourly_pm25.py`
- **Input:** `data/raw/hourly_88101_2025.csv`
- **Output:** `data/interim/la_pm25FF_hrly_0125.csv`
- **Notes:** Keeps Jan 1–30, LA County only; builds `site_id` and `timestamp_local`.

## 5. PM2.5 merging
`python src/data_ingestion/combined_pm25_data.py`
- **Input:** `data/interim/la_pm25FF_hrly_0125.csv`, `data/interim/la_pm25nFF_hrly_0125.csv`
- **Output:** `data/interim/la_pm25_0925_combined.csv` + `reports/figures/la_pm25_monitor_locations.png`
- **Notes:** Standardizes column names, drops unusable rows, adds source flags.

## 6. Negative clipping & time features
`python src/features/initial_rfr_data_cleaning.py`
- **Input:** `data/interim/la_pm25_0125_combined.csv`
- **Output:** `data/interim/la_pm25(nonegative)_0125_combined.csv`
- **Notes:** Adds `hr_of_day`, `day_of_week`, `day_of_month`; clips `pm25 < 0` to zero.

## 7. PM ↔ Weather linkage
`python src/features/pm_site_to_weather_knn.py`
- **Input:** `data/interim/la_pm25(nonegative)_0125_combined.csv`, `data/interim/weather_hourly_jan2025_decoded_v1.csv`
- **Output:** `data/interim/pm_site_to_weather_knn_k5.csv`

`python src/visualization/temp.py`
- **Input:** `data/interim/pm_site_to_weather_knn_k5.csv`
- **Output:** `data/interim/pm_site_to_weather_weights_k5.csv`
- **Notes:** Computes inverse-distance weights (k=5).

## 8. PM + weather fusion
`python src/visualization/extended_temp.py`
- **Input:** `data/interim/la_pm25(nonegative)_0125_combined.csv`, `data/interim/weather_hourly_jan2025_decoded_v1.csv`, `data/interim/pm_site_to_weather_weights_k5.csv`
- **Output:** `data/processed/pm25_with_weather_idw_k5_jan2025.csv`
- **Notes:** Weighted blends weather features onto PM readings per hour.

## 9. Modeling
- `python src/models/initial_rfr.py` — baseline RandomForestRegressor using spatial + time features; saves `reports/figures/initial_rfr_prediction.png`.
- `python src/models/new_rfr.py` — weather-aware RFR with regime gating; saves `reports/figures/new_rfr_prediction.png`.
- `python src/models/link.py` — compares baseline vs weather-aware per site; saves figures per site into `reports/figures/`.

## Validation ideas
- Add schema checks (column presence/dtypes) after each stage.
- Track row counts and NA fractions to catch regressions.
- Unit-test the KNN distance/weighting functions with small fixtures.
