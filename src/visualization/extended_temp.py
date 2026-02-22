from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ---- Inputs ----
in_pm25_data = INTERIM_DIR / "la_pm25(nonegative)_0125_combined.csv"
in_weather_data = INTERIM_DIR / "weather_hourly_jan2025_decoded_v1.csv"
in_weights = INTERIM_DIR / "pm_site_to_weather_weights_k5.csv"

out_merged = PROCESSED_DIR / "pm25_with_weather_idw_k5_jan2025.csv"

# ---- Load ----
pm = pd.read_csv(in_pm25_data, low_memory=False)
wx = pd.read_csv(in_weather_data, low_memory=False)
wts = pd.read_csv(in_weights, low_memory=False)

# ---- Force ID dtypes to match (CRITICAL for merges) ----
pm["site_id"] = pm["site_id"].astype("string")
wx["station_id"] = wx["station_id"].astype("string")
wts["site_id"] = wts["site_id"].astype("string")
wts["station_id"] = wts["station_id"].astype("string")

# ---- Parse/align time keys ----
pm["timestamp_local"] = pd.to_datetime(pm["timestamp_local"], errors="coerce")
wx["date"] = pd.to_datetime(wx["date"], errors="coerce")

# Make a shared join key name
pm["date"] = pm["timestamp_local"]

# Optional: force hourly alignment (safe even if already hourly)
pm["date"] = pm["date"].dt.floor("h")
wx["date"] = wx["date"].dt.floor("h")

# ---- Keep only needed columns ----
pm_keep = ["site_id", "date", "pm25", "latitude", "longitude"]
pm = pm[pm_keep].copy()

wx_features = [
    "wind_speed_mps", "u_wind", "v_wind",
    "air_temp_c", "dew_point_c", "temp_dew_spread_c",
    "slp_hpa", "slp_anom_hpa",
    "visibility_m", "log_visibility",
    "ceiling_m",
]
wx_keep = ["station_id", "date"] + wx_features
wx = wx[wx_keep].copy()

wts = wts[["site_id", "station_id", "weight"]].copy()

# ---- Expand PM rows by k stations (site_id -> station_id, weight) ----
pm_expanded = pm.merge(wts, on="site_id", how="left")

# ---- Join weather by station and hour ----
pm_wx = pm_expanded.merge(wx, on=["station_id", "date"], how="left")

# Drop rows where weather is entirely missing (no station match for that hour)
# We'll detect station availability via one representative feature (wind_speed_mps)
pm_wx["has_wx"] = pm_wx["wind_speed_mps"].notna()
pm_wx_valid = pm_wx[pm_wx["has_wx"]].copy()

# ---- Renormalize weights per (site_id, date) over available stations ----
den = pm_wx_valid.groupby(["site_id", "date"])["weight"].transform("sum")
pm_wx_valid["weight_eff"] = pm_wx_valid["weight"] / den

# ---- Weighted blend for each feature ----
for f in wx_features:
    pm_wx_valid[f + "_w"] = pm_wx_valid["weight_eff"] * pm_wx_valid[f]

blended = (
    pm_wx_valid.groupby(["site_id", "date"], as_index=False)[[f + "_w" for f in wx_features]]
    .sum()
)

# Rename blended columns back to feature names
rename_map = {f + "_w": f for f in wx_features}
blended = blended.rename(columns=rename_map)

# ---- Merge blended weather back onto original PM rows ----
final = pm.merge(blended, on=["site_id", "date"], how="left")

# ---- Diagnostics ----
print("PM rows:", len(pm))
print("Expanded rows (PM x k):", len(pm_expanded))
print("Rows with station weather match:", len(pm_wx_valid))
print("Final rows:", len(final))

coverage = final[wx_features].notna().mean().sort_values()
print("\nFeature non-null fraction (coverage):")
print(coverage)

# ---- Save ----
final.to_csv(out_merged, index=False)
print("\nSaved:", out_merged)
