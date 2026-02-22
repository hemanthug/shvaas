from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

in_pm25_data = INTERIM_DIR / "la_pm25(nonegative)_0125_combined.csv"
in_weather_data = INTERIM_DIR / "weather_hourly_jan2025_decoded_v1.csv"
out_knn_k5 = INTERIM_DIR / "pm_site_to_weather_knn_k5.csv"

df_pm25 = pd.read_csv(in_pm25_data, low_memory=False)
df_weather = pd.read_csv(in_weather_data, low_memory=False)

# Parse timestamps correctly
df_pm25["timestamp_local"] = pd.to_datetime(df_pm25["timestamp_local"], errors="coerce")
df_weather["date"] = pd.to_datetime(df_weather["date"], errors="coerce")

# Build station metadata (one row per station)
df_weather_metadata = (
    df_weather[["station_id", "latitude", "longitude"]]
    .dropna(subset=["station_id", "latitude", "longitude"])
    .drop_duplicates(subset="station_id")
    .copy()
)
df_weather_metadata["latitude"] = pd.to_numeric(df_weather_metadata["latitude"], errors="coerce")
df_weather_metadata["longitude"] = pd.to_numeric(df_weather_metadata["longitude"], errors="coerce")
df_weather_metadata = df_weather_metadata.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

# Build PM site metadata (one row per PM site)
df_pm25_metadata = (
    df_pm25[["site_id", "latitude", "longitude"]]
    .dropna(subset=["site_id", "latitude", "longitude"])
    .drop_duplicates(subset="site_id")
    .copy()
)
df_pm25_metadata["latitude"] = pd.to_numeric(df_pm25_metadata["latitude"], errors="coerce")
df_pm25_metadata["longitude"] = pd.to_numeric(df_pm25_metadata["longitude"], errors="coerce")
df_pm25_metadata = df_pm25_metadata.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

def haversine_km(lat1, lon1, lat2, lon2, radius=6371.0):
    """
    lat1, lon1: scalars (degrees)
    lat2, lon2: numpy arrays/Series (degrees)
    returns: numpy array of distances in km
    """
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return radius * c

# Compute k=5 nearest weather stations for each PM site
k = 5
results = []

station_lats = df_weather_metadata["latitude"].to_numpy()
station_lons = df_weather_metadata["longitude"].to_numpy()
station_ids = df_weather_metadata["station_id"].to_numpy()

for _, pm_row in df_pm25_metadata.iterrows():
    site_id = pm_row["site_id"]
    pm_lat = float(pm_row["latitude"])
    pm_lon = float(pm_row["longitude"])

    distances = haversine_km(pm_lat, pm_lon, station_lats, station_lons)

    idx = np.argsort(distances)[:k]
    nearest = pd.DataFrame(
        {
            "site_id": site_id,
            "station_id": station_ids[idx],
            "distance_km": distances[idx],
        }
    )
    nearest = nearest.sort_values("distance_km").reset_index(drop=True)
    nearest["rank"] = np.arange(1, len(nearest) + 1)

    results.append(nearest)

knn_k5 = pd.concat(results, ignore_index=True)

print("PM sites:", df_pm25_metadata["site_id"].nunique())
print("Weather stations:", df_weather_metadata["station_id"].nunique())
print("KNN rows:", len(knn_k5))
print(knn_k5.head(10))

knn_k5.to_csv(out_knn_k5, index=False)
print("Saved:", out_knn_k5)
