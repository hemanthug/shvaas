from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import EARTH_RADIUS_KM


def _km_to_lat_deg(km: float) -> float:
    return np.rad2deg(km / EARTH_RADIUS_KM)


def _km_to_lon_deg(km: float, lat_deg: float) -> float:
    denom = EARTH_RADIUS_KM * np.cos(np.deg2rad(lat_deg))
    if abs(denom) < 1e-9:
        return 0.0
    return np.rad2deg(km / denom)


def generate_candidate_grid(
    stations_df: pd.DataFrame,
    grid_spacing_km: float,
    pad_deg: float = 0.2,
) -> pd.DataFrame:
    """Generate a regular source candidate grid over the station bounding box."""
    required = {"latitude", "longitude"}
    missing = required - set(stations_df.columns)
    if missing:
        raise ValueError(f"stations_df missing required columns: {sorted(missing)}")

    stations = stations_df[["latitude", "longitude"]].dropna().copy()
    if stations.empty:
        raise ValueError("No station coordinates available for candidate generation.")

    lat_min = float(stations["latitude"].min() - pad_deg)
    lat_max = float(stations["latitude"].max() + pad_deg)
    lon_min = float(stations["longitude"].min() - pad_deg)
    lon_max = float(stations["longitude"].max() + pad_deg)

    lat0 = float(stations["latitude"].mean())
    lat_step = max(_km_to_lat_deg(grid_spacing_km), 1e-5)
    lon_step = max(_km_to_lon_deg(grid_spacing_km, lat0), 1e-5)

    lats = np.arange(lat_min, lat_max + lat_step, lat_step)
    lons = np.arange(lon_min, lon_max + lon_step, lon_step)

    lat_mesh, lon_mesh = np.meshgrid(lats, lons, indexing="ij")
    df_sources = pd.DataFrame(
        {
            "source_id": np.arange(lat_mesh.size, dtype=int),
            "latitude": lat_mesh.ravel(),
            "longitude": lon_mesh.ravel(),
        }
    )
    return df_sources

