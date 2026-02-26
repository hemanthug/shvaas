from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AttributionConfig
from .geo import latlon_to_xy_km, rotate_into_wind_frame, wind_dir_met_to_bearing_to_deg


def compute_influence(
    sensor_lat: float,
    sensor_lon: float,
    source_lat: float,
    source_lon: float,
    wind_speed: float,
    wind_dir_deg: float,
    L_km: float,
    sigma_km: float,
) -> float:
    """Compute wind-aligned source influence from candidate source to sensor."""
    del wind_speed  # reserved for future scaling, retained in signature by design
    dx_km, dy_km = latlon_to_xy_km(sensor_lat, sensor_lon, source_lat, source_lon)
    wind_to_deg = wind_dir_met_to_bearing_to_deg(wind_dir_deg)
    downwind_km, crosswind_km = rotate_into_wind_frame(dx_km, dy_km, wind_to_deg)
    if float(downwind_km) <= 0.0:
        return 0.0
    along = np.exp(-float(downwind_km) / max(L_km, 1e-6))
    lateral = np.exp(-(float(crosswind_km) ** 2) / (2.0 * max(sigma_km, 1e-6) ** 2))
    return float(along * lateral)


def build_influence_matrix(
    rows_df: pd.DataFrame,
    sources_df: pd.DataFrame,
    config: AttributionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build A matrix (rows x sources) and target y from spike residual rows."""
    required_rows = {
        "latitude",
        "longitude",
        "wind_speed_mps",
        "wind_dir_deg",
        "residual",
    }
    required_sources = {"latitude", "longitude"}
    missing_rows = required_rows - set(rows_df.columns)
    missing_sources = required_sources - set(sources_df.columns)
    if missing_rows:
        raise ValueError(f"rows_df missing required columns: {sorted(missing_rows)}")
    if missing_sources:
        raise ValueError(f"sources_df missing required columns: {sorted(missing_sources)}")

    clean_rows = rows_df.dropna(
        subset=["latitude", "longitude", "wind_speed_mps", "wind_dir_deg", "residual"]
    ).copy()
    if clean_rows.empty:
        return np.zeros((0, len(sources_df))), np.zeros(0)

    n_rows = len(clean_rows)
    n_sources = len(sources_df)
    A = np.zeros((n_rows, n_sources), dtype=float)
    y = clean_rows["residual"].to_numpy(dtype=float)

    src_lat = sources_df["latitude"].to_numpy(dtype=float)
    src_lon = sources_df["longitude"].to_numpy(dtype=float)

    for i, row in enumerate(clean_rows.itertuples(index=False)):
        sensor_lat = float(row.latitude)
        sensor_lon = float(row.longitude)
        wind_dir = float(row.wind_dir_deg)
        wind_speed = float(row.wind_speed_mps)

        dx_km, dy_km = latlon_to_xy_km(sensor_lat, sensor_lon, src_lat, src_lon)
        wind_to_deg = wind_dir_met_to_bearing_to_deg(wind_dir)
        downwind_km, crosswind_km = rotate_into_wind_frame(dx_km, dy_km, wind_to_deg)

        mask = downwind_km > 0.0
        vals = np.zeros(n_sources, dtype=float)
        if np.any(mask):
            along = np.exp(-downwind_km[mask] / max(config.kernel_L_km, 1e-6))
            lateral = np.exp(
                -(crosswind_km[mask] ** 2) / (2.0 * max(config.kernel_sigma_km, 1e-6) ** 2)
            )
            vals[mask] = along * lateral
            vals = vals * (1.0 + 0.0 * wind_speed)  # keeps signature intent explicit
        A[i, :] = vals

    return A, y

