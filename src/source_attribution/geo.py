from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0


def latlon_to_xy_km(
    lat: float | np.ndarray,
    lon: float | np.ndarray,
    lat0: float,
    lon0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert lat/lon to local x/y kilometers using equirectangular approximation."""
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    lat0_rad = np.deg2rad(lat0)
    x_km = EARTH_RADIUS_KM * np.deg2rad(lon_arr - lon0) * np.cos(lat0_rad)
    y_km = EARTH_RADIUS_KM * np.deg2rad(lat_arr - lat0)
    return x_km, y_km


def wind_dir_met_to_bearing_to_deg(wind_dir_from_deg: float | np.ndarray) -> np.ndarray:
    """Convert meteorological wind direction (from) to plume travel direction (to)."""
    wd = np.asarray(wind_dir_from_deg, dtype=float)
    return np.mod(wd + 180.0, 360.0)


def rotate_into_wind_frame(
    dx_km: float | np.ndarray,
    dy_km: float | np.ndarray,
    wind_to_deg: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate vector into wind-aligned coordinates, returning downwind and crosswind distances."""
    dx = np.asarray(dx_km, dtype=float)
    dy = np.asarray(dy_km, dtype=float)
    theta = np.deg2rad(np.asarray(wind_to_deg, dtype=float))
    ux = np.sin(theta)  # East component of bearing
    uy = np.cos(theta)  # North component of bearing
    downwind_km = dx * ux + dy * uy
    crosswind_km = -dx * uy + dy * ux
    return downwind_km, crosswind_km

