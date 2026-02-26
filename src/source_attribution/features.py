from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AttributionConfig
from .kernel import compute_influence


def compute_source_influence_features(
    full_rows_df: pd.DataFrame,
    top_sources_df: pd.DataFrame,
    sources_df: pd.DataFrame,
    config: AttributionConfig,
    mode: str = "global",
) -> pd.DataFrame:
    """Compute time-aligned source influence features for each row."""
    del sources_df  # top_sources_df already carries source coordinates
    required = {"site_id", "date", "latitude", "longitude", "wind_speed_mps", "wind_dir_deg"}
    missing = required - set(full_rows_df.columns)
    if missing:
        raise ValueError(f"full_rows_df missing required columns: {sorted(missing)}")

    if top_sources_df.empty:
        out = full_rows_df[["site_id", "date"]].copy()
        return out.drop_duplicates().reset_index(drop=True)

    top = top_sources_df.head(config.top_k_sources).copy().reset_index(drop=True)
    out = full_rows_df[["site_id", "date"]].copy()

    if mode not in {"global", "windowed"}:
        raise ValueError("mode must be either 'global' or 'windowed'.")

    coef_by_window = {}
    if mode == "windowed" and "window_start" in top.columns and "coef" in top.columns:
        coef_by_window = (
            top.groupby(["window_start", "source_id"], as_index=False)["coef"].sum().set_index(
                ["window_start", "source_id"]
            )["coef"]
        )
        work_df = full_rows_df.copy()
        work_df["window_start"] = pd.to_datetime(work_df["date"], errors="coerce").dt.floor(
            f"{int(config.window_hours)}h"
        )
    else:
        work_df = full_rows_df

    for idx, src in enumerate(top.itertuples(index=False), start=1):
        feature_name = f"srcinf_{idx:02d}"
        src_id = int(src.source_id)
        src_lat = float(src.latitude)
        src_lon = float(src.longitude)
        global_weight = float(getattr(src, "total_weight", 1.0))
        values = np.zeros(len(work_df), dtype=float)

        for i, row in enumerate(work_df.itertuples(index=False)):
            influence = compute_influence(
                sensor_lat=float(row.latitude),
                sensor_lon=float(row.longitude),
                source_lat=src_lat,
                source_lon=src_lon,
                wind_speed=float(row.wind_speed_mps),
                wind_dir_deg=float(row.wind_dir_deg),
                L_km=config.kernel_L_km,
                sigma_km=config.kernel_sigma_km,
            )
            if mode == "windowed" and coef_by_window:
                key = (row.window_start, src_id)
                coef = float(coef_by_window.get(key, 0.0))
                values[i] = influence * coef
            else:
                values[i] = influence * global_weight

        out[feature_name] = values

    return out

