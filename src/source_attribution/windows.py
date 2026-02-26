from __future__ import annotations

import pandas as pd


def make_time_windows(df: pd.DataFrame, window_hours: int) -> pd.DataFrame:
    """Attach fixed-width window starts for time aggregation."""
    if "date" not in df.columns:
        raise ValueError("Expected column 'date' in dataframe.")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out["window_start"] = out["date"].dt.floor(f"{int(window_hours)}h")
    return out


def select_spike_rows(df_with_residuals: pd.DataFrame, spike_quantile: float) -> pd.DataFrame:
    """Keep positive residual rows above quantile threshold per site."""
    required = {"residual", "site_id"}
    missing = required - set(df_with_residuals.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df_with_residuals.copy()
    df = df[df["residual"] > 0].copy()
    if df.empty:
        return df

    thresh = (
        df.groupby("site_id")["residual"]
        .quantile(spike_quantile)
        .rename("spike_threshold")
        .reset_index()
    )
    df = df.merge(thresh, on="site_id", how="left")
    spikes = df[df["residual"] >= df["spike_threshold"]].drop(columns=["spike_threshold"])
    return spikes

