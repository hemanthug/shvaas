from __future__ import annotations

import pandas as pd


def aggregate_results(window_results: pd.DataFrame, sources_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-window inferred source weights into a ranked source table."""
    if window_results.empty:
        out = sources_df[["source_id", "latitude", "longitude"]].copy()
        out["total_weight"] = 0.0
        out["active_count"] = 0
        out["active_fraction"] = 0.0
        out["cumulative_share"] = 0.0
        return out.sort_values("source_id").reset_index(drop=True)

    n_windows = max(window_results["window_start"].nunique(), 1)
    agg = (
        window_results.groupby("source_id", as_index=False)
        .agg(
            total_weight=("coef", "sum"),
            active_count=("is_active", "sum"),
        )
        .sort_values("total_weight", ascending=False)
    )
    agg["active_fraction"] = agg["active_count"] / float(n_windows)
    total = float(agg["total_weight"].sum())
    if total > 0:
        agg["cumulative_share"] = agg["total_weight"].cumsum() / total
    else:
        agg["cumulative_share"] = 0.0

    merged = sources_df.merge(agg, on="source_id", how="left")
    merged["total_weight"] = merged["total_weight"].fillna(0.0)
    merged["active_count"] = merged["active_count"].fillna(0).astype(int)
    merged["active_fraction"] = merged["active_fraction"].fillna(0.0)
    merged = merged.sort_values("total_weight", ascending=False).reset_index(drop=True)

    total_all = float(merged["total_weight"].sum())
    if total_all > 0:
        merged["cumulative_share"] = merged["total_weight"].cumsum() / total_all
    else:
        merged["cumulative_share"] = 0.0
    return merged

