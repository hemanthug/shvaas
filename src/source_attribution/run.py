from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from .aggregate import aggregate_results
from .candidates import generate_candidate_grid
from .config import AttributionConfig
from .features import compute_source_influence_features
from .inversion import infer_sources_elasticnet
from .kernel import build_influence_matrix
from .plots import save_all_plots
from .windows import make_time_windows, select_spike_rows

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DATASET = PROJECT_ROOT / "data" / "processed" / "pm25_with_weather_idw_k5_jan2025.csv"

BASELINE_FEATURES = ["latitude", "longitude", "hr_of_day", "day_of_week", "day_of_month"]
WEATHER_FEATURES = [
    "wind_speed_mps",
    "u_wind",
    "v_wind",
    "air_temp_c",
    "dew_point_c",
    "temp_dew_spread_c",
    "slp_hpa",
    "slp_anom_hpa",
    "visibility_m",
    "log_visibility",
    "ceiling_m",
]


def resolve_canonical_model_dataset() -> Path:
    """Resolve the dataset used by weather-aware model training."""
    if DEFAULT_MODEL_DATASET.exists():
        return DEFAULT_MODEL_DATASET

    processed_dir = PROJECT_ROOT / "data" / "processed"
    csv_files = sorted(processed_dir.glob("*.csv"))
    required_cols = {
        "site_id",
        "date",
        "pm25",
        "latitude",
        "longitude",
        "wind_speed_mps",
    }
    for path in csv_files:
        try:
            sample = pd.read_csv(path, nrows=10)
        except Exception:
            continue
        if required_cols.issubset(set(sample.columns)):
            return path

    raise FileNotFoundError(
        "Could not resolve canonical processed model dataset. "
        "Expected data/processed/pm25_with_weather_idw_k5_jan2025.csv or another processed CSV "
        "with model-ready PM+weather columns."
    )


def make_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out["hr_of_day"] = out[date_col].dt.hour
    out["day_of_week"] = out[date_col].dt.dayofweek
    out["day_of_month"] = out[date_col].dt.day
    return out


def ensure_wind_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure wind_dir_deg exists, deriving from u/v if required."""
    out = df.copy()
    if "wind_dir_deg" in out.columns:
        return out
    if {"u_wind", "v_wind"}.issubset(set(out.columns)):
        out["wind_dir_deg"] = (np.degrees(np.arctan2(out["u_wind"], out["v_wind"])) + 360.0) % 360.0
        return out
    raise ValueError(
        "Dataset needs 'wind_dir_deg' or both 'u_wind' and 'v_wind' to derive wind direction."
    )


def spike_mae(y_true: np.ndarray, y_pred: np.ndarray, site_ids: pd.Series) -> float:
    eval_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "site_id": site_ids.values})
    thresholds = eval_df.groupby("site_id")["y_true"].quantile(0.90).rename("thresh").reset_index()
    eval_df = eval_df.merge(thresholds, on="site_id", how="left")
    spikes = eval_df[eval_df["y_true"] >= eval_df["thresh"]]
    if spikes.empty:
        return float("nan")
    return float(mean_absolute_error(spikes["y_true"], spikes["y_pred"]))


def main() -> None:
    config = AttributionConfig()
    artifacts_dir = PROJECT_ROOT / config.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    data_path = resolve_canonical_model_dataset()
    df = pd.read_csv(data_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df = ensure_wind_direction(df)
    df = make_time_features(df, "date")
    df = df.dropna(subset=WEATHER_FEATURES).copy()

    block_indices = np.array_split(df.index, 4)
    df["block_id"] = 0
    for block_id, indices in enumerate(block_indices, start=1):
        df.loc[indices, "block_id"] = block_id

    train_data = df[df["block_id"].isin([1, 3])].copy()
    val_data = df[df["block_id"] == 4].copy()

    baseline_model = RandomForestRegressor(random_state=1)
    baseline_model.fit(train_data[BASELINE_FEATURES], train_data["pm25"])

    df["pred_baseline"] = baseline_model.predict(df[BASELINE_FEATURES])
    df["residual"] = df["pm25"] - df["pred_baseline"]

    spikes = select_spike_rows(df, config.spike_quantile)
    spikes = make_time_windows(spikes, config.window_hours)

    stations = df[["site_id", "latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    sources_df = generate_candidate_grid(stations, grid_spacing_km=config.grid_spacing_km, pad_deg=0.2)
    sources_df.to_csv(artifacts_dir / "candidates.csv", index=False)

    window_result_frames = []
    if not spikes.empty:
        for window_start, g in spikes.groupby("window_start"):
            A, y = build_influence_matrix(g, sources_df, config)
            coef, diag = infer_sources_elasticnet(A, y, config)
            frame = pd.DataFrame(
                {
                    "window_start": pd.Timestamp(window_start),
                    "source_id": sources_df["source_id"].to_numpy(dtype=int),
                    "coef": coef,
                }
            )
            frame["is_active"] = frame["coef"] > 1e-12
            frame["n_spikes"] = int(len(g))
            frame["r2"] = float(diag["r2"])
            frame["mse"] = float(diag["mse"])
            window_result_frames.append(frame)

    if window_result_frames:
        window_results = pd.concat(window_result_frames, ignore_index=True)
    else:
        window_results = pd.DataFrame(
            columns=["window_start", "source_id", "coef", "is_active", "n_spikes", "r2", "mse"]
        )
    window_results.to_csv(artifacts_dir / "window_results.csv", index=False)

    ranked_sources = aggregate_results(window_results, sources_df)
    ranked_sources.to_csv(artifacts_dir / "ranked_sources.csv", index=False)

    top_sources = ranked_sources.head(config.top_k_sources).copy()
    source_features = compute_source_influence_features(
        full_rows_df=df,
        top_sources_df=top_sources,
        sources_df=sources_df,
        config=config,
        mode="global",
    )
    source_features.to_csv(artifacts_dir / "source_features.csv", index=False)

    df_aug = df.merge(source_features, on=["site_id", "date"], how="left")
    src_cols = [c for c in df_aug.columns if c.startswith("srcinf_")]
    df_aug[src_cols] = df_aug[src_cols].fillna(0.0)

    train_aug = df_aug[df_aug["block_id"].isin([1, 3])].copy()
    val_aug = df_aug[df_aug["block_id"] == 4].copy()

    model_aug = RandomForestRegressor(random_state=1)
    model_aug.fit(train_aug[BASELINE_FEATURES + src_cols], train_aug["pm25"])

    pred_base = baseline_model.predict(val_data[BASELINE_FEATURES])
    pred_aug = model_aug.predict(val_aug[BASELINE_FEATURES + src_cols])

    metrics = {
        "dataset_path": str(data_path),
        "n_rows_model_df": int(len(df)),
        "n_rows_spikes": int(len(spikes)),
        "n_candidates": int(len(sources_df)),
        "baseline_mae_overall": float(mean_absolute_error(val_data["pm25"], pred_base)),
        "baseline_mae_spikes_top10pct_per_sensor": spike_mae(
            val_data["pm25"].to_numpy(),
            pred_base,
            val_data["site_id"],
        ),
        "baseline_plus_source_mae_overall": float(mean_absolute_error(val_aug["pm25"], pred_aug)),
        "baseline_plus_source_mae_spikes_top10pct_per_sensor": spike_mae(
            val_aug["pm25"].to_numpy(),
            pred_aug,
            val_aug["site_id"],
        ),
    }
    print(pd.DataFrame([metrics]))

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "source_attribution_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_all_plots(stations_df=stations, ranked_sources_df=ranked_sources, config=config)
    print(f"Saved source attribution artifacts to: {artifacts_dir}")
    print(f"Saved source attribution plots to: {PROJECT_ROOT / config.output_dir}")


if __name__ == "__main__":
    main()
