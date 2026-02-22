from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Inputs
# ----------------------------
DATA_PATH = PROCESSED_DIR / "pm25_with_weather_idw_k5_jan2025.csv"

TARGET_LAT = 34.19925
TARGET_LON = -118.53276
TOL = 1e-4

# Plot window (same style as your PM-only plot)
PLOT_START = pd.Timestamp("2025-01-27 00:00:00")
PLOT_END = pd.Timestamp("2025-01-31 00:00:00")

# Regime gate smoothing settings
ROLLING_HOURS = 6            # 3 or 6 are good. 6 is smoother
SOFTNESS_MPS = 0.35          # larger = smoother blending, smaller = more switching

# ----------------------------
# Helpers
# ----------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def soft_gate(wind_roll, threshold, softness):
    """
    Returns alpha in [0,1].
    alpha ~ 0 => trust baseline more
    alpha ~ 1 => trust weather model more
    """
    z = (wind_roll - threshold) / max(softness, 1e-6)
    return sigmoid(z)

def make_time_features(df, date_col="date"):
    df = df.copy()
    df["hr_of_day"] = df[date_col].dt.hour
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["day_of_month"] = df[date_col].dt.day
    return df

def add_grouped_rolling_wind(df, wind_col="wind_speed_mps", group_col="site_id", date_col="date", window_hours=6):
    df = df.sort_values([group_col, date_col]).copy()
    df["wind_speed_roll"] = (
        df.groupby(group_col)[wind_col]
        .transform(lambda s: s.rolling(window=window_hours, min_periods=1).mean())
    )
    return df

# ----------------------------
# Load + prep
# ----------------------------
pm25_data = pd.read_csv(DATA_PATH, low_memory=False)

pm25_data["date"] = pd.to_datetime(pm25_data["date"], errors="coerce")
pm25_data = pm25_data.sort_values("date").reset_index(drop=True)

pm25_data = make_time_features(pm25_data, "date")

baseline_features = ["latitude", "longitude", "hr_of_day", "day_of_week", "day_of_month"]

weather_features = [
    "wind_speed_mps", "u_wind", "v_wind",
    "air_temp_c", "dew_point_c", "temp_dew_spread_c",
    "slp_hpa", "slp_anom_hpa",
    "visibility_m", "log_visibility",
    "ceiling_m",
]

# Keep rows where weather exists
pm25_data = pm25_data.dropna(subset=weather_features).copy()

# ----------------------------
# Split (block approach)
# ----------------------------
block_indices = np.array_split(pm25_data.index, 4)

pm25_data["block_id"] = 0
for block_id, indices in enumerate(block_indices, start=1):
    pm25_data.loc[indices, "block_id"] = block_id

train_data = pm25_data[pm25_data["block_id"].isin([1, 3])].copy()
gate_data = pm25_data[pm25_data["block_id"] == 2].copy()
val_data = pm25_data[pm25_data["block_id"] == 4].copy()

# ----------------------------
# Train two models
# ----------------------------
baseline_model = RandomForestRegressor(random_state=1)
weather_model = RandomForestRegressor(random_state=1)

baseline_model.fit(train_data[baseline_features], train_data["pm25"])
weather_model.fit(train_data[baseline_features + weather_features], train_data["pm25"])

# ----------------------------
# Tune a single global wind threshold on gate_data (block 2)
# Use soft blending so the regime-aware line looks stable and intentional
# ----------------------------
gate_data = add_grouped_rolling_wind(
    gate_data,
    wind_col="wind_speed_mps",
    group_col="site_id",
    date_col="date",
    window_hours=ROLLING_HOURS,
)

gate_data["pred_baseline"] = baseline_model.predict(gate_data[baseline_features])
gate_data["pred_weather"] = weather_model.predict(gate_data[baseline_features + weather_features])

candidates = np.quantile(gate_data["wind_speed_roll"].dropna(), np.linspace(0.05, 0.95, 19))

best_t = None
best_mae = None

for t in candidates:
    alpha = soft_gate(gate_data["wind_speed_roll"].values, float(t), SOFTNESS_MPS)
    pred = (1.0 - alpha) * gate_data["pred_baseline"].values + alpha * gate_data["pred_weather"].values
    mae = mean_absolute_error(gate_data["pm25"], pred)
    if best_mae is None or mae < best_mae:
        best_mae = mae
        best_t = float(t)

print(f"Chosen soft-gate threshold on rolling wind: {best_t:.3f} m/s")
print(f"Gate tuning MAE (block 2): {best_mae:.3f}")

# ----------------------------
# Evaluate on validation block 4
# ----------------------------
val_data = add_grouped_rolling_wind(
    val_data,
    wind_col="wind_speed_mps",
    group_col="site_id",
    date_col="date",
    window_hours=ROLLING_HOURS,
)

val_results = val_data.copy()
val_results["pred_baseline"] = baseline_model.predict(val_results[baseline_features])
val_results["pred_weather"] = weather_model.predict(val_results[baseline_features + weather_features])

alpha_val = soft_gate(val_results["wind_speed_roll"].values, best_t, SOFTNESS_MPS)
val_results["prediction"] = (1.0 - alpha_val) * val_results["pred_baseline"].values + alpha_val * val_results["pred_weather"].values

mae_baseline = mean_absolute_error(val_results["pm25"], val_results["pred_baseline"])
mae_weather = mean_absolute_error(val_results["pm25"], val_results["pred_weather"])
mae_regime = mean_absolute_error(val_results["pm25"], val_results["prediction"])

print(f"Validation MAE baseline: {mae_baseline:.3f}")
print(f"Validation MAE weather:  {mae_weather:.3f}")
print(f"Validation MAE regime:   {mae_regime:.3f}")

# ----------------------------
# Plot at target location, matching your PM-only plot style
# ----------------------------
site_mask = (
    (val_results["latitude"].sub(TARGET_LAT).abs() < TOL)
    & (val_results["longitude"].sub(TARGET_LON).abs() < TOL)
)
site_results = val_results.loc[site_mask].copy()

if site_results.empty:
    print("No validation rows found for the target lat/lon.")
    print("Try increasing TOL (e.g., 1e-3) or check the exact site coordinates:")
    print(val_results[["latitude", "longitude"]].drop_duplicates().head(25))
else:
    site_results = site_results.sort_values("date")

    window_results = site_results[
        (site_results["date"] >= PLOT_START)
        & (site_results["date"] <= PLOT_END)
    ].copy()

    if window_results.empty:
        print("No rows found in the requested plot window. Adjust PLOT_START/PLOT_END.")
    else:
        plt.figure(figsize=(10, 5))
        plt.plot(window_results["date"], window_results["pm25"], label="Actual PM2.5")
        plt.plot(window_results["date"], window_results["prediction"], label="Predicted PM2.5")

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=35)

        # Match your PM-only plot scale styling
        ax.set_ylim(0, 22.5)
        ax.set_yticks(np.arange(0, 22.5 + 2.5, 2.5))

        plt.xlabel("Date")
        plt.ylabel("PM2.5")
        plt.title("Hourly Actual vs Predicted PM2.5 (Jan 27 - Jan 31, 2025)")
        plt.legend()
        plt.tight_layout()
        plot_path = FIGURES_DIR / "new_rfr_prediction.png"
        plt.savefig(plot_path, dpi=300)
        print(f"Saved plot: {plot_path}")
        plt.show()
