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

# Plot window (kept for filtering, but NOT shown in title)
PLOT_START = pd.Timestamp("2025-01-27 00:00:00")
PLOT_END = pd.Timestamp("2025-01-31 00:00:00")

# Regime gate smoothing settings
ROLLING_HOURS = 6
SOFTNESS_MPS = 0.35

# Axis styling
Y_MAX = 80.0
Y_TICK_STEP = 10.0

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

def plot_site(
    val_results,
    site_id,
    plot_start,
    plot_end,
    y_max,
    y_tick_step,
    headline,
    show_details=True
):
    site_df = val_results[val_results["site_id"] == site_id].copy()
    if site_df.empty:
        print(f"No validation rows for site_id={site_id}")
        return

    site_df = site_df.sort_values("date")
    window_df = site_df[(site_df["date"] >= plot_start) & (site_df["date"] <= plot_end)].copy()
    if window_df.empty:
        print(f"No rows for site_id={site_id} in plot window. Try adjusting PLOT_START/PLOT_END.")
        return

    lat = float(site_df["latitude"].iloc[0])
    lon = float(site_df["longitude"].iloc[0])

    mae_base = mean_absolute_error(site_df["pm25"], site_df["pred_baseline"])
    mae_wx = mean_absolute_error(site_df["pm25"], site_df["pred_weather"])
    delta = mae_wx - mae_base  # weather MAE minus baseline MAE (negative = improvement)

    plt.figure(figsize=(10, 5))
    plt.plot(window_df["date"], window_df["pm25"], label="Actual PM2.5")
    plt.plot(window_df["date"], window_df["prediction"], label="Regime-aware prediction")

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=35)

    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + y_tick_step, y_tick_step))

    plt.xlabel("Date")
    plt.ylabel("PM2.5")

    # Clean title: no date range in title
    plt.title(headline)

    # Small, unambiguous detail line (vs baseline explicitly)
    if show_details:
        detail = (
            f"site {site_id} | ({lat:.5f}, {lon:.5f}) | "
            f"MAE change vs baseline = {delta:+.2f} ug/m3"
        )
        ax.text(
            0.01, 0.02,
            detail,
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            alpha=0.85
        )

    plt.legend()
    plt.tight_layout()
    plot_path = FIGURES_DIR / f"site_{site_id}_regime_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot for site {site_id}: {plot_path}")
    plt.show()

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
# Find one site where weather helps and one where it hurts (val block)
# Uses baseline vs weather MAE per site on val block
# ----------------------------
site_rows = []
for sid, g in val_results.groupby("site_id"):
    if len(g) < 10:
        continue
    b = mean_absolute_error(g["pm25"], g["pred_baseline"])
    w = mean_absolute_error(g["pm25"], g["pred_weather"])
    site_rows.append({"site_id": sid, "baseline_mae": b, "weather_mae": w, "delta": w - b})

site_summary = pd.DataFrame(site_rows).sort_values("delta")

if site_summary.empty:
    print("No per-site summary could be computed. Check val_results.")
else:
    best_site = site_summary.iloc[0]["site_id"]       # most negative delta (weather helps)
    worst_site = site_summary.iloc[-1]["site_id"]     # most positive delta (weather hurts)

    print("\nSelected sites (val block baseline vs weather MAE):")
    print(site_summary.head(3))
    print(site_summary.tail(3))
    print(f"\nWeather helps most at:  {best_site}")
    print(f"Weather hurts most at:  {worst_site}")

    plot_site(
        val_results,
        best_site,
        PLOT_START,
        PLOT_END,
        Y_MAX,
        Y_TICK_STEP,
        headline="Weather helps here: Actual vs Regime-aware PM2.5",
        show_details=True
    )

    plot_site(
        val_results,
        worst_site,
        PLOT_START,
        PLOT_END,
        Y_MAX,
        Y_TICK_STEP,
        headline="Weather hurts here: Actual vs Regime-aware PM2.5",
        show_details=True
    )
