from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

pm25_path = INTERIM_DIR / "la_pm25(nonegative)_0125_combined.csv"
pm25_data = pd.read_csv(pm25_path)
# pm25_data is assumed to be sorted by timestamp_local.

num_rows = len(pm25_data)
block_indices = np.array_split(pm25_data.index, 4)

pm25_data = pm25_data.copy()
pm25_data["block_id"] = 0
for block_id, indices in enumerate(block_indices, start=1):
    pm25_data.loc[indices, "block_id"] = block_id

feature_columns = ["latitude", "longitude", "hr_of_day", "day_of_week", "day_of_month"]

train_data = pm25_data[pm25_data["block_id"].isin([1, 3])]
val_data = pm25_data[pm25_data["block_id"] == 4]

train_x = train_data[feature_columns]
train_y = train_data["pm25"]
val_x = val_data[feature_columns]
val_y = val_data["pm25"]

model = RandomForestRegressor(random_state=1)
model.fit(train_x, train_y)

val_predictions = model.predict(val_x)
mae = mean_absolute_error(val_y, val_predictions)
print(f"Validation MAE: {mae:.3f}")

val_results = val_data.copy()
val_results["prediction"] = val_predictions

site_columns = ["latitude", "longitude"]
site_key = val_results[site_columns].drop_duplicates().iloc[0]
lat = site_key["latitude"]
lon = site_key["longitude"]
print(f"Plotting site at latitude={lat:.5f}, longitude={lon:.5f}")
site_mask = (val_results["latitude"] == site_key["latitude"]) & (
    val_results["longitude"] == site_key["longitude"]
)
site_results = val_results.loc[site_mask].copy()
site_results["timestamp_local"] = pd.to_datetime(site_results["timestamp_local"])
site_results = site_results.sort_values("timestamp_local")

min_timestamp = site_results["timestamp_local"].min().normalize()
max_timestamp = site_results["timestamp_local"].max()
candidate_start = min_timestamp + pd.offsets.Week(weekday=0)
window_start = candidate_start if candidate_start <= max_timestamp else min_timestamp
window_end = window_start + pd.Timedelta(days=4)
window_results = site_results[
    (site_results["timestamp_local"] >= window_start)
    & (site_results["timestamp_local"] <= window_end)
].copy()

plt.figure(figsize=(10, 5))
plt.plot(
    window_results["timestamp_local"],
    window_results["pm25"],
    label="Actual PM2.5",
)
plt.plot(
    window_results["timestamp_local"],
    window_results["prediction"],
    label="Predicted PM2.5",
)
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=35)
plt.xlabel("Date", labelpad=10)
plt.ylabel("PM2.5", labelpad=10)
date_range = f"{window_start:%b %d} - {window_end:%b %d, %Y}"
plt.title(f"Hourly Actual vs Predicted PM2.5 ({date_range})")
plt.legend()
plt.tight_layout()
figure_path = FIGURES_DIR / "initial_rfr_prediction.png"
plt.savefig(figure_path, dpi=300)
print(f"Saved plot: {figure_path}")
plt.show()
