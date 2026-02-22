from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

in_weather_data = pd.read_csv(RAW_DIR / "weather_hourly_jan2025_noaa_v1.csv", low_memory=False)
out_weather_data = INTERIM_DIR / "weather_hourly_jan2025_decoded_v1.csv"

def convert_column_data(col, token, sentinel, scale):
    temp = col.astype("string").str.split(",")
    x = pd.to_numeric(temp.str[token], errors="coerce").replace(sentinel, np.nan)
    return x * scale

in_weather_data["wind_dir_deg"] = convert_column_data(in_weather_data["WND"], 0, 999, 1.0)
in_weather_data["wind_speed_mps"] = convert_column_data(in_weather_data["WND"], 3, 9999, 0.1)
in_weather_data["air_temp_c"] = convert_column_data(in_weather_data["TMP"], 0, [9999, -9999], 0.1)
in_weather_data["dew_point_c"] = convert_column_data(in_weather_data["DEW"], 0, [9999, -9999], 0.1)
in_weather_data["slp_hpa"] = convert_column_data(in_weather_data["SLP"], 0, 99999, 0.1)
in_weather_data["visibility_m"] = convert_column_data(in_weather_data["VIS"], 0, 999999, 1.0)
in_weather_data["ceiling_m"] = convert_column_data(in_weather_data["CIG"], 0, 99999, 1.0)

in_weather_data["wind_dir_rad"] = in_weather_data["wind_dir_deg"] * np.pi / 180.0
in_weather_data["wind_dir_sin"] = np.sin(in_weather_data["wind_dir_rad"])
in_weather_data["wind_dir_cos"] = np.cos(in_weather_data["wind_dir_rad"])

# per your existing convention: u = speed*sin, v = speed*cos
in_weather_data["u_wind"] = in_weather_data["wind_speed_mps"] * in_weather_data["wind_dir_sin"]
in_weather_data["v_wind"] = in_weather_data["wind_speed_mps"] * in_weather_data["wind_dir_cos"]


in_weather_data["temp_dew_spread_c"] = in_weather_data["air_temp_c"] - in_weather_data["dew_point_c"]
in_weather_data["slp_anom_hpa"] = in_weather_data["slp_hpa"] - in_weather_data.groupby("station_id")["slp_hpa"].transform("mean")
in_weather_data["log_visibility"] = np.log1p(in_weather_data["visibility_m"])

# build output with lowercase names requested
out_df = pd.DataFrame({
    "date": in_weather_data["DATE"],
    "station_id": in_weather_data["station_id"],
    "latitude": in_weather_data["LATITUDE   "],
    "longitude": in_weather_data["LONGITUDE"],
    "wind_speed_mps": in_weather_data["wind_speed_mps"],
    "u_wind": in_weather_data["u_wind"],
    "v_wind": in_weather_data["v_wind"],
    "air_temp_c": in_weather_data["air_temp_c"],
    "dew_point_c": in_weather_data["dew_point_c"],
    "temp_dew_spread_c": in_weather_data["temp_dew_spread_c"],
    "slp_hpa": in_weather_data["slp_hpa"],
    "slp_anom_hpa": in_weather_data["slp_anom_hpa"],
    "visibility_m": in_weather_data["visibility_m"],
    "log_visibility": in_weather_data["log_visibility"],
    "ceiling_m": in_weather_data["ceiling_m"],
})

print(out_df.columns)
print(out_df.head())
out_df.to_csv(out_weather_data, index=False)

