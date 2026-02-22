from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

input_path = INTERIM_DIR / "la_pm25_0125_combined.csv"
output_path = INTERIM_DIR / "la_pm25(nonegative)_0125_combined.csv"

read_data = pd.read_csv(input_path)
# print(read_data.head())

read_data["timestamp_local"] = pd.to_datetime(read_data["timestamp_local"])
# print("\n\n")
# print(read_data.head())
# print("\n\n")
# print(read_data["timestamp_local"].dtype)

read_data["hr_of_day"] = read_data["timestamp_local"].dt.hour
read_data["day_of_week"] = read_data["timestamp_local"].dt.weekday
read_data["day_of_month"] = read_data["timestamp_local"].dt.day
# print("\n\n")
# print(read_data.head())

mask = read_data["pm25"] < 0
i = mask.sum()
read_data.loc[read_data["pm25"] < 0, "pm25"] = 0
print(i)
read_data.to_csv(output_path, index=False)

print(read_data["latitude"].isna().any())
print(read_data["longitude"].isna().any())
print(read_data["pm25"].isna().any())
print(read_data["hr_of_day"].isna().any())
print(read_data["day_of_week"].isna().any())
print(read_data["day_of_month"].isna().any())
