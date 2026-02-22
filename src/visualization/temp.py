from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

input_data = INTERIM_DIR / "pm_site_to_weather_knn_k5.csv"
df_input = pd.read_csv(input_data, low_memory=False)
output_data = INTERIM_DIR / "pm_site_to_weather_weights_k5.csv"


e = 1e-3
p = 2

weight_raw = 1.0 / (df_input["distance_km"] + e) ** p

# 2) Per-site denominator (broadcast back to each row)
den = df_input.groupby("site_id")["distance_km"].transform(lambda s: (1.0 / (s + e) ** p).sum())

# 3) Normalized weights
df_input["weight"] = weight_raw / den

print(df_input[["site_id","weight"]].head(5))

df_input.to_csv(output_data, index=False)
print("Saved weights to:", output_data)
