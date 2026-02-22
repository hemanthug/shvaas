from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
STATION_FOLDER = RAW_DIR / "noaa_station_files"
STATION_FOLDER.mkdir(parents=True, exist_ok=True)

folder = STATION_FOLDER

# Only read what you need
columns = [
    "DATE",
    "STATION",
    "LATITUDE",
    "LONGITUDE",
    "WND",
    "TMP",
    "DEW",
    "SLP",
    "VIS",
    "CIG",
]

start = pd.Timestamp("2025-01-01")
end = pd.Timestamp("2025-02-01")

results = []
files = list(folder.glob("*.csv"))

total_files = 0
files_with_data = 0
total_hourly_rows = 0

for file in tqdm(files, desc="Files", unit="file"):
    total_files += 1

    # Accumulator for this station file: one row per hour_bucket (eventually)
    best = None

    for chunk in pd.read_csv(
        file,
        usecols=columns,
        chunksize=500_000,
        low_memory=False,
    ):
        # Parse timestamps, drop bad ones early
        chunk["DATE"] = pd.to_datetime(chunk["DATE"], errors="coerce")
        chunk = chunk.dropna(subset=["DATE"])

        # Filter to January
        chunk = chunk[(chunk["DATE"] >= start) & (chunk["DATE"] < end)]
        if chunk.empty:
            continue

        # Add hour bucket
        chunk["hour_bucket"] = chunk["DATE"].dt.floor("h")

        # Reduce THIS chunk to last obs per hour (by max DATE inside the hour)
        idx = chunk.groupby("hour_bucket")["DATE"].idxmax()
        chunk_hourly = chunk.loc[idx].copy()

        # Keep/standardize station_id now
        chunk_hourly["station_id"] = chunk_hourly["STATION"]

        # Keep only the columns we want + hour_bucket for merging
        chunk_hourly = chunk_hourly[
            [
                "hour_bucket",
                "DATE",
                "station_id",
                "LATITUDE",
                "LONGITUDE",
                "WND",
                "TMP",
                "DEW",
                "SLP",
                "VIS",
                "CIG",
            ]
        ]

        # Merge into per-file accumulator and re-reduce to last per hour_bucket
        if best is None:
            best = chunk_hourly
        else:
            best = pd.concat([best, chunk_hourly], ignore_index=True)

            # If multiple candidates exist for the same hour_bucket, keep the latest DATE
            idx_best = best.groupby("hour_bucket")["DATE"].idxmax()
            best = best.loc[idx_best].copy()

    # Done reading chunks for this file
    if best is None or best.empty:
        continue

    # Make the dataset truly hourly: timestamp = hour_bucket
    best["DATE"] = best["hour_bucket"]
    best = best.drop(columns=["hour_bucket"]).sort_values("DATE")

    results.append(best)
    files_with_data += 1
    total_hourly_rows += len(best)

# Combine all stations
if results:
    aggregate = pd.concat(results, ignore_index=True)
else:
    aggregate = pd.DataFrame(
        columns=[
            "DATE",
            "station_id",
            "LATITUDE",
            "LONGITUDE",
            "WND",
            "TMP",
            "DEW",
            "SLP",
            "VIS",
            "CIG",
        ]
    )

print("Total files:", total_files)
print("Files with Jan data:", files_with_data)
print("Total hourly rows:", len(aggregate))
print("Unique stations:", aggregate["station_id"].nunique())
print("Min DATE:", aggregate["DATE"].min())
print("Max DATE:", aggregate["DATE"].max())

#quick duplicate check (should be 0)
dupes = aggregate.duplicated(subset=["station_id", "DATE"]).sum()
print("Duplicate station_id+DATE rows:", dupes)

output_path = RAW_DIR / "weather_hourly_jan2025_noaa_v1.csv"
aggregate.to_csv(output_path, index=False)
