from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

in_hourly = RAW_DIR / "hourly_88101_2025.csv"
out_recent = INTERIM_DIR / "la_pm25FF_hrly_0125.csv"

state_CA = 6
county_LA = 37
chunksize = 500_000

def make_site_id(state, county, site):
    return (
        state.astype("int64").astype(str).str.zfill(2) + "_" +
        county.astype("int64").astype(str).str.zfill(3) + "_" +
        site.astype("int64").astype(str).str.zfill(4)
    )

first = True
unique_sites = set()
rows_written = 0

for i, chunk in enumerate(pd.read_csv(in_hourly, chunksize=chunksize, low_memory=False), start=1):

    chunk.columns = chunk.columns.str.strip()

    chunk["State Code"] = pd.to_numeric(chunk["State Code"], errors="coerce")
    chunk["County Code"] = pd.to_numeric(chunk["County Code"], errors="coerce")
    chunk["Site Num"] = pd.to_numeric(chunk["Site Num"], errors="coerce")

    chunk["Date Local"] = pd.to_datetime(chunk["Date Local"], errors="coerce")

    chunk = chunk[(chunk["State Code"] == state_CA) & (chunk["County Code"] == county_LA)]
    if chunk.empty:
        continue

    chunk = chunk[(chunk["Date Local"] >= "2025-01-01") & (chunk["Date Local"] <= "2025-01-30")]
    if chunk.empty:
        continue

    chunk["site_id"] = make_site_id(chunk["State Code"], chunk["County Code"], chunk["Site Num"])

    # track unique sites present in August data
    unique_sites.update(chunk["site_id"].dropna().unique())

    chunk["timestamp_local"] = pd.to_datetime(
        chunk["Date Local"].dt.strftime("%Y-%m-%d") + " " + chunk["Time Local"].astype(str),
        errors="coerce"
    )

    out = chunk[["site_id", "Latitude", "Longitude", "timestamp_local", "Sample Measurement"]].rename(
        columns={"Latitude": "latitude", "Longitude": "longitude", "Sample Measurement": "pm25"}
    )

    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out["pm25"] = pd.to_numeric(out["pm25"], errors="coerce")

    out = out.dropna(subset=["latitude", "longitude", "timestamp_local", "pm25"])

    out.to_csv(out_recent, mode="w" if first else "a", index=False, header=first)
    first = False

    rows_written += len(out)

    # light progress update every 10 chunks
    if i % 10 == 0:
        print(f"Processed {i} chunks | Rows written so far: {rows_written:,} | Unique sites so far: {len(unique_sites)}")

print()
print("done:", out_recent)
print(f"Total rows written: {rows_written:,}")
print(f"Unique sites in data: {len(unique_sites)}")
