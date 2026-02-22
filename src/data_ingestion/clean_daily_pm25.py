import pandas as pd

in_daily = r"C:\Users\heman\Downloads\daily_88101_2024\daily_88101_2024.csv"
out_aug = "la_pm25_daily_aug_2024.csv"

STATE_CA = 6
COUNTY_LA = 37
CHUNKSIZE = 500_000

first = True
unique_sites = set()
rows_written = 0

for chunk in pd.read_csv(in_daily, chunksize=CHUNKSIZE, low_memory=False):

    # 1) Filter LA County first
    chunk = chunk[(chunk["State Code"] == STATE_CA) & (chunk["County Code"] == COUNTY_LA)]
    if chunk.empty:
        continue

    # 2) Filter August 2024 (simple and fast)
    chunk = chunk[chunk["Date Local"].astype(str).str.startswith("2024-08")]
    if chunk.empty:
        continue

    # 3) Build site_id
    chunk["site_id"] = (
        chunk["State Code"].astype(int).astype(str).str.zfill(2) + "_" +
        chunk["County Code"].astype(int).astype(str).str.zfill(3) + "_" +
        chunk["Site Num"].astype(int).astype(str).str.zfill(4)
    )

    # Track unique sites
    unique_sites.update(chunk["site_id"].unique())

    # 4) Keep only what you need
    out = chunk[["site_id", "Latitude", "Longitude", "Date Local", "Arithmetic Mean"]].rename(
        columns={
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Date Local": "date_local",
            "Arithmetic Mean": "pm25_daily"
        }
    )

    out["pm25_daily"] = pd.to_numeric(out["pm25_daily"], errors="coerce")
    out = out.dropna(subset=["latitude", "longitude", "date_local", "pm25_daily"])

    out.to_csv(out_aug, mode="w" if first else "a", index=False, header=first)
    first = False

    rows_written += len(out)

print("done:", out_aug)
print("total rows written:", rows_written)
print("unique sites in August daily data:", len(unique_sites))

df = pd.read_csv(out_aug)
print(df.groupby("site_id")["date_local"].nunique().describe())
