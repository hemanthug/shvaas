from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

in_sites = RAW_DIR / "aqs_sites.csv"
out_sites = INTERIM_DIR / "la_sites.csv"

state_CA = 6
county_LA = 37

def make_site_id(state, county, site):
    return (
        state.astype("int64").astype(str).str.zfill(2) + "_" +
        county.astype("int64").astype(str).str.zfill(3) + "_" +
        site.astype("int64").astype(str).str.zfill(4)
    )

sites = pd.read_csv(in_sites, low_memory=False)

# fix hidden spaces in column names
sites.columns = sites.columns.str.strip()

# force codes to numeric so comparisons work
sites["State Code"] = pd.to_numeric(sites["State Code"], errors="coerce")
sites["County Code"] = pd.to_numeric(sites["County Code"], errors="coerce")
sites["Site Number"] = pd.to_numeric(sites["Site Number"], errors="coerce")

# keep only LA County, CA
sites = sites[(sites["State Code"] == state_CA) & (sites["County Code"] == county_LA)].copy()

print("Rows after CA+LA filter:", len(sites))
print()

# build stable site id
sites["Site_ID"] = make_site_id(sites["State Code"], sites["County Code"], sites["Site Number"])

# keep only what you need
sites = sites[["Site_ID", "State Code", "County Code", "Site Number", "Latitude", "Longitude"]]

# clean lat lon
sites["Latitude"] = pd.to_numeric(sites["Latitude"], errors="coerce")
sites["Longitude"] = pd.to_numeric(sites["Longitude"], errors="coerce")

sites = (
    sites.dropna(subset=["Latitude", "Longitude"])
         .drop_duplicates(subset=["Site_ID"])
         .reset_index(drop=True)
)

sites["State Code"] = sites["State Code"].astype(int)
sites["County Code"] = sites["County Code"].astype(int)
sites["Site Number"] = sites["Site Number"].astype(int)

sites.to_csv(out_sites, index=False)

print(f"Saved {len(sites)} LA sites to {out_sites}")
print()
print(sites.head().to_string(index=False))

plt.figure(figsize=(10, 6))
plt.scatter(sites["Longitude"], sites["Latitude"], c = "red",  s=30, alpha=0.5)
plt.xlabel("Longitude", labelpad=15)
plt.ylabel("Latitude", labelpad=15)
plt.title("LA County AQI Monitoring Sites", pad=20)
plot_path = FIGURES_DIR / "la_aqi_monitoring_sites.png"
plt.savefig(plot_path, dpi=300)
print(f"Saved plot: {plot_path}")

plt.show()
