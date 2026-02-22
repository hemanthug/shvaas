from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SITES_FILE = INTERIM_DIR / "la_sites.csv"
PM25_COMBINED = INTERIM_DIR / "la_pm25(nonegative)_0125_combined.csv"

sites = pd.read_csv(SITES_FILE)
pm = pd.read_csv(PM25_COMBINED)

sites.columns = sites.columns.str.strip()
pm.columns = pm.columns.str.strip()

sites["site_id"] = sites["site_id"].astype(str)
pm["site_id"] = pm["site_id"].astype(str)

sites_xy = sites[["site_id", "Longitude", "Latitude"]].drop_duplicates()
pm_xy = pm[["site_id", "latitude", "longitude"]].drop_duplicates()
pm_xy = pm_xy.rename(columns={"longitude": "Longitude", "latitude": "Latitude"})

sites_set = set(sites_xy["site_id"])
pm_set = set(pm_xy["site_id"])

print()
print("LA sites in sites.csv:", len(sites_set))
print("PM2.5 sites in combined data:", len(pm_set))
print("Overlap:", len(sites_set & pm_set))
print("PM2.5 sites not found in sites.csv:", len(pm_set - sites_set))
print("Example missing ids:", list(pm_set - sites_set)[:10])

plt.figure(figsize=(14, 6))

plt.scatter(sites_xy["Longitude"], sites_xy["Latitude"], c="green", s=20, alpha=0.25, label="All LA County sites")
plt.scatter(pm_xy["Longitude"], pm_xy["Latitude"], c="red", s=30, alpha=0.9, label="PM2.5 sites")

plt.xlabel("Longitude", labelpad=15)
plt.ylabel("Latitude", labelpad=15)
plt.title("LA County Sites vs PM2.5 Reporting Sites", pad=20)
plt.legend()
plot_path = FIGURES_DIR / "la_sites_vs_pm25_sites.png"
plt.savefig(plot_path, dpi=300)
print(f"Saved plot: {plot_path}")
plt.show()
