from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 1) Set your input files here
in_frm_fem = INTERIM_DIR / "la_pm25FF_hrly_0125.csv"
in_non_frm_fem = INTERIM_DIR / "la_pm25nFF_hrly_0125.csv"

# 2) Output combined file
out_combined = INTERIM_DIR / "la_pm25_0925_combined.csv"

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make the two files consistent so they can be combined.
    Expected final columns:
    site_id, latitude, longitude, timestamp_local, pm25
    """
    # Normalize column names
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Build a rename map for common variants
    rename_map = {}

    for c in df.columns:
        cl = c.strip().lower()

        if cl in ["lat", "latitude"]:
            rename_map[c] = "latitude"
        elif cl in ["lon", "long", "longitude"]:
            rename_map[c] = "longitude"
        elif cl in ["timestamp_local", "datetime_local", "time_local", "timestamp"]:
            rename_map[c] = "timestamp_local"
        elif cl in ["pm25", "pm2.5", "sample measurement", "sample_measurement", "value"]:
            rename_map[c] = "pm25"
        elif cl in ["site_id", "siteid"]:
            rename_map[c] = "site_id"

    df = df.rename(columns=rename_map)

    required = ["site_id", "latitude", "longitude", "timestamp_local", "pm25"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns after standardization: "
            + ", ".join(missing)
            + "\nColumns found: "
            + ", ".join(df.columns)
        )

    # Coerce types
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"], errors="coerce")

    # Drop unusable rows
    df = df.dropna(subset=["site_id", "latitude", "longitude", "timestamp_local", "pm25"])

    # Make site_id string for safety
    df["site_id"] = df["site_id"].astype(str)

    return df

# Read and standardize
frm = standardize(pd.read_csv(in_frm_fem, low_memory=False))
frm["pm25_source"] = "frm_fem"

non = standardize(pd.read_csv(in_non_frm_fem, low_memory=False))
non["pm25_source"] = "non_frm_fem"

# Combine
combined = pd.concat([frm, non], ignore_index=True)

# Save combined dataset
combined.to_csv(out_combined, index=False)

# Build unique monitor location tables for plotting
frm_sites = frm[["site_id", "latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
non_sites = non[["site_id", "latitude", "longitude"]].drop_duplicates().reset_index(drop=True)

# Some sites might appear in both files. That is fine.
all_sites = combined[["site_id", "latitude", "longitude", "pm25_source"]].drop_duplicates()

print()
print("Saved combined file:", out_combined)
print("Rows FRM_FEM:", len(frm))
print("Rows NON_FRM_FEM:", len(non))
print("Rows combined:", len(combined))
print()
print("Unique sites FRM_FEM:", frm_sites["site_id"].nunique())
print("Unique sites NON_FRM_FEM:", non_sites["site_id"].nunique())
print("Unique sites combined:", combined["site_id"].nunique())

# Plot locations
plt.figure(figsize=(10, 6))

plt.scatter(frm_sites["longitude"], frm_sites["latitude"], s=35, alpha=0.7, label="FRM/FEM sites")
plt.scatter(non_sites["longitude"], non_sites["latitude"], s=35, alpha=0.7, label="Non-FRM/FEM sites")

plt.xlabel("Longitude", labelpad=15)
plt.ylabel("Latitude", labelpad=15)
plt.title("LA PM2.5 Monitor Locations: FRM/FEM vs Non-FRM/FEM", pad=20)

plt.legend()
plot_path = FIGURES_DIR / "la_pm25_monitor_locations.png"
plt.savefig(plot_path, dpi=300)
print(f"Saved plot: {plot_path}")
plt.show()
