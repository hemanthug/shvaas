from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
in_sites = RAW_DIR / "aqs_sites.csv"
in_frm_fem = INTERIM_DIR / "la_pm25FF_hrly_0125.csv"
in_non_frm_fem = INTERIM_DIR / "la_pm25nFF_hrly_0125.csv"

out_sites = INTERIM_DIR / "la_sites.csv"
out_combined = INTERIM_DIR / "la_pm25_0925_combined.csv"

state_CA = 6
county_LA = 37

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def make_site_id(state, county, site):
    return (
        state.astype("int64").astype(str).str.zfill(2) + "_" +
        county.astype("int64").astype(str).str.zfill(3) + "_" +
        site.astype("int64").astype(str).str.zfill(4)
    )

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make the two files consistent so they can be combined.
    Expected final columns:
    site_id, latitude, longitude, timestamp_local, pm25
    """
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
    df["site_id"] = df["site_id"].astype(str)

    return df

# ============================================================================
# PART 1: PROCESS SITE METADATA
# ============================================================================
print("=" * 60)
print("PROCESSING SITE METADATA")
print("=" * 60)

sites = pd.read_csv(in_sites, low_memory=False)
sites.columns = sites.columns.str.strip()

# Force codes to numeric
sites["State Code"] = pd.to_numeric(sites["State Code"], errors="coerce")
sites["County Code"] = pd.to_numeric(sites["County Code"], errors="coerce")
sites["Site Number"] = pd.to_numeric(sites["Site Number"], errors="coerce")

# Keep only LA County, CA
sites = sites[(sites["State Code"] == state_CA) & (sites["County Code"] == county_LA)].copy()
print(f"Rows after CA+LA filter: {len(sites)}")

# Build stable site ID
sites["Site_ID"] = make_site_id(sites["State Code"], sites["County Code"], sites["Site Number"])

# Keep only what you need
sites = sites[["Site_ID", "State Code", "County Code", "Site Number", "Latitude", "Longitude"]]

# Clean lat/lon
sites["Latitude"] = pd.to_numeric(sites["Latitude"], errors="coerce")
sites["Longitude"] = pd.to_numeric(sites["Longitude"], errors="coerce")

sites = (
    sites.dropna(subset=["Latitude", "Longitude"])
         .drop_duplicates(subset=["Site_ID"])
         .reset_index(drop=True)
)

# Save site metadata
sites.to_csv(out_sites, index=False)
print(f"Saved site metadata: {out_sites}")
print(f"Unique sites: {len(sites)}\n")

# ============================================================================
# PART 2: PROCESS PM2.5 MEASUREMENT DATA
# ============================================================================
print("=" * 60)
print("PROCESSING PM2.5 MEASUREMENT DATA")
print("=" * 60)

# Read and standardize
frm = standardize(pd.read_csv(in_frm_fem, low_memory=False))
frm["pm25_source"] = "frm_fem"

non = standardize(pd.read_csv(in_non_frm_fem, low_memory=False))
non["pm25_source"] = "non_frm_fem"

# Combine
combined = pd.concat([frm, non], ignore_index=True)

# Save combined dataset
combined.to_csv(out_combined, index=False)

# Build unique monitor location tables
frm_sites = frm[["site_id", "latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
non_sites = non[["site_id", "latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
all_pm25_sites = combined[["site_id", "latitude", "longitude"]].drop_duplicates().reset_index(drop=True)

print(f"Saved combined file: {out_combined}")
print(f"Rows FRM_FEM: {len(frm)}")
print(f"Rows NON_FRM_FEM: {len(non)}")
print(f"Rows combined: {len(combined)}\n")
print(f"Unique sites FRM_FEM: {frm_sites['site_id'].nunique()}")
print(f"Unique sites NON_FRM_FEM: {non_sites['site_id'].nunique()}")
print(f"Unique sites combined: {combined['site_id'].nunique()}\n")

# ============================================================================
# PART 3: VISUALIZATION - ALL SITES WITH PM2.5 HIGHLIGHTED
# ============================================================================
print("=" * 60)
print("CREATING VISUALIZATION")
print("=" * 60)

plt.figure(figsize=(12, 8))

# Plot ALL LA County monitoring sites (background) - RED
plt.scatter(sites["Longitude"], sites["Latitude"],
            s=50, alpha=0.3, color='red',
            label=f'All LA monitoring sites (n={len(sites)})',
            marker='o')

# Overlay PM2.5 sites with distinct colors - BLUE and GREEN, all circles
plt.scatter(frm_sites["longitude"], frm_sites["latitude"],
            s=80, alpha=0.8, color='blue',
            label=f'FRM/FEM PM2.5 sites (n={len(frm_sites)})',
            marker='o', edgecolors='darkblue', linewidths=1.5)

plt.scatter(non_sites["longitude"], non_sites["latitude"],
            s=80, alpha=0.8, color='green',
            label=f'Non-FRM/FEM PM2.5 sites (n={len(non_sites)})',
            marker='o', edgecolors='darkgreen', linewidths=1.5)

plt.xlabel("Longitude", fontsize=12, labelpad=15)
plt.ylabel("Latitude", fontsize=12, labelpad=15)
plt.title("LA County Air Quality Monitoring Sites", fontsize=14, pad=20)

plt.legend(loc='best', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
figure_path = FIGURES_DIR / "la_monitoring_sites_combined.png"
plt.savefig(figure_path, dpi=300)
print(f"Saved plot: {figure_path}")
plt.show()

print("Done!")
