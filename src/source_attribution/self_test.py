from __future__ import annotations

import pandas as pd

from .candidates import generate_candidate_grid
from .kernel import compute_influence


def run_self_tests() -> int:
    failures = 0

    # Wind from north (0 deg) blows toward south.
    positive = compute_influence(
        sensor_lat=33.95,
        sensor_lon=-118.25,
        source_lat=34.00,
        source_lon=-118.25,
        wind_speed=3.0,
        wind_dir_deg=0.0,
        L_km=10.0,
        sigma_km=3.0,
    )
    zero = compute_influence(
        sensor_lat=34.05,
        sensor_lon=-118.25,
        source_lat=34.00,
        source_lon=-118.25,
        wind_speed=3.0,
        wind_dir_deg=0.0,
        L_km=10.0,
        sigma_km=3.0,
    )

    if not (positive > 0):
        print("FAIL: expected positive influence for downwind sensor.")
        failures += 1
    if not (zero == 0):
        print("FAIL: expected zero influence for upwind sensor.")
        failures += 1

    stations = pd.DataFrame(
        {
            "latitude": [34.0, 34.1],
            "longitude": [-118.3, -118.2],
        }
    )
    grid = generate_candidate_grid(stations_df=stations, grid_spacing_km=5.0)
    if grid.empty:
        print("FAIL: candidate grid should not be empty.")
        failures += 1

    if failures == 0:
        print("PASS: all source attribution self-tests passed.")
    return failures


if __name__ == "__main__":
    raise SystemExit(run_self_tests())

