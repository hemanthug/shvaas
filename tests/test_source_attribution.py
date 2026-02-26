import pandas as pd

from src.source_attribution.candidates import generate_candidate_grid
from src.source_attribution.kernel import compute_influence


def test_compute_influence_directionality():
    # Wind from north (0 deg) transports plume southward.
    pos = compute_influence(
        sensor_lat=33.95,
        sensor_lon=-118.25,
        source_lat=34.00,
        source_lon=-118.25,
        wind_speed=4.0,
        wind_dir_deg=0.0,
        L_km=10.0,
        sigma_km=3.0,
    )
    zero = compute_influence(
        sensor_lat=34.05,
        sensor_lon=-118.25,
        source_lat=34.00,
        source_lon=-118.25,
        wind_speed=4.0,
        wind_dir_deg=0.0,
        L_km=10.0,
        sigma_km=3.0,
    )
    assert pos > 0.0
    assert zero == 0.0


def test_generate_candidate_grid_non_empty():
    stations = pd.DataFrame(
        {
            "latitude": [34.02, 34.08, 34.12],
            "longitude": [-118.40, -118.30, -118.25],
        }
    )
    grid = generate_candidate_grid(stations_df=stations, grid_spacing_km=5.0)
    assert not grid.empty
    assert {"source_id", "latitude", "longitude"}.issubset(set(grid.columns))

