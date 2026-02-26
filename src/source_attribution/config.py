from dataclasses import dataclass


@dataclass(frozen=True)
class AttributionConfig:
    """Configuration for source attribution inference and feature generation."""

    grid_spacing_km: float = 5.0
    window_hours: int = 6
    spike_quantile: float = 0.90
    kernel_L_km: float = 10.0
    kernel_sigma_km: float = 3.0
    top_k_sources: int = 10
    elasticnet_alpha: float = 0.001
    elasticnet_l1_ratio: float = 0.8
    random_state: int = 42
    output_dir: str = "reports/figures/source_attribution"
    artifacts_dir: str = "data/processed/source_attribution"

