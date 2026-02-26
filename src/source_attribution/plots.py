from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import AttributionConfig


def plot_stations_and_sources(
    stations_df: pd.DataFrame,
    top_sources_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        stations_df["longitude"],
        stations_df["latitude"],
        c="tab:blue",
        s=40,
        label="Stations",
        alpha=0.8,
    )
    ax.scatter(
        top_sources_df["longitude"],
        top_sources_df["latitude"],
        c="tab:red",
        s=60,
        label="Top inferred sources",
        alpha=0.9,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Stations and Top Inferred Source Candidates")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "stations_top_sources_map.png", dpi=250)
    plt.close(fig)


def plot_top_weights(top_sources_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(top_sources_df))
    ax.bar(x, top_sources_df["total_weight"].to_numpy(dtype=float), color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{int(sid)}" for sid in top_sources_df["source_id"]], rotation=45)
    ax.set_ylabel("Total inferred weight")
    ax.set_title("Top source weights")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "top_source_weights.png", dpi=250)
    plt.close(fig)


def plot_cumulative_share(ranked_sources_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    n = len(ranked_sources_df)
    x = np.arange(1, n + 1)
    y = ranked_sources_df["cumulative_share"].to_numpy(dtype=float)
    ax.plot(x, y, color="tab:green", linewidth=2)
    ax.set_xlabel("Ranked source index")
    ax.set_ylabel("Cumulative share")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.2)
    ax.set_title("Cumulative share of inferred source weight")
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_share_curve.png", dpi=250)
    plt.close(fig)


def save_all_plots(
    stations_df: pd.DataFrame,
    ranked_sources_df: pd.DataFrame,
    config: AttributionConfig,
) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    top_sources_df = ranked_sources_df.head(config.top_k_sources).copy()
    if top_sources_df.empty:
        return
    plot_stations_and_sources(stations_df, top_sources_df, output_dir)
    plot_top_weights(top_sources_df, output_dir)
    plot_cumulative_share(ranked_sources_df, output_dir)

