# Source Attribution (Residual-Spike Inference)

This repo now includes a source attribution pipeline under `src/source_attribution/` that infers practical candidate PM2.5 source zones for the LA monitoring network.

## What "practical sources" means here

With 9 EPA PM2.5 sensors, true physical source identification is underdetermined.  
So we infer practical sources as grid cells that repeatedly align with positive model residual spikes under observed wind direction.

In this workflow:

- Baseline model explains what spatial-time structure can already explain.
- Positive residual spikes (`observed - predicted > 0`) represent unexplained excess.
- A wind-aligned transport kernel maps candidate source cells to affected sensors.
- Sparse non-negative inversion identifies source cells that best explain spike patterns.

## Why residual spikes are used

Residual spikes isolate events the baseline misses, which is where source-like episodic transport should appear most strongly.  
Using the top residual quantile (default `0.90`) focuses attribution on high-impact outliers instead of fitting background noise.

## New module layout

- `src/source_attribution/config.py`: config dataclass.
- `src/source_attribution/geo.py`: lightweight lat/lon and wind-frame math.
- `src/source_attribution/candidates.py`: LA candidate grid generation from station bbox.
- `src/source_attribution/kernel.py`: influence kernel and influence matrix builder.
- `src/source_attribution/inversion.py`: non-negative sparse ElasticNet inversion.
- `src/source_attribution/windows.py`: 6-hour windowing and spike-row selection.
- `src/source_attribution/aggregate.py`: source ranking and cumulative share.
- `src/source_attribution/features.py`: generation of `srcinf_01...srcinf_K` features.
- `src/source_attribution/plots.py`: map, top-weights bar, cumulative share plots.
- `src/source_attribution/run.py`: end-to-end runnable pipeline.

## Artifacts and interpretation

Running the pipeline writes:

- `data/processed/source_attribution/candidates.csv`: full source candidate grid.
- `data/processed/source_attribution/window_results.csv`: per-window inferred coefficients.
- `data/processed/source_attribution/ranked_sources.csv`: ranked aggregated source strengths.
- `data/processed/source_attribution/source_features.csv`: time-aligned source influence features for model merge.

How to read `ranked_sources.csv`:

- `total_weight`: aggregate inferred contribution across windows.
- `active_count`: number of windows with non-zero inferred weight.
- `active_fraction`: activity ratio across windows.
- `cumulative_share`: cumulative mass explained by ranked sources.

Higher `total_weight` and `active_fraction` together indicate more consistent practical source candidates.

## End-to-end commands in this repo

1. Build model-ready PM+weather dataset (existing flow):

```bash
python src/visualization/extended_temp.py
```

2. Run source attribution:

```bash
python -m src.source_attribution.run
```

3. Run model training with optional source feature merge at hookpoint:

```bash
python src/models/new_rfr.py
```

If `data/processed/source_attribution/source_features.csv` exists, `new_rfr.py` merges it at the `# SOURCE_ATTRIBUTION_HOOKPOINT` and prints baseline vs baseline+source metrics (overall MAE and spike MAE), then saves:

- `reports/new_rfr_source_feature_metrics.json`

