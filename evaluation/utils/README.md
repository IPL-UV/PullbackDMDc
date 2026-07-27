# Taylor Diagram Workflow

This folder contains the metric and plotting utilities used by the Taylor diagram pipeline.

## Key Files

- `taylor_diagram.py`
  - `TaylorDiagramEstimator`: direct Taylor metrics on full fields
  - `Ensemble_PCA_TD_Estimator`: Taylor metrics for ensemble predictions represented in EOF/PC space
- `taylor_diagram_plot.py`: plotting support
- `metrics.py`: helper utilities used by evaluation scripts

## Why EOF-Space Taylor Metrics Are Faster

The evaluation pipeline computes predictions in reduced EOF coordinates (PC space), then evaluates Taylor statistics there with `Ensemble_PCA_TD_Estimator`.

Raw-field Taylor metrics require operations over all spatial grid points. EOF-space metrics reduce this to a much smaller latent dimension:

$$
\text{cost}_{\text{raw}} \sim O(T \cdot N_{\text{grid}}),\quad
\text{cost}_{\text{EOF}} \sim O(T \cdot N_{\text{EOF}}),\quad
N_{\text{EOF}} \ll N_{\text{grid}}.
$$

Because `N_EOF` is typically much smaller than the number of grid cells, this substantially reduces memory and compute while preserving the dominant variability captured by EOF truncation.

## End-to-End Commands

Compute raw Taylor metrics:

```bash
python -m evaluation.compute_taylor_vis
```

Compute trend Taylor metrics:

```bash
python -m evaluation.compute_trend_taylor_viz
```

Plot Taylor diagrams (set mode with env var):

```bash
PBDMDC_TAYLOR_MODE=raw python -m evaluation.plot_taylor_vis
PBDMDC_TAYLOR_MODE=trend python -m evaluation.plot_taylor_vis
```

## Inputs and Outputs

- Inputs:
  - truth fields from `load_truth(...)`
  - model predictions from `load_truncated_prediction(...)`
- Outputs:
  - JSON metrics files in `evaluation_results/results*/td_metrics*.json`
  - Taylor diagram figures in artifact/pdf output directories

See `utils/params.py` for data and artifact root configuration.