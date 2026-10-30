# Evaluation scripts

One script per figure family, in two stages:

- `compute_*.py` build caches that are expensive or shared by several figures. They write JSON/pkl/npz/csv to
  `$PBDMDC_ARTIFACT_ROOT/evaluation_results/results{year}{run}/`.
- `plot_*.py` read the caches (or the fitted model pickles directly) and write PDFs and summary CSVs to
  `$PBDMDC_PDF_ROOT/results{year}{run}/`.

`slurm/02_compute_metrics.sbatch` runs the compute stage, `slurm/03_plot_main.sbatch` and
`slurm/04_plot_tier1.sbatch` run the plots. Run any script on its own with `python -m evaluation.<name>`.

## Caches

| Script | Writes |
|---|---|
| `compute_taylor` | `td_metrics_{var}.json` |
| `compute_trend_taylor` | `td_metrics_trend_{var}.json`; with `PBDMDC_TREND_WINDOW`, `td_metrics_trend_{window}_{var}.json` |
| `compute_trend_maps` | `trend_maps_{window}_{var}.npz`, `trend_maps_{window}_{var}_metrics.csv` |
| `compute_gm_timeseries` | `gm_timeseries_{var}.pkl`, `region_timeseries_{var}_{region}.pkl` |
| `compute_acfs` | `acfs_{var}_{method}.pkl` |

## Figures

| Script | Reads | Writes |
|---|---|---|
| `plot_taylor` | `compute_taylor` (`PBDMDC_TAYLOR_MODE=raw`) or `compute_trend_taylor` (`=trend`) | `classic_taylor_diagram[_trend[_{window}]]_{var}.pdf` |
| `plot_trend_maps` | `compute_trend_maps` | `trend_maps_{window}_{var}.pdf` |
| `plot_gm_timeseries` | `compute_gm_timeseries` | `gm_{var}.pdf`, `region_{var}_{region}.pdf`, `*_rmsd_by_period_ci.csv` |
| `plot_seasonal_cycle_shift` | models, means | `seasonal_cycle_drift_20CRv3_{tas,psl}.pdf`, `seasonal_cycle_drift_esm_ensmean.pdf`, `seasonal_cycle_drift_20CRv3_summary.csv` |
| `plot_modes` | models, EOFs | `dmd_modes_{dataset}_{var}.pdf`, `forced_variance_ratio_{var}.csv` |
| `plot_eigenvalue_scatter` | models | `eigenvalue_scatter_by_decay_rank_{var}.pdf` |
| `plot_decay_frequency` | models | `decay_frequency_top{n}_{var}.pdf` |
| `plot_decay_lines` | models, `compute_acfs` | `decay_lines_all_modes_ci95_commonN{n}.pdf` |
| `plot_psd_mtm` | models | `psd_mtm_{var}_{method}.pdf` |
| `plot_B_vis_small` | models | `B_vis_small_{var}.pdf` |
| `plot_acf_zero_crossing` | `compute_acfs`, models | `acf_zero_crossing_{vars}_with_rotation.pdf` |
| `plot_acf_efolding` | `compute_acfs`, models | `acf_efolding_{vars}.pdf`, `acf_timescales_summary.csv` |
| `plot_mode_summary` | `compute_acfs`, models | `mode_summary_three_rows_{var}_{method}_sigma{s}.pdf` |
| `plot_forced_fraction_vs_timescale` | `compute_acfs`, models | `forced_fraction_vs_timescale_{var}_{method}_sigma{s}.pdf` |

## Adding a figure

- One new `plot_<output name>.py`, with its work under `if __name__ == '__main__':`.
- Write outputs under `pdf_root` and read caches from `artifact_root`, both from `utils/params.py`. No paths
  relative to the working directory.
- Take labels, dataset colors and timescale bands from `utils/style.py`. Do not copy them into the script.
- Never import another `compute_*` or `plot_*` script. Code that two scripts need goes in `utils/`.
- Add a `compute_*` script only when the result is expensive or shared by several figures, and add both scripts
  to `slurm/`.

`utils/README.md` describes the Taylor-diagram metrics.
