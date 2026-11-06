# Parameter sensitivity

`pullback_rmsd.pdf` shows how the headline PullbackDMDc (-3d for SAT and OSAT, -2d for SLP) responds to one parameter at a time, with the others held at their main values:

| Parameter | Main | Tried |
|---|---|---|
| Truncation N | 20 (SAT, OSAT), 200 (SLP) | ×0.1, ×0.25, ×0.5, ×1, ×2, ×4 |
| Lag | 3 months | 1, 3, 6, 12 |
| Forcing history before 1850 | 99 yr | 0, 1, 2, 5, 10, 25, 50, 99 yr |

The score is the distance to (1, 0) in the raw-field Taylor diagram, as in `evaluation/compute_taylor.py`. That is the centered field RMSD against the ensemble-mean field over 1850–2014, in % of the truth std. The SLP truth is smoothed over 3 yr.

Each member's score is taken minus its own score at the main config. The figure shows the median and IQR, with the four ESMs weighted equally.

A member is dropped from a whole sweep if, at any setting of that sweep, its leading |λ| ≥ 1, its fit is not finite, or the fit fails.

Results are written to `$PBDMDC_SENS_ROOT`, which defaults to `$PBDMDC_ARTIFACT_ROOT/param_sensitivity`. The figure goes to `$PBDMDC_PDF_ROOT/param_sensitivity/`.

## Running (slurm, from the repo root)
1. Build the 400-EOF psl store, needed for SLP N = 400 (~1 h, 200 GB):
   ```bash
   PBDMDC_EOF_OUT=$SENS_ROOT/psl_n400 PBDMDC_EOF_TRUNCATION=400 PBDMDC_CLIM_VARS=psl PBDMDC_YEAR_RANGES=1850-2014 \
     python -m data_preparation.compute_eofs
   ```
2. Run the sweeps (< 1 min per task):
   ```bash
   sbatch --array=0-11%4 param_sensitivity/sweep.sbatch
   ```
3. Draw the figure:
   ```bash
   python -m param_sensitivity.plot_pullback_rmsd
   ```
