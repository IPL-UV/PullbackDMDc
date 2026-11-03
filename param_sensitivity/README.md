# Parameter sensitivity

`pullback_rmsd.pdf` shows how the headline PullbackDMDc (-3d for SAT and OSAT, -2d for SLP) responds to one parameter at a time, with the others held at their main values:

| Parameter | Main | Tried |
|---|---|---|
| Truncation N | 20 (SAT, OSAT), 200 (SLP) | ×0.1, ×0.25, ×0.5, ×1, ×2, ×4 |
| Lag | 3 months | 1, 3, 6, 12 |
| tas_ocean band | 40S–65N | 65S–65N, 90S–90N |
| Forcing history before 1850 | 99 yr | 0, 1, 2, 5, 10, 25, 50, 99 yr |

The score is the distance to (1, 0) in the raw-field Taylor diagram, as in `evaluation/compute_taylor.py`. That is the centered field RMSD against the ensemble-mean field over 1850–2014, in % of the truth std. The SLP truth is smoothed over 3 yr. In the domain panel, each band is scored against its own truth.

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

## Domain panel (~75 GB and ~1.5 h per band)
For each band, rebuild tas_ocean, its means and its EOFs, then score the main config. For 65S–65N:
```bash
DOM=$SENS_ROOT/domain_65S65N
PBDMDC_TAS_OCEAN_OUT=$DOM PBDMDC_LAT_MIN=-65 PBDMDC_LAT_MAX=65 python -m data_preparation.create_tas_ocean
export PBDMDC_CLIM_VARS=tas_ocean PBDMDC_YEAR_RANGES=1850-2014
PBDMDC_DATA_ROOT=$DOM PBDMDC_MEANS_OUT=$DOM python -m data_preparation.compute_means
PBDMDC_DATA_ROOT=$DOM PBDMDC_EOF_OUT=$DOM python -m data_preparation.compute_eofs
for ESM in CESM2 MPI-ESM1-2-LR CanESM5 MIROC6; do
  python -m param_sensitivity.compute_sweep --var tas_ocean --esm $ESM --eof-root $DOM/eofs --means-root $DOM/means \
    --domain 65S65N --out $SENS_ROOT/results/domain/sweep_tas_ocean_${ESM}_65S65N.csv
done
```
For 90S–90N, use `PBDMDC_LAT_MIN=-90 PBDMDC_LAT_MAX=90` and `--domain 90S90N`.
