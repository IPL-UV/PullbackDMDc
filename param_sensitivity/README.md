# Parameter sensitivity (truncation, lag, tas_ocean domain)

The parameters are varied one at a time, with the other two held at their main values:

| Parameter | Main value | Values tried |
|---|---|---|
| Truncation N | 20 for SAT/OSAT, 200 for SLP | {10, 20, 40, 80} for SAT/OSAT, {100, 200, 400} for SLP |
| Lag | 3 months | {1, 3, 6, 12} months |
| tas_ocean latitude band | 40S–65N | 65S–65N, 90S–90N |

For each setting, every method is refitted on the shared EOF basis: the headline PullbackDMDc (-3d for SAT and OSAT, -2d for SLP), LIM, LIM-opt, LR and RegGMST. The fit is then scored against the ensemble mean, for all 1850–2014 members of CESM2, MPI-ESM1-2-LR, CanESM5 and MIROC6.

The score is the full-period RMSD of the global-mean index, in % of the truth's standard deviation. For SLP the index is the polar-cap mean S of 65°S and N of 65°N, and the truth is smoothed over 3 years. RMSD is computed per member, as in `evaluation/plot_gm_timeseries.py`. Lines show the member median and bands the IQR.

A member is dropped from a sweep if it has a hard failure for any method at any setting of that sweep:
- `unstable`: leading |λ| ≥ 1;
- `nonfinite`;
- `fit_error`.

This way every line in a sweep uses the same members, and the plot script prints how many are kept.

## Outputs
Results go under `SENS_ROOT`, which is `$PBDMDC_SENS_ROOT`, or `$PBDMDC_ARTIFACT_ROOT/param_sensitivity` if that is unset:
- `psl_n400/eofs/`: the 400-EOF psl store.
- `results/main/sweep_{var}_{ESM}.csv`: one row per (member, method, N, lag).
- `results/domain/`: the domain sweep.

Figures go to `evaluation_results/param_sensitivity/`:
- **`pullback_rmsd.{pdf,png}`:** PullbackDMDc only. The columns are N (relative to the main value), lag and domain, with one line per variable or index.
- **`methods_rmsd.{pdf,png}`:** all methods. The rows are SAT, OSAT, SLP S of 65°S and SLP N of 65°N; the columns are N, lag and domain. The domain column has only the OSAT row.
- **`rmsd_quantiles.csv`:** the quartiles behind both figures.

## Running (slurm, from the repo root)
```bash
sbatch param_sensitivity/psl_eofs_n400.sbatch            # ~1 h, ~8 GB
sbatch --array=0-11%4 param_sensitivity/sweep.sbatch     # 3 vars x 4 ESMs, ~2-3 h at 4 tasks at once
sbatch param_sensitivity/plot.sbatch                      # < 1 min
```

## Domain sweep (optional, ~75 GB and ~1.5 h per band)
The domain column is drawn only if `results/domain/*.csv` exist. To make them for one band (65S–65N shown here; the run uses about 12 cores):

1. Rebuild tas_ocean on the wider band in its own root.
2. Recompute the ensemble means and EOFs there.
3. Score at the main config.

```bash
DOM=$SENS_ROOT/domain_65S65N
PBDMDC_DATA_ROOT=/data/databases/dmdc-variants/mmlea_v2/ PBDMDC_TAS_OCEAN_OUT=$DOM \
  PBDMDC_LAT_MIN=-65 PBDMDC_LAT_MAX=65 PBDMDC_PROOF_PLOT=$DOM/proof.png python -m data_preparation.create_tas_ocean
export PBDMDC_CLIM_VARS=tas_ocean PBDMDC_YEAR_RANGES=1850-2014
PBDMDC_DATA_ROOT=$DOM PBDMDC_MEANS_OUT=$DOM python -m data_preparation.compute_means
PBDMDC_DATA_ROOT=$DOM PBDMDC_EOF_OUT=$DOM python -m data_preparation.compute_eofs
for ESM in CESM2 MPI-ESM1-2-LR CanESM5 MIROC6; do
  python -m param_sensitivity.sweep --var tas_ocean --esm $ESM --truncations 20 --lags 3 \
    --eof-root $DOM/eofs --means-root $DOM/means --domain 65S65N \
    --out $SENS_ROOT/results/domain/sweep_tas_ocean_${ESM}_65S65N.csv
done
```

For 90S–90N, use `LAT_MIN=-90 LAT_MAX=90` and `--domain 90S90N`.
