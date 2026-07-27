# Dataset Download Guide

This directory centralizes all dataset acquisition helpers used by PullbackDMDc.

## Output Data Root

All download scripts target:

`/data/databases/dmdc-variants/mmlea_v2/`

Main pipeline scripts read this via `PBDMDC_DATA_ROOT` (default set in `utils/params.py`).

## 1) Download Model Ensembles (GDEX)

Directory: `downloads/models_gdex/`

Run on cluster:

```bash
sbatch downloads/models_gdex/run_download.slurm
```

What it does:
- downloads historical tas/psl files for CanESM5, CESM2, MIROC6, MPI-ESM1-2-LR
- writes into per-variable/model directories under `/data/databases/dmdc-variants/mmlea_v2/`

Useful helpers:
- `downloads/models_gdex/count_files_by_subdir.py` to verify downloaded counts
- `downloads/models_gdex/update_gdex_filelists.py` to sanitize download filelists

## 2) Download 20CRv3 Observation Data

Directory: `downloads/obs_20cr_v3/`

Download raw files:

```bash
bash downloads/obs_20cr_v3/download.sh
```

Regrid and format to repository expectations:

```bash
python downloads/obs_20cr_v3/regrid_downloaded_data.py
```

This script:
- remaps obs products to the model grid
- slices to 1850-01 through 2014-12
- renames variables to match model naming (`tas`, `psl`)

## Validation Checklist

1. Confirm directories exist under data root:
   - `ensembles/tas/<MODEL>/...`
   - `ensembles/psl/<MODEL>/...`
   - `ensembles/tas/20CRv3/...`
   - `ensembles/psl/20CRv3/...`
2. Run:

```bash
python downloads/models_gdex/count_files_by_subdir.py /data/databases/dmdc-variants/mmlea_v2 --depth 2
```

3. After downloads are complete, continue with data preparation:

```bash
python -m data_preparation.interpolate_full_forcing
python -m data_preparation.compute_means
python -m data_preparation.compute_eofs
python -m data_preparation.create_tas_ocean
```