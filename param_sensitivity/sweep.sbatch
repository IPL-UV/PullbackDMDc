#!/bin/bash
#SBATCH --job-name=sens_sweep
#SBATCH --output=param-sens-sweep-%A_%a.out
#SBATCH --error=param-sens-sweep-%A_%a.err
#SBATCH --time=0-12:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --ntasks=1
#SBATCH --exclude=node01

# Truncation + lag sweep, one array task per (var, ESM); task = var_index * 4 + esm_index with
# vars (tas_ocean psl tas) and ESMs (CESM2 MPI-ESM1-2-LR CanESM5 MIROC6). Needs psl_eofs_n400.sbatch first.
#   sbatch --array=0-11%4 param_sensitivity/sweep.sbatch
set -euo pipefail

REPO_ROOT=/home/nathan/PullbackDMDc
export PBDMDC_DATA_ROOT=/data/databases/dmdc-variants/mmlea_v2/
export PBDMDC_ARTIFACT_ROOT=/data/users/nate/PullbackDMDc
SENS_ROOT=${PBDMDC_SENS_ROOT:-$PBDMDC_ARTIFACT_ROOT/param_sensitivity}

VARS=(tas_ocean psl tas)
ESMS=(CESM2 MPI-ESM1-2-LR CanESM5 MIROC6)
VAR=${VARS[$(( SLURM_ARRAY_TASK_ID / ${#ESMS[@]} ))]}
ESM=${ESMS[$(( SLURM_ARRAY_TASK_ID % ${#ESMS[@]} ))]}
case $VAR in
  psl) TRUNC=100,200,400 ;;
  *)   TRUNC=10,20,40,80 ;;
esac

set +u  # conda (de)activate hooks reference unset variables
source /opt/ohpc/pub/miniforge3/bin/activate dmdc_variants
set -u
cd "$REPO_ROOT"
python -m param_sensitivity.sweep --var "$VAR" --esm "$ESM" \
    --truncations "$TRUNC" --lags 1,3,6,12 --n-jobs "$SLURM_CPUS_PER_TASK" \
    --eof-root-large "$SENS_ROOT/psl_n400/eofs" \
    --out "$SENS_ROOT/results/main/sweep_${VAR}_${ESM}.csv"
