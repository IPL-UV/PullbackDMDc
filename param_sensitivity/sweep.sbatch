#!/bin/bash
#SBATCH --job-name=pbdmdc_sens_sweep
#SBATCH --output=sens-sweep-%A_%a.out
#SBATCH --error=sens-sweep-%A_%a.err
#SBATCH --time=0-02:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --ntasks=1
#SBATCH --exclude=node01

set -euo pipefail

REPO_ROOT=/home/nathan/PullbackDMDc
export PBDMDC_DATA_ROOT=/data/databases/dmdc-variants/mmlea_v2/
export PBDMDC_ARTIFACT_ROOT=/data/users/nate/PullbackDMDc

# One task per (var, ESM): task = var_index * 4 + esm_index.
VARS=(tas_ocean psl tas)
ESMS=(CESM2 MPI-ESM1-2-LR CanESM5 MIROC6)
VAR=${VARS[$(( SLURM_ARRAY_TASK_ID / ${#ESMS[@]} ))]}
ESM=${ESMS[$(( SLURM_ARRAY_TASK_ID % ${#ESMS[@]} ))]}
case $VAR in
  psl) TRUNC=20,50,100,200,400 ;;
  *)   TRUNC=2,5,10,20,40,80 ;;
esac

set +u  # conda (de)activate hooks reference unset variables
source /opt/ohpc/pub/miniforge3/bin/activate dmdc_variants
set -u

cd "$REPO_ROOT"
python -m param_sensitivity.compute_sweep --var "$VAR" --esm "$ESM" --truncations "$TRUNC" --lags 1,3,6,12 \
    --histories 0,12,24,60,120,300,600 --n-jobs "$SLURM_CPUS_PER_TASK"
