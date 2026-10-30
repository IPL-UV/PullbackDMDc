#!/bin/bash
#SBATCH --job-name=sens_history
#SBATCH --output=param-sens-history-%A_%a.out
#SBATCH --error=param-sens-history-%A_%a.err
#SBATCH --time=0-02:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --ntasks=1
#SBATCH --exclude=node01

# Forcing-history sweep (0, 1, 2, 5, 10, 25 and 50 yr before 1850; the main 99 yr rows come from sweep.sbatch), headline
# PullbackDMDc only, one array task per (var, ESM); task = var_index * 4 + esm_index as in sweep.sbatch.
#   sbatch --array=0-11%4 param_sensitivity/history.sbatch
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
  psl) METHOD=PullbackDMDc-2d ;;
  *)   METHOD=PullbackDMDc-3d ;;
esac

set +u  # conda (de)activate hooks reference unset variables
source /opt/ohpc/pub/miniforge3/bin/activate dmdc_variants
set -u
cd "$REPO_ROOT"
python -m param_sensitivity.sweep --var "$VAR" --esm "$ESM" --methods "$METHOD" \
    --histories "${HISTORIES:-0,12,24,60,120,300,600}" --n-jobs "$SLURM_CPUS_PER_TASK" ${MAX_MEMBERS:+--max-members $MAX_MEMBERS} \
    --out "${OUT:-$SENS_ROOT/results/history/sweep_${VAR}_${ESM}.csv}"
