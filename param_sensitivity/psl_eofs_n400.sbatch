#!/bin/bash
#SBATCH --job-name=sens_eofs400
#SBATCH --output=param-sens-eofs400-%j.out
#SBATCH --error=param-sens-eofs400-%j.err
#SBATCH --time=0-06:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --exclude=node01

# psl EOF store with 400 EOFs (1850-2014) for the N=400 sweep configs; the main eofs/ store is untouched.
set -euo pipefail

REPO_ROOT=/home/nathan/PullbackDMDc
export PBDMDC_DATA_ROOT=/data/databases/dmdc-variants/mmlea_v2/
export PBDMDC_ARTIFACT_ROOT=/data/users/nate/PullbackDMDc
SENS_ROOT=${PBDMDC_SENS_ROOT:-$PBDMDC_ARTIFACT_ROOT/param_sensitivity}
export PBDMDC_EOF_OUT="$SENS_ROOT/psl_n400"
export PBDMDC_EOF_TRUNCATION=400
export PBDMDC_CLIM_VARS=psl
export PBDMDC_YEAR_RANGES=1850-2014

set +u  # conda (de)activate hooks reference unset variables
source /opt/ohpc/pub/miniforge3/bin/activate dmdc_variants
set -u

cd "$REPO_ROOT"
python -m data_preparation.compute_eofs
