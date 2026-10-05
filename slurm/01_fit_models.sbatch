#!/bin/bash
#SBATCH --job-name=pbdmdc_fit
#SBATCH --output=fit-%j.out
#SBATCH --error=fit-%j.err
#SBATCH --time=5-00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --exclude=node01

set -euo pipefail

REPO_ROOT=/home/nathan/PullbackDMDc
export PBDMDC_DATA_ROOT=/data/databases/dmdc-variants/mmlea_v2/
export PBDMDC_ARTIFACT_ROOT=/data/users/nate/PullbackDMDc
export PBDMDC_PDF_ROOT="$REPO_ROOT/evaluation_results"
export PBDMDC_LAGS=3
export PBDMDC_YEAR_RANGES=1850-2014,1950-2014

source /opt/ohpc/pub/miniforge3/bin/activate dmdc_variants

cd "$REPO_ROOT"
python experiments.py