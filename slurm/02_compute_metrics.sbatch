#!/bin/bash
#SBATCH --job-name=pbdmdc_metrics
#SBATCH --output=metrics-%j.out
#SBATCH --error=metrics-%j.err
#SBATCH --time=5-00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --exclude=node01

set -euo pipefail

REPO_ROOT=/home/nathan/DMDcForcedResponse
export PBDMDC_DATA_ROOT=/data/databases/dmdc-variants/mmlea_v2/
export PBDMDC_ARTIFACT_ROOT=/data/users/nate/PullbackDMDc
export PBDMDC_PDF_ROOT="$REPO_ROOT/evaluation_results"
export PBDMDC_LAGS=3

source /opt/ohpc/pub/miniforge3/bin/activate dmdc_variants

cd "$REPO_ROOT"

export PBDMDC_YEAR_RANGES=1850-2014,1950-2014
python -m evaluation.compute_taylor_vis
python -m evaluation.compute_gm_timeseries

export PBDMDC_YEAR_RANGES=1850-2014
python -m evaluation.compute_trend_taylor_viz
python -m evaluation.compute_acfs