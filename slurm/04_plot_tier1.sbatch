#!/bin/bash
#SBATCH --job-name=pbdmdc_plot_tier1
#SBATCH --output=plot-tier1-%j.out
#SBATCH --error=plot-tier1-%j.err
#SBATCH --time=2-00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --exclude=node01

set -euo pipefail

REPO_ROOT=/home/nathan/PullbackDMDc
export PBDMDC_DATA_ROOT=/data/databases/dmdc-variants/mmlea_v2/
export PBDMDC_ARTIFACT_ROOT=/data/users/nate/PullbackDMDc
export PBDMDC_PDF_ROOT="$REPO_ROOT/evaluation_results"
export PBDMDC_LAGS=3
export PBDMDC_YEAR_RANGES=1950-2014

source /opt/ohpc/pub/miniforge3/bin/activate dmdc_variants

cd "$REPO_ROOT"

export PBDMDC_TAYLOR_MODE=raw
python -m evaluation.plot_taylor_vis

python -m evaluation.plot_gm_timeseries