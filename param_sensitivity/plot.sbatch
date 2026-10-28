#!/bin/bash
#SBATCH --job-name=sens_plot
#SBATCH --output=param-sens-plot-%j.out
#SBATCH --error=param-sens-plot-%j.err
#SBATCH --time=0-00:30
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --exclude=node01

# RMSD figures from the sweep CSVs (run after sweep.sbatch).
set -euo pipefail

REPO_ROOT=/home/nathan/PullbackDMDc
export PBDMDC_ARTIFACT_ROOT=/data/users/nate/PullbackDMDc
export PBDMDC_PDF_ROOT="$REPO_ROOT/evaluation_results"

set +u  # conda (de)activate hooks reference unset variables
source /opt/ohpc/pub/miniforge3/bin/activate dmdc_variants
set -u
cd "$REPO_ROOT"
python -m param_sensitivity.plot_rmsd
