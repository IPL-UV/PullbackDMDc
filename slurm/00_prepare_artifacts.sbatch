#!/bin/bash
#SBATCH --job-name=pbdmdc_prep
#SBATCH --output=prep-%j.out
#SBATCH --error=prep-%j.err
#SBATCH --time=2-00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --exclude=node01

set -euo pipefail

REPO_ROOT=/home/nathan/PullbackDMDc
export PBDMDC_DATA_ROOT=/data/databases/dmdc-variants/mmlea_v2/
export PBDMDC_ARTIFACT_ROOT=/data/users/nate/PullbackDMDc
export PBDMDC_PDF_ROOT="$REPO_ROOT/evaluation_results"

source /opt/ohpc/pub/miniforge3/bin/activate dmdc_variants

cd "$REPO_ROOT"

python -m data_preparation.create_tas_ocean
python -m data_preparation.compute_means
python -m data_preparation.compute_eofs