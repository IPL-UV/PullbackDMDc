#!/bin/bash
#SBATCH --job-name=pbdmdc_plot_main
#SBATCH --output=plot-main-%j.out
#SBATCH --error=plot-main-%j.err
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
export PBDMDC_YEAR_RANGES=1850-2014

source /opt/ohpc/pub/miniforge3/bin/activate dmdc_variants

cd "$REPO_ROOT"

export PBDMDC_TAYLOR_MODE=raw
python -m evaluation.plot_taylor_vis

export PBDMDC_TAYLOR_MODE=trend
python -m evaluation.plot_taylor_vis

python -m evaluation.plot_gm_timeseries
python -m evaluation.plot_modes
python -m evaluation.eig_vis_circle
python -m evaluation.plot_decay_frequency
python -m evaluation.plot_mode_selection
python -m evaluation.plot_psd_mtm
python -m evaluation.plot_timescale_vs_forced_summary
python -m evaluation.plot_acf_summary
python -m evaluation.plot_mode_summary_three_rows
python -m evaluation.B_vis_small