#!/bin/bash
# Regenerate every synthetic ablation run and every figure.
#   results/<run>/{ablations.csv,config.json,provenance.json}
#   figures/ablations/<run>/, figures/diagnostics/{data,system}/
# results/b_unscaled/ is deliberately not regenerated: it is the pre-B-scaling
# calibration, kept for comparison, and current code cannot reproduce it.
set -eo pipefail

# conda's deactivate hooks reference unset vars, so -u only goes on after activation
source /opt/ohpc/pub/miniforge3/bin/activate dmdc_variants
set -u
export OMP_NUM_THREADS=1   # the per-step loop is BLAS-thread bound; see DATA_GENERATION.md
export MPLBACKEND=Agg
cd "$(dirname "$(readlink -f "$0")")"

echo "=== ablation runs ==="
python run_ablation_studies.py --name baseline
python run_ablation_studies.py --name dip1 --set gauss_dip_amp=1

echo "=== run figures ==="
python plot_ablation_run_results.py --runs baseline
python plot_ablation_run_results.py --runs baseline dip1

echo "=== diagnostics ==="
python plot_ablation_diagnostics.py
python plot_system_diagnostics.py
python plot_system_diagnostics.py --study slow_timescale_snr --level 100
python plot_forcing_comparison.py
