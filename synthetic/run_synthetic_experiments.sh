#!/bin/bash
# Regenerate every synthetic ablation run and every figure.
#   results/<run>/{ablations.csv,config.json,provenance.json,slow_modes.npz}
#   figures/ablations/<run>/, figures/generator/<config>/
set -eo pipefail

# conda's deactivate hooks reference unset vars, so -u only goes on after activation
source /opt/ohpc/pub/miniforge3/bin/activate dmdc_variants
set -u
export OMP_NUM_THREADS=1   # the per-step loop is BLAS-thread bound; see README.md
export MPLBACKEND=Agg
cd "$(dirname "$(readlink -f "$0")")"

echo "=== ablation runs ==="
python run_ablation_studies.py --name baseline --n-jobs 12
python run_ablation_studies.py --name dip1 --set depth=1 --n-jobs 12

echo "=== run figures ==="
python plot_ablation_run_results.py --runs baseline
python plot_ablation_run_results.py --runs baseline dip1
python plot_ablation_run_results.py --runs baseline --metric corr
python plot_ablation_run_results.py --runs baseline dip1 --metric corr
python plot_ablation_run_results.py --runs baseline --metric r2
python plot_ablation_run_results.py --runs baseline dip1 --metric r2

echo "=== generator diagnostics ==="
python plot_generator_diagnostics.py
python plot_generator_diagnostics.py --set depth=1
