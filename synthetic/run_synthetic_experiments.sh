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
# uncomment to rerun experiments
#for delta in 0.2 0.5; do
#    python run_ablation_studies.py --set delta=$delta --n-jobs $SLURM_CPUS_PER_TASK
#done

echo "=== run figures ==="
python plot_ablation_run_results.py --deltas 0.2 0.5

echo "=== generator diagnostics ==="
python plot_generator_diagnostics.py
