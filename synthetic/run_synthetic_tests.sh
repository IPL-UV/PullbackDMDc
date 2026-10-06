#!/bin/bash
# Synthetic test suite: the unit tests, then the end-to-end system checks.
# Arguments are passed to run_tests.py (e.g. --only forcing, --list).
set -eo pipefail

# conda's deactivate hooks reference unset vars, so -u only goes on after activation
source /opt/ohpc/pub/miniforge3/bin/activate dmdc_variants
set -u
export OMP_NUM_THREADS=1   # the per-step loop is BLAS-thread bound; see DATA_GENERATION.md
export MPLBACKEND=Agg      # the plot tests render headlessly
cd "$(dirname "$(readlink -f "$0")")"

python run_tests.py "$@"

echo
echo "=== tests/test_synthetic_pipeline.py -> figures/tests/ ==="
python tests/test_synthetic_pipeline.py
