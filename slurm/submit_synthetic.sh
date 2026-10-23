#!/usr/bin/env bash
# Submit the synthetic tests, then the synthetic experiments once the tests pass.
#   ./submit_synthetic.sh               # tests -> experiments
#   ./submit_synthetic.sh --skip-tests  # experiments only
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ "${1:-}" == "--skip-tests" ]]; then
    exp_job=$(sbatch "$SCRIPT_DIR/synthetic_experiments.sbatch" | awk '{print $4}')
    echo "Submitted jobs:"
    echo "  experiments: $exp_job"
    exit 0
fi

tests_job=$(sbatch "$SCRIPT_DIR/synthetic_tests.sbatch" | awk '{print $4}')
exp_job=$(sbatch --dependency=afterok:"$tests_job" "$SCRIPT_DIR/synthetic_experiments.sbatch" | awk '{print $4}')

echo "Submitted jobs:"
echo "  tests:       $tests_job"
echo "  experiments: $exp_job"
