#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

prep_job=$(sbatch "$SCRIPT_DIR/00_prepare_artifacts.sbatch" | awk '{print $4}')
fit_job=$(sbatch --dependency=afterok:"$prep_job" "$SCRIPT_DIR/01_fit_models.sbatch" | awk '{print $4}')
metrics_job=$(sbatch --dependency=afterok:"$fit_job" "$SCRIPT_DIR/02_compute_metrics.sbatch" | awk '{print $4}')
main_plot_job=$(sbatch --dependency=afterok:"$metrics_job" "$SCRIPT_DIR/03_plot_main.sbatch" | awk '{print $4}')
tier1_plot_job=$(sbatch --dependency=afterok:"$metrics_job" "$SCRIPT_DIR/04_plot_tier1.sbatch" | awk '{print $4}')

echo "Submitted jobs:"
echo "  prep:        $prep_job"
echo "  fit:         $fit_job"
echo "  metrics:     $metrics_job"
echo "  plot_main:   $main_plot_job"
echo "  plot_tier1:  $tier1_plot_job"
