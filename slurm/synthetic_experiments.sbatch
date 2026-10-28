#!/bin/bash
#SBATCH --job-name=pbdmdc_syn_exp
#SBATCH --output=syn-exp-%j.out
#SBATCH --error=syn-exp-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=100
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --exclude=node01

set -euo pipefail

# under sbatch $0 is a spooled copy, so the repo path is fixed here
REPO_ROOT=.

"$REPO_ROOT/synthetic/run_synthetic_experiments.sh"
