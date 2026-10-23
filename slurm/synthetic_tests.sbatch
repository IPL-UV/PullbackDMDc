#!/bin/bash
#SBATCH --job-name=pbdmdc_syn_tests
#SBATCH --output=syn-tests-%j.out
#SBATCH --error=syn-tests-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --exclude=node01

set -euo pipefail

# under sbatch $0 is a spooled copy, so the repo path is fixed here
REPO_ROOT=/home/nathan/PullbackDMDc

# arguments are passed to run_tests.py (e.g. --only forcing)
"$REPO_ROOT/synthetic/run_synthetic_tests.sh" "$@"
