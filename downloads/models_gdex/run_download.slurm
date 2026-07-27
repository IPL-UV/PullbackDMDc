#!/bin/bash
#SBATCH --job-name=mmlea_v2
#SBATCH --output=out.log
#SBATCH --error=error.log          
#SBATCH --time=15-00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --exclude=node02,node03,node04,node09


# source activate /home/nathan/miniconda3/envs/dmdc_variants
source /opt/ohpc/pub/miniforge3/bin/activate dmdc_variants

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd /data/databases/dmdc-variants/mmlea_v2/psl/CanESM5
python "${SCRIPT_DIR}/gdex-download2.py"

cd /data/databases/dmdc-variants/mmlea_v2/psl/CESM2
python "${SCRIPT_DIR}/gdex-download1.py"

cd /data/databases/dmdc-variants/mmlea_v2/psl/MIROC6
python "${SCRIPT_DIR}/gdex-download3.py"

cd /data/databases/dmdc-variants/mmlea_v2/psl/MPI-ESM1-2-LR
python "${SCRIPT_DIR}/gdex-download4.py"

cd /data/databases/dmdc-variants/mmlea_v2/tas/CanESM5
python "${SCRIPT_DIR}/gdex-download6.py"

cd /data/databases/dmdc-variants/mmlea_v2/tas/CESM2
python "${SCRIPT_DIR}/gdex-download5.py"

cd /data/databases/dmdc-variants/mmlea_v2/tas/MIROC6
python "${SCRIPT_DIR}/gdex-download7.py"

cd /data/databases/dmdc-variants/mmlea_v2/tas/MPI-ESM1-2-LR
python "${SCRIPT_DIR}/gdex-download8.py"
