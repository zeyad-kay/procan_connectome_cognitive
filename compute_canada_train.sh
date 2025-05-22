#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:30:00
#SBATCH --output=run.out
#SBATCH --error=run.err

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

python -m pip install --upgrade pip

python -m pip install -r compute_canada_requirements.txt --no-index
python -m pip install -e .

python ./procan_connectome/main.py
