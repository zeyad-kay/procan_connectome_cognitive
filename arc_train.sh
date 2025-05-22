#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=run.out
#SBATCH --error=run.err

rm -rf ~/software/miniforge3/envs/procan

source ~/software/init-conda
conda env create --file environment.yml
conda activate procan

python ./procan_connectome/main.py

