#!/bin/bash

#SBATCH --array=1-80
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --time=5:00:00

#rm -rf ~/software/miniforge3/envs/procan

source ~/software/init-conda
source .env
conda env create --file environment.yml
conda activate procan

wandb agent --count 1 ${WANDB_ENTITY}/${WANDB_PROJECT}/${1}