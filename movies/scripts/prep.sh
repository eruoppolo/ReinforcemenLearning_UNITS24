#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=01:59:00
#SBATCH --partition THIN
#SBATCH --mem=16G

module load conda
conda activate tabula_env
module load cuda
python ../preprocessing.py
