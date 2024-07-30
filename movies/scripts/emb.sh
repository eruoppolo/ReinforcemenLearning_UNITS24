#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:40:00
#SBATCH --partition GPU

module load conda
conda activate tabula_env
module load cuda
python ../embedding.py
