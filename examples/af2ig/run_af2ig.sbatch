#!/bin/bash
#SBATCH --job-name=af2ig
#SBATCH --output=af2ig_%j.out
#SBATCH --error=af2ig_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhaoyangli@stanford.edu
#SBATCH --partition=btrippe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00

source "$HOME"/.bashrc
conda activate "$GROUP_HOME"/"$USER"/pdmodels

nvidia-smi

bash ./monomers.sh
