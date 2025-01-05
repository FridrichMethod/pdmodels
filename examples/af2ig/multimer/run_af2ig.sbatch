#!/bin/bash
#SBATCH -J af2ig
#SBATCH -o af2ig_%j.out
#SBATCH -e af2ig_%j.err
#SBATCH -p gpu_4l
#SBATCH -A lhlai_g1
#SBATCH -q lhlaig4c
#SBATCH -N 1
#SBATCH -c 7
#SBATCH --gres=gpu:1
#SBATCH --no-requeue

source ./envs.sh

nvidia-smi

sh ./multimers.sh &

sleep 180s
nvidia-smi
sleep 120s
nvidia-smi

wait
