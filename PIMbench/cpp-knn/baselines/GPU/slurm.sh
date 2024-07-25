#!/bin/bash

#SBATCH -n 1
#SBATCH -t 1-00:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=knn_gpu_test
#SBATCH --mem=50GB
#SBATCH --output=output_slurm.txt

./knn.out