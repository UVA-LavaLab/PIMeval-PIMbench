#!/bin/bash
#SBATCH -n 1  # number of tasks (e.g. MPI)
#SBATCH -t 3-00:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=hdc
#SBATCH --mem=128GB
#SBATCH --hint=nomultithread    # don't use hyperthreading

if [[ "$1" == "bs" ]]; then
  ./hdc.out -c ../../../configs/PIMeval_BitSimdV.cfg
elif [[ "$1" == "bl" ]]; then
  ./hdc.out -c ../../../configs/PIMeval_BankLevel.cfg
elif [[ "$1" == "fulcrum" ]]; then
  ./hdc.out -c ../../../configs/PIMeval_Fulcrum.cfg
else
  echo "specify the architecture: \"bs\", \"bl\", \"fulcrum\""
fi
