#!/bin/bash
#SBATCH -n 1  # number of tasks (e.g. MPI)
#SBATCH -t 3-00:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=hdc
#SBATCH --mem=128GB
#SBATCH --hint=nomultithread    # don't use hyperthreading

module purge
module load apptainer

# Check if "cuda" is passed as a parameter
if [[ "$1" == "gpu" ]]; then
    apptainer exec --nv ../apptainer.sif python3 hd_oms.py --cuda
else
    apptainer exec ../apptainer.sif python3 hd_oms.py
fi
