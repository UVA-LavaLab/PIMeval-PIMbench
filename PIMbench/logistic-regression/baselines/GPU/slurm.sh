#!/bin/bash

#SBATCH --job-name=gpu_lr           # Job name
#SBATCH --output=slurm-out.txt                     # Output log
#SBATCH --mail-type=END                            # Email notification when job ends
#SBATCH --mail-user=yzp7fe@virginia.edu            # Your UVA email

#SBATCH --partition=gpu                            # GPU partition
#SBATCH --gpus=2                                    # Request 2 GPUs
#SBATCH --constraint=a100_80gb                     # Specifically A100 80GB
#SBATCH --cpus-per-task=64                         # Number of CPU cores
#SBATCH --mem=512G                                 # Request 512 GB RAM
#SBATCH --time=3-00:00:00                          # 3-day walltime
#SBATCH -n 1                                        # One task (multi-threaded)

# Load necessary modules
module purge
module load gcc/12.1.0
module load cuda/12.2.0

# Run your GPU-aware executable (must support multiple GPUs)
./lr.out -l 134217728 > out_gpu.txt