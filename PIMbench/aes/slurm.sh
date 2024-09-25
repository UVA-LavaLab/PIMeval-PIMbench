#!/bin/bash
#SBATCH --job-name="aes"
#SBATCH --error="aes-bitsimd-enc.err"
#SBATCH --output="aes-bitsimd-enc.out"
#SBATCH -n 1
#SBATCH -t 3-00:00:00
#SBATCH -p cpu
#SBATCH --cpus-per-task=60
#SBATCH --mem=700GB
./PIM/aes.out
