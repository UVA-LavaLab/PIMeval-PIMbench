#!/bin/bash
#SBATCH --job-name="aes_cpu"
#SBATCH --error="aes_cpu.err"
#SBATCH --output="aes_cpu.out"
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --nodelist=cheetah01
#SBATCH --reservation=skadron
#SBATCH --partition=gpu
./build/aes

