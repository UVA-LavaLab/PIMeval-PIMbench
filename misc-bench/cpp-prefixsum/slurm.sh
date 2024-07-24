#!/bin/bash
#SBATCH --job-name="Prefix_sum_pim"
#SBATCH --error="Error_sum_pim.err"
#SBATCH --output="Prefix_sum_pim.out"
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=cpu

#./prefix_sum 1048576
./prefixsum 16

