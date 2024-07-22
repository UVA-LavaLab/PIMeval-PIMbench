#!/bin/bash
#SBATCH --job-name="aes_bitserial_pim"
#SBATCH --error="aes_bitserial_pim.err"
#SBATCH --output="aes_bitserial_pim.out"
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --nodelist=cheetah01
#SBATCH --reservation=skadron
#SBATCH --partition=gpu
./aes.out

