#!/bin/bash

#SBATCH -n 1
#SBATCH -t 3-00:00:00
#SBATCH -p cpu
#SBATCH --job-name=tc_bitserial
#SBATCH --mem=250GB
#SBATCH --output=output_v18772.txt

./PIM/tc.out Dataset/v18772_symetric
