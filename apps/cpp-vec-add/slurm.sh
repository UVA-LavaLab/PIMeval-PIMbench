#!/bin/bash

#SBATCH -n 1
#SBATCH -t 3-00:00:00
#SBATCH -p cpu
#SBATCH --job-name=va_bitserial
#SBATCH --mem=250GB
#SBATCH --output=output_1035544320.txt

./vec-add.out -l 1035544320
