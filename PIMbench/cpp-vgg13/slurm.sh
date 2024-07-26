#!/bin/bash

#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH -p cpu
#SBATCH --job-name=vgg13
#SBATCH --mem=250000
#SBATCH --output=output_vgg13.txt

./PIM/vgg13.out