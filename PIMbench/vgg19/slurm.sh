#!/bin/bash

#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH -p cpu
#SBATCH --job-name=vgg19
#SBATCH --mem=250000
#SBATCH --output=output_vgg19.txt

./PIM/vgg19.out
