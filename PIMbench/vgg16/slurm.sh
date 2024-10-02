#!/bin/bash

#SBATCH -n 1
#SBATCH -t 8:00:00
#SBATCH -p cpu
#SBATCH --job-name=vgg16
#SBATCH --mem=250000
#SBATCH --output=output_vgg16.txt

./PIM/vgg16.out