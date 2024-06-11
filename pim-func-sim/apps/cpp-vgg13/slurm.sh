#!/bin/bash

#SBATCH -n 1
#SBATCH -t 3-00:00:00
#SBATCH -p gpu
#SBATCH --cpus-per-task=64
#SBATCH --job-name=vgg_f
#SBATCH --mem=750GB
#SBATCH --reservation=skadron
./vgg13.out
../cpp-vgg16/vgg16.out
../cpp-vgg19/vgg19.out
