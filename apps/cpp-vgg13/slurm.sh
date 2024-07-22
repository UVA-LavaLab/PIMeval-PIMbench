#!/bin/bash

#SBATCH -n 1
#SBATCH -t 3-00:00:00
#SBATCH -p gpu
#SBATCH --cpus-per-task=60
#SBATCH --job-name=vgg_fulcrum
#SBATCH --mem=700GB
#SBATCH --reservation=fas9nw_98

./vgg13.out
../cpp-vgg16/vgg16.out
../cpp-vgg19/vgg19.out
