#!/bin/bash

#SBATCH --gpus=1
#SBATCH -n 1
#SBATCH -t 3-00:00:00
#SBATCH -p gpu
#SBATCH --job-name=logistic-regression
#SBATCH --mem=900000
#SBATCH --output=slurm-out.txt
#SBATCH --cpus-per-task=130
#SBATCH --constraint=a100_80gb
#SBATCH --mail-type=end
#SBATCH --mail-user=yzp7fe@virginia.edu

./lr.out -l 134217728 -v t -c /u/yzp7fe/PIMeval-PIMbench/configs/taco/PIMeval_Bank_Rank1.cfg > bank_rank1.txt
./lr.out -l 134217728 -v t -c /u/yzp7fe/PIMeval-PIMbench/configs/taco/PIMeval_Bank_Rank4.cfg > bank_rank4.txt
./lr.out -l 134217728 -v t -c /u/yzp7fe/PIMeval-PIMbench/configs/taco/PIMeval_Bank_Rank8.cfg > bank_rank8.txt
./lr.out -l 134217728 -v t -c /u/yzp7fe/PIMeval-PIMbench/configs/taco/PIMeval_Fulcrum_Rank1.cfg > Fulcrum_Rank1.txt
./lr.out -l 134217728 -v t -c /u/yzp7fe/PIMeval-PIMbench/configs/taco/PIMeval_Fulcrum_Rank4.cfg > Fulcrum_Rank4.txt
./lr.out -l 134217728 -v t -c /u/yzp7fe/PIMeval-PIMbench/configs/taco/PIMeval_Fulcrum_Rank8.cfg > Fulcrum_Rank8.txt
./lr.out -l 134217728 -v t -c /u/yzp7fe/PIMeval-PIMbench/configs/taco/PIMeval_Fulcrum_Rank16.cfg > Fulcrum_Rank16.txt


