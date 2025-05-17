#!/bin/bash
#SBATCH -n 1
#SBATCH -t 1-00:00:00 
#SBATCH -p cpu 
#SBATCH --job-name=lr_BitSerial
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=12
#SBATCH --output=/u/bg9qq/beenishPIM/PIMeval-PIMbench/misc-bench/cpp-prefix-sum/out/out.txt
#SBATCH --error=/u/bg9qq/beenishPIM/PIMeval-PIMbench/misc-bench/cpp-prefix-sum/out/error.txt


./prefix-sum.out -l 8192 -c ../../configs/taco/PIMeval_BitSerial_Rank16.cfg > ./out/BitSerial_Rank_16.txt
#./prefix-sum.out -l 8192 -c ../../configs/taco/PIMeval_Fulcrum_Rank16.cfg > ./out/PIMeval_Fulcrum_Rank16.txt


#./prefix-sum.out -l 10 -c ../../configs/taco/PIMeval_Bank_Rank16.cfg > ./out/PIMeval_Bank_Rank16.txt