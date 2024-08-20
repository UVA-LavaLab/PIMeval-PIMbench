#!/bin/bash

#SBATCH -n 1
#SBATCH -t 3-00:00:00
#SBATCH -p cpu
#SBATCH --job-name=lr_fulcrum
#SBATCH --mem=240GB
#SBATCH --output=output_fulcrum.txt

./lr.out -l 2500000000 -c ../../configs/iiswc/PIMeval_Fulcrum_Rank1.cfg > ./out/fulcrum_rank_1.txt
./lr.out -l 2500000000 -c ../../configs/iiswc/PIMeval_Fulcrum_Rank4.cfg > ./out/fulcrum_rank_4.txt
./lr.out -l 2500000000 -c ../../configs/iiswc/PIMeval_Fulcrum_Rank8.cfg > ./out/fulcrum_rank_8.txt
./lr.out -l 2500000000 -c ../../configs/iiswc/PIMeval_Fulcrum_Rank16.cfg > ./out/fulcrum_rank_16.txt
./lr.out -l 2500000000 -c ../../configs/iiswc/PIMeval_Fulcrum_Rank32.cfg > ./out/fulcrum_rank_32.txt

