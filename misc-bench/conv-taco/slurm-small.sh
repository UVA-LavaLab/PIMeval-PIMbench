#!/bin/bash

#SBATCH -n 1
#SBATCH -t 3-00:00:00
#SBATCH -p cpu
#SBATCH --job-name=conv-small
#SBATCH --cpus-per-task=40
#SBATCH --mem=250GB
#SBATCH --output=output_conv_small.txt
#SBATCH --mail-type=end
#SBATCH --mail-user=fas9nw@virginia.edu

./conv-batch.out -r 56 -c 56 -d 128 -b 64 -z 256 -o ../../configs/taco/PIMeval_Fulcrum_Rank16.cfg  > ./out/conv3_1/fulcrum_rank_16.txt
./conv-batch.out -r 56 -c 56 -d 128 -b 64 -z 256 -o ../../configs/taco/PIMeval_Bank_Rank16.cfg  > ./out/conv3_1/bank_rank_16.txt
./conv-batch.out -r 56 -c 56 -d 128 -b 64 -z 256 -o ../../configs/taco/PIMeval_BitSerial_Rank16.cfg > ./out/conv3_1/bitserial_rank_16.txt

./conv-batch.out -r 56 -c 56 -d 256 -b 64 -z 256 -o ../../configs/taco/PIMeval_Fulcrum_Rank16.cfg  > ./out/conv3_2/fulcrum_rank_16.txt
./conv-batch.out -r 56 -c 56 -d 256 -b 64 -z 256 -o ../../configs/taco/PIMeval_Bank_Rank16.cfg  > ./out/conv3_2/bank_rank_16.txt
./conv-batch.out -r 56 -c 56 -d 256 -b 64 -z 256 -o ../../configs/taco/PIMeval_BitSerial_Rank16.cfg > ./out/conv3_2/bitserial_rank_16.txt


./conv-batch.out -r 28 -c 28 -d 256 -b 64 -z 512 -o ../../configs/taco/PIMeval_Fulcrum_Rank16.cfg  > ./out/conv4_1/fulcrum_rank_16.txt
./conv-batch.out -r 28 -c 28 -d 256 -b 64 -z 512 -o ../../configs/taco/PIMeval_Bank_Rank16.cfg  > ./out/conv4_1/bank_rank_16.txt
./conv-batch.out -r 28 -c 28 -d 256 -b 64 -z 512 -o ../../configs/taco/PIMeval_BitSerial_Rank16.cfg > ./out/conv4_1/bitserial_rank_16.txt

./conv-batch.out -r 28 -c 28 -d 512 -b 64 -z 512 -o ../../configs/taco/PIMeval_Fulcrum_Rank16.cfg  > ./out/conv4_2/fulcrum_rank_16.txt
./conv-batch.out -r 28 -c 28 -d 512 -b 64 -z 512 -o ../../configs/taco/PIMeval_Bank_Rank16.cfg  > ./out/conv4_2/bank_rank_16.txt
./conv-batch.out -r 28 -c 28 -d 512 -b 64 -z 512 -o ../../configs/taco/PIMeval_BitSerial_Rank16.cfg > ./out/conv4_2/bitserial_rank_16.txt

./conv-batch.out -r 14 -c 14 -d 512 -b 64 -z 512 -o ../../configs/taco/PIMeval_Fulcrum_Rank16.cfg  > ./out/conv5_1/fulcrum_rank_16.txt
./conv-batch.out -r 14 -c 14 -d 512 -b 64 -z 512 -o ../../configs/taco/PIMeval_Bank_Rank16.cfg  > ./out/conv5_1/bank_rank_16.txt
./conv-batch.out -r 14 -c 14 -d 512 -b 64 -z 512 -o ../../configs/taco/PIMeval_BitSerial_Rank16.cfg > ./out/conv5_1/bitserial_rank_16.txt