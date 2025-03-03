#!/bin/bash

#SBATCH -n 1
#SBATCH -t 3-00:00:00
#SBATCH -p cpu
#SBATCH --job-name=spmv-packed-bitserial
#SBATCH --mem=250GB
#SBATCH --output=output_26843545.txt
#SBATCH --exclusive

lscpu
./spmv.out -x 2048 -y 2048 -n 26843545 -b 0 -v t
echo "2048 x 2048 - 26843545 nnz"

echo "--------------------------------"
./spmv.out -x 4096 -y 4096 -n 26843545 -b 0 -v t
echo "4096 x 4096 - 26843545 nnz"

echo "--------------------------------"
./spmv.out -x 2048 -y 2048 -n 13421773 -b 0 -v t
echo "2048 x 2048 - 13421773 nnz"

echo "--------------------------------"
./spmv.out -x 2048 -y 2048 -n 2684355 -b 0 -v t
echo "2048 x 2048 - 2684355 nnz"

echo "--------------------------------"
./spmv.out -x 2048 -y 2048 -n 268435 -b 0 -v t
echo "2048 x 2048 - 268435 nnz"

echo "--------------------------------"
./spmv.out -x 2048 -y 2048 -n 26843 -b 0 -v t
echo "2048 x 2048 - 26843 nnz"

echo "--------------------------------"
./spmv.out -x 4096 -y 4096 -n 13421773 -b 0 -v t
echo "4096 x 4096 - 13421773 nnz"
