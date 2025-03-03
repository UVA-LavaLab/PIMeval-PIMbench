#!/bin/bash
valgrind --leak-check=full --show-leak-kinds=all --verbose --track-origins=yes --log-file=leak-check.txt ./spmv.out $1 $2 $3 $4 $5 $6 $7 $8 $9 $10
