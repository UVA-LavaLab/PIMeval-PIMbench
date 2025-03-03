#!/bin/bash
valgrind --leak-check=full --show-leak-kinds=all --verbose --track-origins=yes --log-file=leak-check.txt ./spmv.out $@
