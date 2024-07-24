# all:
# 	gcc mlp_openmp.c -o mlp_openmp -fopenmp -std=c99
# run:
# 	./mlp_openmp

# Makefile: C++ version of MLP
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

include ../Makefile.common

EXEC := mlp.out
SRC := mlp.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM
