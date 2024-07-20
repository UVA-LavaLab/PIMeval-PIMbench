# Makefile for Triangle Counting Benchmark
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

include ../Makefile.common

CXXFLAGS += -O3 
# CXXFLAGS += -DENABLE_PARALLEL -fopenmp  
EXEC := tc.out
SRC := tc.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

