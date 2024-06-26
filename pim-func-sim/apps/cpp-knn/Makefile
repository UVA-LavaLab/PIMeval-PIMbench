# Makefile: C++ version of K-Nearest Neighbor
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

include ../Makefile.common

EXEC := knn.out
SRC := knn.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -fopenmp -o $@

clean:
	rm -rf $(EXEC) *.dSYM

