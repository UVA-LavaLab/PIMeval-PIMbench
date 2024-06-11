# Makefile: C++ version of vector addition
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

include ../Makefile.common

EXEC := knn
SRC := knn.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -fopenmp -o $@

clean:
	rm -rf $(EXEC) *.dSYM

