# Makefile: C++ version of transitive closure
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

include ../Makefile.common

EXEC := transitive-closure.out
SRC := transitive-closure.cpp

FOPENMP := -fopenmp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) $(FOPENMP) -o $@

clean:
	rm -rf $(EXEC) *.dSYM
