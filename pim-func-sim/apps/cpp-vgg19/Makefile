# Makefile: C++ version of vgg
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

include ../Makefile.common

EXEC := vgg19.out
SRC := vgg.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -fopenmp -o $@

clean:
	rm -rf $(EXEC) *.dSYM

