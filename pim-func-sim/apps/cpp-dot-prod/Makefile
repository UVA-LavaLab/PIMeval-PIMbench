# Makefile: C++ version of vector addition
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

include ../Makefile.common

EXEC := dot
SRC := dot-prod.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

