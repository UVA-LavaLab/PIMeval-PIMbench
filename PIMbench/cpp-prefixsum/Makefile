# Makefile: C++ version of prefixsum
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

include ../Makefile.common

EXEC := prefixsum
SRC := prefixsum.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM
