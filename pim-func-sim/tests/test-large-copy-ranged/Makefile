# Makefile: Test large data copy within a range
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

include ../Makefile.common

EXEC := test-large-copy-ranged.out
SRC := test-large-copy-ranged.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

