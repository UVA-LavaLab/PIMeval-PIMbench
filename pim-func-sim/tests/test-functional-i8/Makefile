# Makefile: Test functional behavior
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

include ../Makefile.common

EXEC := test-functional.out
SRC := test-functional.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

