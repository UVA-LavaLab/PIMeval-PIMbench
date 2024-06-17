# Makefile: BitSIMD-V basic tests
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

include ../Makefile.common

EXEC := bs.out
SRCS = $(wildcard *.cpp)
HEADERS = $(wildcard *.h)

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRCS) $(HEADERS) $(DEPS)
	$(CXX) $(SRCS) $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

