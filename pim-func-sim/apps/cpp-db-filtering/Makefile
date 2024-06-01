# Makefile: Database filtering
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

include ../Makefile.common

EXEC := db-filtering.out
SRC := db-filtering.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS)  -o $@

clean:
	rm -rf $(EXEC) *.dSYM

