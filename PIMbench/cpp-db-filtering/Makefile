# Makefile: Database filtering
# Copyright (c) 2024 University of Virginia
# This file is licensed under the MIT License.
# See the LICENSE file in the root of this repository for more details.

include ../Makefile.common

EXEC := db-filtering.out
SRC := db-filtering.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS)  -o $@

clean:
	rm -rf $(EXEC) *.dSYM

