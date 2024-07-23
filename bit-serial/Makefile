# Makefile: Bit-serial Performance Modeling
# Copyright (c) 2024 University of Virginia
# This file is licensed under the MIT License.
# See the LICENSE file in the root of this repository for more details.

include ./Makefile.common

EXEC := bitSerial.out
SRCS = $(wildcard *.cpp)
HEADERS = $(wildcard *.h)

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRCS) $(HEADERS) $(DEPS)
	$(CXX) $(SRCS) $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

