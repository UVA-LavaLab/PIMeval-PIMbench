# Makefile for PIMeval Simulator - Tests -Functional
# Copyright (c) 2024 University of Virginia
# This file is licensed under the MIT License.
# See the LICENSE file in the root of this repository for more details.

PROJ_ROOT = ../..
include ${PROJ_ROOT}/Makefile.common

EXEC := test-functional.out
SRCS := $(wildcard *.cpp) 
HEADERS := $(wildcard *.h) 

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRCS) $(DEPS) $(HEADERS)
	$(CXX) $(SRCS) $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

