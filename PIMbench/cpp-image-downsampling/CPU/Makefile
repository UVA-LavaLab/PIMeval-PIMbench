# Makefile: C++ version of image downsampling
# Copyright (c) 2024 University of Virginia
# This file is licensed under the MIT License.
# See the LICENSE file in the root of this repository for more details.

CXX := g++
CXXFLAGS := -std=c++17 -Wall

EXEC := image_downsampling_cpu.out
SRC := image_downsampling_cpu.cpp

.PHONY: debug perf dramsim3_integ clean
.DEFAULT_GOAL := perf

# Setup build targets
ifeq ($(MAKECMDGOALS),)
	MAKECMDGOALS := $(.DEFAULT_GOAL)
endif
ifeq ($(MAKECMDGOALS),debug)
	CXXFLAGS += -g -DDEBUG
endif
ifeq ($(MAKECMDGOALS),perf)
	CXXFLAGS += -O3
endif
ifeq ($(MAKECMDGOALS),dramsim3_integ)
	CXXFLAGS += -O3 -DDRAMSIM3_INTEG
endif

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC)
	$(CXX) $(SRC) $(CXXFLAGS) -o $@

clean:
	rm -f $(EXEC)