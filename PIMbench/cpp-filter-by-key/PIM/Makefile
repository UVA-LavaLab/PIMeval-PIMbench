# Makefile: C++ version of filter by key 
# Copyright (c) 2024 University of Virginia
# This file is licensed under the MIT License.
# See the LICENSE file in the root of this repository for more details.

LIBPIMEVAL_PATH = ../../../libpimeval
include ../../Makefile.common

EXEC := filter.out
SRCS := $(wildcard *.cpp) 

# Automatically adds OpenMP in order to avoid errors
CXXFLAGS += -fopenmp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRCS) $(DEPS)
	$(CXX) $(SRCS) $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM
