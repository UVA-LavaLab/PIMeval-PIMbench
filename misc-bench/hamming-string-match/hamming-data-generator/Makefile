# Makefile: C++ data generator for hamming string match
# Copyright (c) 2025 University of Virginia
# This file is licensed under the MIT License.
# See the LICENSE file in the root of this repository for more details.

# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -Wall -Wextra -Werror -std=c++17 -O3 -Wno-unknown-pragmas -I./../../../util

# Executable name
EXEC := hamming-data-generator.out

# Source files
SRC_FILES := hamming-data-generator.cpp

# make USE_OPENMP=1
USE_OPENMP ?= 0
ifeq ($(USE_OPENMP),1)
	CXXFLAGS += -fopenmp
endif

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(SRC_FILES) |
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -rf $(EXEC)
