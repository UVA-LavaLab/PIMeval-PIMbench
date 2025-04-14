# Makefile: C++ data generator for string match
# Copyright (c) 2024 University of Virginia
# This file is licensed under the MIT License.
# See the LICENSE file in the root of this repository for more details.

# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -Wall -Wextra -Werror -std=c++17 -O3 -Wno-unknown-pragmas -I./../../../util

# Executable name
EXEC := data-generator.out

# Source files
SRC_FILES := data-generator.cpp

# Dependencies
HEADER_DEPS := ./../../../util/utilStringMatch.h

# make USE_OPENMP=1
USE_OPENMP ?= 0
ifeq ($(USE_OPENMP),1)
	CXXFLAGS += -fopenmp
endif

.PHONY: all clean

all: $(EXEC)

# Note: Need to avoid feeding .h files to clang command line
$(EXEC): $(SRC_FILES) $(HEADER_DEPS)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC_FILES)

clean:
	rm -rf $(EXEC)
