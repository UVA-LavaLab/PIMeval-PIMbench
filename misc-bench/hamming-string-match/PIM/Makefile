# Makefile: C++ version of hamming string match
# Copyright (c) 2025 University of Virginia
# This file is licensed under the MIT License.
# See the LICENSE file in the root of this repository for more details.

PROJ_ROOT = ../../..
include ${PROJ_ROOT}/Makefile.common

# make USE_OPENMP=1
USE_OPENMP ?= 0
ifeq ($(USE_OPENMP),1)
	CXXFLAGS += -fopenmp
endif

EXEC := hamming-string-match.out
SRC := hamming-string-match.cpp

STRING_MATCH_INCLUDE = -I./../include

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) $(STRING_MATCH_INCLUDE) -o $@

clean:
	rm -rf $(EXEC) *.dSYM
