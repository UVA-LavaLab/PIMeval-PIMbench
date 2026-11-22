# Makefile: C++ version of stencil
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

EXEC := stencil.out
SRC := stencil.cpp

EXEC2 := stencil_alt.out
SRC2 := stencil_alt.cpp

debug perf dramsim3_integ: $(EXEC) $(EXEC2)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

$(EXEC2): $(SRC2) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) $(EXEC2) *.dSYM