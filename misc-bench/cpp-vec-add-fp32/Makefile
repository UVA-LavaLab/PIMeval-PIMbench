# Makefile: C++ version of FP32 vector addition
# Copyright (c) 2024 University of Virginia
# This file is licensed under the MIT License.
# See the LICENSE file in the root of this repository for more details.

PROJ_ROOT = ../..
include ${PROJ_ROOT}/Makefile.common

EXEC := vec-add-fp32.out
SRC := vec-add-fp32.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM
