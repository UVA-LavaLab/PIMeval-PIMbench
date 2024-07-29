# Makefile: C++ version of vgg16
# Copyright (c) 2024 University of Virginia
# This file is licensed under the MIT License.
# See the LICENSE file in the root of this repository for more details.

LIBPIMEVAL_PATH = ../../../libpimeval
include ../../Makefile.common

# make USE_OPENMP=1
USE_OPENMP ?= 0
ifeq ($(USE_OPENMP),1)
	CXXFLAGS += -fopenmp
endif

EXEC := vgg16.out
SRC := vgg16.cpp

COMPILE_WITH_JPEG ?= 1
ifeq ($(COMPILE_WITH_JPEG), 1)
    CXXFLAGS += -DCOMPILE_WITH_JPEG
    # Add linker flags for libjpeg
    LDFLAGS += -ljpeg
endif

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) $(LDFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

