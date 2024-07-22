# Makefile: C++ version of vgg13
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

include ../Makefile.common

# make USE_OPENMP=1
USE_OPENMP ?= 0
ifeq ($(USE_OPENMP),1)
	CXXFLAGS += -fopenmp
endif

EXEC := vgg13.out
SRC := vgg13.cpp

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

