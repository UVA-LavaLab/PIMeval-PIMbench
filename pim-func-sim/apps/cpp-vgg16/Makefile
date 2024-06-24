# Makefile: C++ version of vgg
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

include ../Makefile.common

# make USE_OPENMP=1
USE_OPENMP ?= 0
ifeq ($(USE_OPENMP),1)
	CXXFLAGS += -fopenmp
endif

EXEC := vgg16.out
SRC := vgg.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

