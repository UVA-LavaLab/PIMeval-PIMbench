# Makefile: C++ version of image downsampling
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

EXEC := image_downsampling_pim.out
SRCS := image_downsampling.cpp
OBJS := $(SRCS:.cpp=.o)

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(OBJS) $(DEPS)
	$(CXX) $(OBJS) $(CXXFLAGS) -o $@

%.o: %.cpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

clean:
	rm -f $(EXEC) $(OBJS)
