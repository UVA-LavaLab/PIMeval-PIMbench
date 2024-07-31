# Makefile: C++ version of image downsampling
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

CXX := g++
CXXFLAGS := -std=c++17 -Wall

EXEC := image-downsampling.out
SRC := image-downsampling.cpp

.PHONY: debug perf dramsim3_integ clean
.DEFAULT_GOAL := perf

# Setup build targets
ifeq ($(MAKECMDGOALS),)
	MAKECMDGOALS := $(.DEFAULT_GOAL)
endif
ifeq ($(MAKECMDGOALS),debug)
	CXXFLAGS += -g -DDEBUG
endif
ifeq ($(MAKECMDGOALS),perf)
	CXXFLAGS += -O3
endif
ifeq ($(MAKECMDGOALS),dramsim3_integ)
	CXXFLAGS += -O3 -DDRAMSIM3_INTEG
endif

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC)
	$(CXX) $(SRC) $(CXXFLAGS) -o $@

clean:
	rm -f $(EXEC)