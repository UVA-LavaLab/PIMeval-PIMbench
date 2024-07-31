# Makefile: GPU version of image downsampling
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

NVCC := nvcc
# NVCCFLAGS := 
CXXFLAGS := -std=c++17

EXEC := image-downsampling.out
SRC := image-downsampling.cu

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
	$(NVCC) $(SRC) $(CXXFLAGS) -o $@

clean:
	rm -f $(EXEC)