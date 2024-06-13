# Makefile for Triangle Counting Benchmark
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

include ../Makefile.common

CXXFLAGS += -O3
EXEC := tc.out
SRC := tc.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM


# # Compiler
# CXX := g++

# # Compiler flags
# CXXFLAGS := -Wall -std=c++11 -O3 -fopenmp

# # Setup libpimsim library
# INC_DIR := ../../../libpimsim/include
# LIB_DIR := ../../../libpimsim/lib
# CXXFLAGS += -I$(INC_DIR) -L$(LIB_DIR) -lpimsim
# DEPS := $(INC_DIR)/libpimsim.h $(LIB_DIR)/libpimsim.a

# # Output directory
# BUILD_DIR := ./build

# # Executable name
# TARGET := $(BUILD_DIR)/tc

# .PHONY: all clean

# all: $(TARGET)

# $(TARGET): | $(BUILD_DIR) $(DEPS)
# 	$(CXX) $(CXXFLAGS) -o $@ $^ tc.cpp

# $(BUILD_DIR):
# 	@mkdir -p $(BUILD_DIR)

# clean:
# 	rm -r $(BUILD_DIR)
