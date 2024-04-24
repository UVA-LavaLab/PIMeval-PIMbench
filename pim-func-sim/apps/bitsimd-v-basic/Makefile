# Makefile: BitSIMD-V basic tests
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

DRAMSIM3_SRC=$(DRAMSIM3_PATH)/src

DRAMSIM3_EXT=$(DRAMSIM3_PATH)/ext/headers

CXX := g++
CXXFLAGS := -std=c++17 -Wall -g
INCLUDE_DIRS := -I../../libpimsim/include -I$(DRAMSIM3_SRC) -I$(DRAMSIM3_EXT)
LIB_DIRS := -L../../libpimsim/lib -L$(DRAMSIM3_PATH)
LIBS := -lpimsim -ldramsim3
EXEC := bitsimd-v-basic
SRC := bitsimd-v-basic.cpp

.PHONY: all clean $(INCLUDE_DIRS)/libpimsim.h $(LIB_DIRS)/libpimsim.a

all: $(EXEC)

$(EXEC): $(SRC) $(INCLUDE_DIRS)/libpimsim.h $(LIB_DIRS)/libpimsim.a
	$(CXX) $< $(CXXFLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) -o $@

clean:
	rm -f $(EXEC)
