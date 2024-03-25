# Makefile: C++ version of vector boradcast and popcount
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

CXX := g++
CXXFLAGS := -std=c++17 -Wall -g
INCLUDE_DIRS := -I../../libpimsim/include
LIB_DIRS := -L../../libpimsim/lib
LIBS := -lpimsim
EXEC := vec-brdcast
SRC := vec-broadcast-popcnt.cpp

.PHONY: all clean $(INCLUDE_DIRS)/libpimsim.h $(LIB_DIRS)/libpimsim.a

all: $(EXEC)

$(EXEC): $(SRC) $(INCLUDE_DIRS)/libpimsim.h $(LIB_DIRS)/libpimsim.a
	$(CXX) $< $(CXXFLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) -o $@

clean:
	rm -f $(EXEC)
