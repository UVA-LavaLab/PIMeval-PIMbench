# Makefile: C++ version of vector add
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

CXX := g++
CXXFLAGS := -std=c++17 -Wall -g
INCLUDE_DIRS := -I../../libpimsim/include
LIB_DIRS := -L../../libpimsim/lib
LIBS := -lpimsim
EXEC := sad
SRC := sad.cpp

.PHONY: all clean $(INCLUDE_DIRS)/libpimsim.h $(LIB_DIRS)/libpimsim.a

all: $(EXEC)

$(EXEC): $(SRC) $(INCLUDE_DIRS)/libpimsim.h $(LIB_DIRS)/libpimsim.a
	$(CXX) $< $(CXXFLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) -o $@

clean:
	rm -f $(EXEC)
