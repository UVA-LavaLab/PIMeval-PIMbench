# Makefile: C++ version of vector add
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

CXX := g++
CXXFLAGS := -std=c++17 -Wall -g
INCLUDE_DIRS := -I../../../libpimsim/include 
LIB_DIRS := -L../../../libpimsim/lib
LIBS := -lpimsim
EXEC := aes
SRCS := PIMAuxilary.cpp main.cpp
OBJS := $(SRCS:.cpp=.o)

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(OBJS) $(CXXFLAGS) $(LIB_DIRS) $(LIBS) -o $@

%.o: %.cpp
	$(CXX) -c $< $(CXXFLAGS) $(INCLUDE_DIRS) -o $@

clean:
	rm -f $(EXEC) $(OBJS)
