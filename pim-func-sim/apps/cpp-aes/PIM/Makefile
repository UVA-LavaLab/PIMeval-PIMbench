# Makefile: C++ version of AES
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

CXX := g++
CXXFLAGS := -Wall -std=c++11 -O3 -fopenmp

# Setup libpimsim library
INC_DIR := ../../../libpimsim/include
LIB_DIR := ../../../libpimsim/lib
CXXFLAGS += -I$(INC_DIR) -L$(LIB_DIR) -fopenmp -lpimsim
DEPS := $(INC_DIR)/libpimsim.h $(LIB_DIR)/libpimsim.a

EXEC := aes.out
SRCS := PIMAuxilary.cpp main.cpp
OBJS := $(SRCS:.cpp=.o)

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
	CXXFLAGS += -Ofast
endif
ifeq ($(MAKECMDGOALS),dramsim3_integ)
	CXXFLAGS += -Ofast -DDRAMSIM3_INTEG
endif

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(OBJS) $(DEPS)
	$(CXX) $(OBJS) $(CXXFLAGS) -o $@

%.o: %.cpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

clean:
	rm -f $(EXEC) $(OBJS)
