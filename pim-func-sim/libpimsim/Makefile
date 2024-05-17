# Makefile for PIM Functional Simulator - Library
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++17 -Wno-unused-parameter
INC :=
AR := ar
ARFLAGS := rcs
THIRD_PARTY_LIB :=

SRCDIR := src
BUILDDIR := build
LIBDIR := lib
INCDIR := include

SRC := $(wildcard $(SRCDIR)/*.cpp)
OBJ := $(patsubst $(SRCDIR)/%.cpp,$(BUILDDIR)/%.o,$(SRC))

TARGET := $(LIBDIR)/libpimsim.a

.PHONY: debug perf dramsim3_integ clean create_link
.DEFAULT_GOAL := perf

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
	DRAMSIM3_SRC=$(DRAMSIM3_PATH)/src
	DRAMSIM3_EXT=$(DRAMSIM3_PATH)/ext/headers
	INC += -I$(DRAMSIM3_SRC) -I$(DRAMSIM3_EXT)
	THIRD_PARTY_LIB += $(DRAMSIM3_PATH)/build/libdramsim3.a
endif

debug: $(TARGET)
	@echo "INFO: libpimsim target = $(MAKECMDGOALS), CXXFLAGS = $(CXXFLAGS)"

perf: $(TARGET)
	@echo "INFO: libpimsim target = $(MAKECMDGOALS), CXXFLAGS = $(CXXFLAGS)"

dramsim3_integ: $(TARGET)
	@echo "INFO: libpimsim target = $(MAKECMDGOALS), CXXFLAGS = $(CXXFLAGS)"

$(TARGET): $(OBJ) | $(LIBDIR)
	$(AR) $(ARFLAGS) $@ $^ $(THIRD_PARTY_LIB)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(BUILDDIR) $(LIBDIR):
	mkdir -p $@

clean:
	$(RM) -rv $(BUILDDIR) $(LIBDIR)

create_link:
	mkdir -p $(INCDIR)
	if [ ! -e $(INCDIR)/libpimsim.h ] ; then \
		ln -s ../$(SRCDIR)/libpimsim.h $(INCDIR)/libpimsim.h ; \
	fi

