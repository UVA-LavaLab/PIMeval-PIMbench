# Makefile for PIMeval Simulator - Library
# Copyright (c) 2024 University of Virginia
# This file is licensed under the MIT License.
# See the LICENSE file in the root of this repository for more details.

CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++17
CXXFLAGS_DEBUG := -g
# Note: If specify -DNDEBUG, assert will be disabled
CXXFLAGS_PERF := -Ofast -Wno-unused-parameter
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

TARGET := $(LIBDIR)/libpimeval.a

.PHONY: debug perf dramsim3_integ clean create_link
.DEFAULT_GOAL := perf

ifeq ($(MAKECMDGOALS),)
	MAKECMDGOALS := $(.DEFAULT_GOAL)
endif

ifeq ($(MAKECMDGOALS),debug)
	CXXFLAGS += $(CXXFLAGS_DEBUG) -DDEBUG
endif

ifeq ($(MAKECMDGOALS),perf)
	CXXFLAGS += $(CXXFLAGS_PERF)
endif

ifeq ($(MAKECMDGOALS),dramsim3_integ)
	CXXFLAGS += $(CXX_FLAGS_PERF) -DDRAMSIM3_INTEG
	DRAMSIM3_SRC=$(DRAMSIM3_PATH)/src
	DRAMSIM3_EXT=$(DRAMSIM3_PATH)/ext/headers
	INC += -I$(DRAMSIM3_SRC) -I$(DRAMSIM3_EXT)
	THIRD_PARTY_LIB += $(DRAMSIM3_PATH)/build/libdramsim3.a
endif

# Run "make PIM_SIM_TARGET=<PimDeviceEnum>" to override default simulation target
PIM_SIM_TARGET ?= PIM_DEVICE_NONE
CXXFLAGS += -DPIM_SIM_TARGET=$(PIM_SIM_TARGET)

debug: $(TARGET)
	@echo "INFO: libpimeval target = $(MAKECMDGOALS), CXXFLAGS = $(CXXFLAGS)"

perf: $(TARGET)
	@echo "INFO: libpimeval target = $(MAKECMDGOALS), CXXFLAGS = $(CXXFLAGS)"

dramsim3_integ: $(TARGET)
	@echo "INFO: libpimeval target = $(MAKECMDGOALS), CXXFLAGS = $(CXXFLAGS)"

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
	if [ ! -e $(INCDIR)/libpimeval.h ] ; then \
		ln -s ../$(SRCDIR)/libpimeval.h $(INCDIR)/libpimeval.h ; \
	fi

