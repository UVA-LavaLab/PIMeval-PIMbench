# Makefile for PIM Functional Simulator - Library
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.


DRAMSIM3_SRC=$(DRAMSIM3_PATH)/src

DRAMSIM3_EXT=$(DRAMSIM3_PATH)/ext/headers

CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++17 -g
INC := -I$(DRAMSIM3_SRC) -I$(DRAMSIM3_EXT)
AR := ar
ARFLAGS := rcT

SRCDIR := src
BUILDDIR := build
LIBDIR := lib
INCDIR := include

SRC := $(wildcard $(SRCDIR)/*.cpp)
OBJ := $(patsubst $(SRCDIR)/%.cpp,$(BUILDDIR)/%.o,$(SRC))

TARGET := $(LIBDIR)/libpimsim.a

.PHONY: all clean create_link

all: $(TARGET)

$(TARGET): $(OBJ) | $(LIBDIR)
	$(AR) $(ARFLAGS) $@ $^
#	$(AR) $(ARFLAGS) $@ $^ $(DRAMSIM3_PATH)/build/libdramsim3.a

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

