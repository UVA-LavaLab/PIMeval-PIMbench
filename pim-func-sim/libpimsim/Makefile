# Makefile for PIM Functional Simulator - Library
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++17 -g
INC :=
AR := ar
ARFLAGS := rcs

SRCDIR := src
BUILDDIR := build
LIBDIR := lib
INCDIR := include

SRC := $(wildcard $(SRCDIR)/*.cpp)
OBJ := $(patsubst $(SRCDIR)/%.cpp,$(BUILDDIR)/%.o,$(SRC))

TARGET := $(LIBDIR)/libpimsim.a
HEADER := $(INCDIR)/libpimsim.h

.PHONY: all clean

all: $(TARGET) $(HEADER)

$(TARGET): $(OBJ) | $(LIBDIR)
	$(AR) $(ARFLAGS) $@ $^

$(HEADER): $(INCDIR)
	cp $(SRCDIR)/libpimsim.h $(INCDIR)/libpimsim.h

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(BUILDDIR) $(INCDIR) $(LIBDIR):
	mkdir -p $@

clean:
	$(RM) -rv $(BUILDDIR) $(INCDIR) $(LIBDIR)

