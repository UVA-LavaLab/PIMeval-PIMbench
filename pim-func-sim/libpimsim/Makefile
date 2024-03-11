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

.PHONY: all clean create_link

all: $(TARGET)

$(TARGET): $(OBJ) | $(LIBDIR)
	$(AR) $(ARFLAGS) $@ $^

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

