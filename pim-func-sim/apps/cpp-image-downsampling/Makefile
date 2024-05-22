# Makefile: C++ version of image downsampling
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

include ../Makefile.common

EXEC := image_downsampling
SRC := image_downsampling.cpp
WARNING_FLAGS := -Wall -Wextra -Wpedantic

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) $(WARNING_FLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

