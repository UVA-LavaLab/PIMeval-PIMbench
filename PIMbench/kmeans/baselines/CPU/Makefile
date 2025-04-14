# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -Wall -Wextra -Werror -std=c++17 -O3 -fopenmp

# Executable name
EXEC := km.out

# Source files
SRC_FILES := $(wildcard *.cpp)

# Dependancy
DEP := ../../../../util/

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(SRC_FILES) |
	$(CXX) $(CXXFLAGS) -I$(DEP) -o $@ $^

clean:
	rm -rf $(EXEC)
