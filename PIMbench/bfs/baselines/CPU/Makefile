# Compiler
CXX := g++

# Executable name
EXEC := vec-add.out

# Source files
SRC_FILES := $(wildcard *.cpp)

# Dependancy
DEP := ../../../../util/

# Compiler flags
CXXFLAGS := -Wall -Wextra -march=native -Werror -std=c++17 -O3 -fopenmp

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(SRC_FILES) |
	$(CXX) $(CXXFLAGS) -I$(DEP) -o $@ $^

clean:
	rm -rf $(EXEC)
