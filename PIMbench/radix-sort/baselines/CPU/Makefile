# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -Wall -Wextra -march=native -Werror -std=c++17 -O3 -fopenmp

# Executable name
EXEC := radix-sort.out

# Source files
SRC_FILES := $(wildcard *.cpp)


.PHONY: all clean

all: $(EXEC)

$(EXEC): $(SRC_FILES) |
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -rf $(EXEC)
