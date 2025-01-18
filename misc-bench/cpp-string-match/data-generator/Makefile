# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -Wall -Wextra -Werror -std=c++17 -O3 -fopenmp

# Executable name
EXEC := data-generator.out

# Source files
SRC_FILES := data-generator.cpp


.PHONY: all clean

all: $(EXEC)

$(EXEC): $(SRC_FILES) |
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -rf $(EXEC)
