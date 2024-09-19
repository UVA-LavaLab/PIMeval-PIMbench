# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -Wall -Wextra -Werror -std=c++17 -O3 -fopenmp

# Executable name
EXEC := copy.out

# Source files
SRC_FILES := $(wildcard *.cpp)

#Util Include
INCLUDE_FLAGS := -I../../../../PIMbench


.PHONY: all clean

all: $(EXEC)

$(EXEC): $(SRC_FILES) |
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) -o $@ $^

clean:
	rm -rf $(EXEC)