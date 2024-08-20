# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -Wall -Wextra -Werror -std=c++11 -O3 -fopenmp
LDFLAGS = -lopenblas

# Executable name
EXEC := gemm.out

# Source files
SRC_FILES := $(wildcard *.cpp)


.PHONY: all clean

all: $(EXEC)

$(EXEC): $(SRC_FILES) |
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf $(EXEC)
	