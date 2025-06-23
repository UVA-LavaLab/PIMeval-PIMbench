# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -Wall -Wextra -march=native -Werror -std=c++17 -O3 -fopenmp

# Dependancy
DEP := ../../../../util/

# Executable name
EXEC := knn.out

# Source files
SRC_FILES := $(wildcard *.cpp)

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(SRC_FILES)
	$(CXX) $(CXXFLAGS) -I$(DEP) -o $@ $^ -lopenblas

clean:
	rm -rf $(EXEC)
