# Compiler settings
NVCC := nvcc
CUDA_DIR ?= $(shell dirname $(shell dirname $(shell which nvcc)))
CUDA_INCLUDE := $(CUDA_DIR)/include
CUDA_LIB_DIR := $(CUDA_DIR)/lib64
FOPENMP := -Xcompiler -fopenmp
ARCH_FLAG := -arch=sm_80

# Target executable
EXEC := game-of-life.out

# Source files
SOURCES := game-of-life.cu

# Header files
INCLUDE_FLAGS := -I../../../../PIMbench

# Compiler flags
CFLAGS := -O3
CXXFLAGS := -std=c++17

# Default target
all: $(EXEC)

# Rule to build the target executable
$(EXEC): $(SOURCES)
	$(NVCC) $(SOURCES) -I$(CUDA_INCLUDE) $(ARCH_FLAG) -L$(CUDA_LIB_DIR) $(CUDA_LIB) $(FOPENMP) $(CFLAGS) $(CXXFLAGS) $(INCLUDE_FLAGS) -o $(EXEC)

# Rule to clean the project
clean:
	rm -f $(EXEC)
	