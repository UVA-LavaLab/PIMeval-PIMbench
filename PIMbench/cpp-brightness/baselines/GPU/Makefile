# Compiler settings
NVCC := nvcc
CUDA_DIR ?= $(shell dirname $(shell dirname $(shell which nvcc)))
CUDA_INCLUDE := $(CUDA_DIR)/include
CUDA_LIB_DIR := $(CUDA_DIR)/lib64
CUDA_LIB := -lcublas
ARCH_FLAG := -arch=sm_80

# Target executable
EXEC := brightness.out

# Source files
SOURCES := brightness.cu

# Compiler flags
CFLAGS := -G
CXXFLAGS := -std=c++11

# Default target
all: $(EXEC)

# Rule to build the target executable
$(EXEC): $(SOURCES)
	$(NVCC) $(SOURCES) -I$(CUDA_INCLUDE) $(ARCH_FLAG) -L$(CUDA_LIB_DIR) $(CUDA_LIB) $(CFLAGS) $(CXXFLAGS) -o $(EXEC)

# Rule to clean the project
clean:
	rm -f $(EXEC)
	