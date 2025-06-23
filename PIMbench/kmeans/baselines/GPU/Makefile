# Compiler settings
NVCC := nvcc
CUDA_DIR ?= $(shell dirname $(shell dirname $(shell which nvcc)))
CUDA_INCLUDE := $(CUDA_DIR)/include
CUDA_LIB_DIR := $(CUDA_DIR)/lib64
CUDA_LIB := -lcublas -lnvidia-ml
FOPENMP := -Xcompiler -fopenmp
ARCH_FLAG := -arch=sm_80

# Target executable
EXEC := km.out

# Source files
SOURCES := km.cu

# Compiler flags
CFLAGS := -O3
CXXFLAGS := -std=c++17

# Dependancy
DEP := ../../../../util/

# Default target
all: $(EXEC)

# Rule to build the target executable
$(EXEC): $(SOURCES)
	$(NVCC) $(SOURCES) -DENABLE_CUDA -I$(CUDA_INCLUDE) -I$(DEP) $(ARCH_FLAG) -L$(CUDA_LIB_DIR) $(CUDA_LIB) $(FOPENMP) $(CFLAGS) $(CXXFLAGS) -o $(EXEC)

# Rule to clean the project
clean:
	rm -f $(EXEC)
	