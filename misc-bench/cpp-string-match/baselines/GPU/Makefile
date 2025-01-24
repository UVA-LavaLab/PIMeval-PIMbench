# Compiler settings
NVCC := nvcc
CUDA_DIR ?= $(shell dirname $(shell dirname $(shell which nvcc)))
CUDA_INCLUDE := $(CUDA_DIR)/include
CUDA_LIB_DIR := $(CUDA_DIR)/lib64
CUDA_LIB := -lcublas
FOPENMP := -Xcompiler -fopenmp
ARCH_FLAG := -arch=sm_75

# Target executable
EXEC := string-match.out

# Source files
SOURCES := string-match.cu

# Header files
INCLUDE_FLAGS := -I./../../include

# PFAC flags
PFAC_ROOT := ./PFAC/PFAC
PFAC_FLAGS := -I$(PFAC_ROOT)/include -L$(PFAC_ROOT)/lib -lpfac

# Compiler flags
CFLAGS := -O3
CXXFLAGS := -std=c++17

# Default target
all: $(EXEC)

# Rule to build the target executable
$(EXEC): $(SOURCES) $(PFAC_ROOT)/lib/libpfac.so
	$(NVCC) $(SOURCES) $(PFAC_FLAGS) -I$(CUDA_INCLUDE) $(ARCH_FLAG) -L$(CUDA_LIB_DIR) $(CUDA_LIB) $(FOPENMP) $(CFLAGS) $(CXXFLAGS) $(INCLUDE_FLAGS) -o $(EXEC)

# Rule to clean the project
clean:
	rm -f $(EXEC)
