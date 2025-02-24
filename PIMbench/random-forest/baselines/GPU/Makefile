# Compiler settings
NVCC := nvcc
CUDA_DIR ?= $(shell dirname $(shell dirname $(shell which nvcc)))
CUDA_INCLUDE := $(CUDA_DIR)/include
CUDA_LIB_DIR := $(CUDA_DIR)/lib64
CUDA_LIB := -lcublas -lcudart
FOPENMP := -Xcompiler -fopenmp
ARCH_FLAG := -arch=sm_80

# Executable name
EXEC := knn.out

# Source files
SOURCES := knncuda.cu

# Compiler flags
CFLAGS := -O3
CXXFLAGS := -std=c++17

# Default target
all: $(EXEC)

$(EXEC): $(SOURCES)
	$(NVCC) $(SOURCES) -I$(CUDA_INCLUDE) $(ARCH_FLAG) -L$(CUDA_LIB_DIR) $(CUDA_LIB) $(FOPENMP) $(CFLAGS) $(CXXFLAGS) -o $(EXEC)
clean:
	rm -f $(EXEC)
