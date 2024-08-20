# Compiler settings
NVCC := nvcc
CUDA_DIR ?= $(shell dirname $(shell dirname $(shell which nvcc)))
CUDA_INCLUDE := $(CUDA_DIR)/include
CUDA_LIB_DIR := $(CUDA_DIR)/lib64
CUDA_LIB := -lcublas -lcudart
#ARCH_FLAG := -arch=sm_80

# Executable name
EXEC := knn.out

# Source files
SOURCES := knncuda.cu

# Compiler flags
CFLAGS := -G
CXXFLAGS := -std=c++11

# Default target
all: $(EXEC)

$(EXEC): $(SOURCES)
	nvcc knncuda.cu -o knn.out
clean:
	rm -f $(EXEC)
