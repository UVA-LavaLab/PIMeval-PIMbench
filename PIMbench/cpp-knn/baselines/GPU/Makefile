# Compiler settings
NVCC := nvcc
CUDA_DIR ?= $(shell dirname $(shell dirname $(shell which nvcc)))
CUDA_INCLUDE := $(CUDA_DIR)/include
CUDA_LIB_DIR := $(CUDA_DIR)/lib64
CUDA_LIB := -lcublas
ARCH_FLAG := -arch=sm_80

# Executable name
EXEC := knn.out

# Source files
SOURCES = knncuda.cu test.cpp

# Compiler flags
CFLAGS := -G
CXXFLAGS := -std=c++11

# Default target
all: $(EXEC)

$(EXEC): $(SOURCES)
	$(NVCC) $(SOURCES) $(ARCH_FLAG) -I$(CUDA_INCLUDE) -I./inc -L$(CUDA_LIB_DIR) $(CUDA_LIB) $(CFLAGS) $(CXXFLAGS) -o $(EXEC)
clean:
	rm -f $(EXEC)
