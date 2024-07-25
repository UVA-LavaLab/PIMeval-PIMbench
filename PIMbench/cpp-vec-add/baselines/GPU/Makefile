# Compiler settings
NVCC := nvcc
CUDA_DIR ?= $(shell dirname $(shell dirname $(shell which nvcc)))
CUDA_INCLUDE := $(CUDA_DIR)/include
CUDA_LIB_DIR := $(CUDA_DIR)/lib64
CUDA_LIB := -lcublas
ARCH_FLAG := -arch=sm_80

# Target executable
TARGET := vec_add.out

# Source files
SOURCES := vec_add.cu

# Compiler flags
CFLAGS := -G

# Default target
all: $(TARGET)

# Rule to build the target executable
$(TARGET): $(SOURCES)
	$(NVCC) $(SOURCES) -I$(CUDA_INCLUDE) $(ARCH_FLAG) -L$(CUDA_LIB_DIR) $(CUDA_LIB) $(CFLAGS) -o $(TARGET)

# Rule to clean the project
clean:
	rm -f $(TARGET)