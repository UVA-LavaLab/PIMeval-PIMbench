# Compiler
NVCC := nvcc

# Compiler flags
NVCCFLAGS := -O3

# Output directory
BUILD_DIR := ./build

# Executable name
TARGET := $(BUILD_DIR)/aes.out

# Source file
SRC := aes.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

clean:
	rm -r $(BUILD_DIR)

