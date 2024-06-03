# Compiler
NVCC := nvcc

# Compiler flags
NVCCFLAGS := -O3

# Output directory
BUILD_DIR := ./build

# Executable name
TARGET := $(BUILD_DIR)/aes

.PHONY: all clean

all: $(TARGET)

$(TARGET): | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ aes.cu

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

clean:
	rm -r $(BUILD_DIR)
