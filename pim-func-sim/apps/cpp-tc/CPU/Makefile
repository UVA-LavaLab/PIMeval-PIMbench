# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -Wall -std=c++11 -O3 -fopenmp

# Output directory
BUILD_DIR := ./build

# Executable name
TARGET := $(BUILD_DIR)/TriCountBitSimd

.PHONY: all clean

all: $(TARGET)

$(TARGET): | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ TriCountBitSimd.cpp

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

clean:
	rm -r $(BUILD_DIR)