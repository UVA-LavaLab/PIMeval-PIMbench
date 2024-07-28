# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -Wall -std=c++11 -O3 -fopenmp -lssl -lcrypto

# Output directory
BUILD_DIR := ./build

# Executable name
TARGET := $(BUILD_DIR)/aes.out

# Source file
SRC := aes.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) | $(BUILD_DIR)
	$(CXX) -o $@ $^ $(CXXFLAGS) 

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

clean:
	rm -r $(BUILD_DIR)
