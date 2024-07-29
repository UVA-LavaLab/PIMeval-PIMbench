# Compiler
NVCC := nvcc

# Compiler flags
NVCCFLAGS := -O3

# Executable name
TARGET := aes.out

# Source file
SRC := aes.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

clean:
	rm $(TARGET)

