# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -Wall -std=c++11 -O3 -fopenmp -lssl -lcrypto 

TARGET := aes.out

# Source file
SRC := aes.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) 
	$(CXX) -o $@ $^ $(CXXFLAGS) 

clean:
	rm $(TARGET)
