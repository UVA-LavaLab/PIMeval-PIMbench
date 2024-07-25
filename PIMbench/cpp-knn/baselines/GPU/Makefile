# Compiler
CXX := nvcc

# Executable name
EXEC := knn.out

all:
	$(CXX) -o $(EXEC) -I./inc -I/sw/ubuntu-22.04/cuda/extras/CUPTI/include -L/sw/ubuntu-22.04/cuda/extras/CUPTI/lib64 -DONLY_TIME_KERNELS test.cpp -I./inc knncuda.cu -lcuda -lcublas -lcudart -Wno-deprecated-gpu-targets

clean:
	rm -f $(EXEC)
