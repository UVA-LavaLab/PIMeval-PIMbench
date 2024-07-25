/* File:     vec_add.cu
 * Purpose:  Implement vector addition on a gpu using cuda
 *
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include "../../../util.h"

vector<int> A;
vector<int> B;
vector<int> C;

using namespace std;

__global__ void vecAdd(int* x, int* y, int* z)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    z[index] = x[index] + y[index];
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cout << argc << "\n";
        printf("Vector size required.\n");
        printf("Syntax: %s <vector_size>.\n", argv[0]);
        exit(1);
    }

    uint64_t n = atoll(argv[1]);
    getVector(n, A);
    getVector(n, B);
    C.resize(n);

    int *x, *y, *z;
    int blockSize = 1024;
    int numBlock = (n + blockSize - 1) / blockSize;

    int n_pad = numBlock * blockSize;
    cudaError_t errorCode;

    errorCode = cudaMalloc(&x, n_pad * sizeof(int));
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }
    errorCode = cudaMalloc(&y, n_pad * sizeof(int));
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }
    errorCode = cudaMalloc(&z, n_pad * sizeof(int));
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    errorCode = cudaMemcpy(x, A.data(), n_pad * sizeof(int), cudaMemcpyHostToDevice);
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    errorCode = cudaMemcpy(y, B.data(), n_pad * sizeof(int), cudaMemcpyHostToDevice);
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    // Event creation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float timeElapsed = 0;

    // Start timer
    cudaEventRecord(start, 0);

    /* Kernel Call */
    vecAdd<<<numBlock, blockSize>>>(x, y, z);

    // Check for kernel launch errors
    errorCode = cudaGetLastError();
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    // End timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeElapsed, start, stop);

    printf("Execution time = %f ms\n", timeElapsed);
    errorCode = cudaMemcpy(C.data(), z, n * sizeof(int), cudaMemcpyDeviceToHost);
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    for (int i = 0; i < n; i++)
    {
        if (C[i] != A[i] + B[i])
        {
            cout << "Addition failed at index: " << i << " value: " << z[i] << endl;
            break;
        }
    }

    /* Free memory */
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return 0;
} /* main */
