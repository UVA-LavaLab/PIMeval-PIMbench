/* File:     vec-add.cu
 * Purpose:  Implement vector addition on a gpu using cuda
 *
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <nvml.h>
#include <chrono>
#include <thread>

#include "utilBaselines.h"

vector<int> A;
vector<int> B;
vector<int> C;

using namespace std;

// Struct for Parameters
struct Params
{
    uint64_t vectorSize = 2048; // Default vector size
};

/**
 * @brief Displays usage information
 */
void usage()
{
    cerr << "\nUsage:  ./vec-add.out [options]\n"
         << "\nOptions:\n"
         << "    -l    vector size (default=2048 elements)\n"
         << "    -h    display this help message\n";
}

/**
 * @brief Parses command line input parameters
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return Parsed parameters
 */
Params parseParams(int argc, char **argv)
{
    Params params;

    int opt;
    while ((opt = getopt(argc, argv, "l:h")) != -1)
    {
        switch (opt)
        {
        case 'h':
            usage();
            exit(0);
        case 'l':
            params.vectorSize = stoull(optarg);
            break;
        default:
            cerr << "\nUnrecognized option: " << opt << "\n";
            usage();
            exit(1);
        }
    }

    return params;
}

__global__ void vecAdd(int* x, int* y, int* z)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    z[index] = x[index] + y[index];
}

int main(int argc, char *argv[])
{
    // Parse input parameters
    Params params = parseParams(argc, argv);
    uint64_t vectorSize = params.vectorSize;
    int *x, *y, *z;
    int blockSize = 1024;
    int numBlock = (vectorSize + blockSize - 1) / blockSize;

    int n_pad = numBlock * blockSize;

    getVector<int32_t>(n_pad, A);
    getVector<int32_t>(n_pad, B);
    C.resize(n_pad);
    std::cout << "Running vector addition for GPU on vector of size: " << vectorSize << std::endl;

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

    std::cout << "Launching CUDA Kernel." << std::endl;

    // CUDA Event Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float timeElapsed = 0;

    // **Get active CUDA device**
    int cudaDevice;
    errorCode = cudaGetDevice(&cudaDevice);
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    // **NVML Power Measurement**
    nvmlReturn_t result;
    nvmlDevice_t device;
    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        return 1;
    }

    result = nvmlDeviceGetHandleByIndex(cudaDevice, &device);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get GPU handle: " << nvmlErrorString(result) << std::endl;
        return 1;
    }
    // **Start Power Measurement**
    unsigned int powerBefore;
    nvmlDeviceGetPowerUsage(device, &powerBefore);

    // Start timer
    cudaEventRecord(start, 0);
    // End timer
    cudaEventRecord(stop, 0);

    /* Kernel Call */
    vecAdd<<<numBlock, blockSize>>>(x, y, z);

    // Check for kernel launch errors
    errorCode = cudaGetLastError();
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeElapsed, start, stop);

    // **End Power Measurement**
    unsigned int powerAfter;
    nvmlDeviceGetPowerUsage(device, &powerAfter);

    float avgPower_mW = (powerBefore + powerAfter) / 2.0;  // Average power in mW

    // **Compute Energy in milliJoules (mJ)**
    float energy_mJ = avgPower_mW * timeElapsed / 1000;

    printf("Execution time of vector addition = %f ms\n", timeElapsed);
    printf("Average Power = %f mW\n", avgPower_mW);
    printf("Energy Consumption = %f mJ\n", energy_mJ);

    errorCode = cudaMemcpy(C.data(), z, vectorSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    for (int i = 0; i < vectorSize; i++)
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

    nvmlShutdown();

    return 0;
} /* main */
