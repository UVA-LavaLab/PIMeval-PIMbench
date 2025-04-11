/* File:     vec-add.cu
 * Purpose:  Implement vector addition on a gpu using cuda
 *
 */

#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <nvml.h>
#include <chrono>
#include <thread>

#include "utilBaselines.h"

vector<float> A;
vector<float> B;
vector<float> C;

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

__global__ void vecAdd(float* x, float* y, float* z)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    z[index] = x[index] + y[index];
}

int main(int argc, char *argv[])
{
    // Parse input parameters
    Params params = parseParams(argc, argv);
    uint64_t vectorSize = params.vectorSize;
    float *x, *y, *z;
    int blockSize = 1024;
    u_int64_t numBlock = (vectorSize + blockSize - 1) / blockSize;

    uint64_t n_pad = numBlock * blockSize;

    getVector<float>(n_pad, A);
    getVector<float>(n_pad, B);
    C.resize(n_pad);
    std::cout << "Running vector addition for GPU on vector of size: " << vectorSize << std::endl;

    cudaError_t errorCode;

    errorCode = cudaMalloc(&x, n_pad * sizeof(float));
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }
    errorCode = cudaMalloc(&y, n_pad * sizeof(float));
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }
    errorCode = cudaMalloc(&z, n_pad * sizeof(float));
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    errorCode = cudaMemcpy(x, A.data(), vectorSize * sizeof(float), cudaMemcpyHostToDevice);
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    errorCode = cudaMemcpy(y, B.data(), vectorSize * sizeof(float), cudaMemcpyHostToDevice);
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    std::cout << "Launching CUDA Kernel." << std::endl;

    // // CUDA Event Timing
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // float timeElapsed = 0;

    // // **Get active CUDA device**
    // int cudaDevice;
    // errorCode = cudaGetDevice(&cudaDevice);
    // if (errorCode != cudaSuccess)
    // {
    //     cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    //     exit(1);
    // }

    // nvmlReturn_t result;
    // nvmlDevice_t device;
    // result = nvmlInit();
    // if (result != NVML_SUCCESS) {
    //     std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
    //     return 1;
    // }

    // result = nvmlDeviceGetHandleByIndex(cudaDevice, &device);
    // if (result != NVML_SUCCESS) {
    //     std::cerr << "Failed to get GPU handle: " << nvmlErrorString(result) << std::endl;
    //     return 1;
    // }
    
    // // Vector for power sampling
    // std::vector<unsigned> powerSamples;
    // bool donePolling = false;
    // // Start timer
    // cudaEventRecord(start, 0);

    // /* Kernel Call */
    // std::thread powerPoller([&]() {
    // while (true) {
    //     unsigned tempPower;
    //     if (nvmlDeviceGetPowerUsage(device, &tempPower) == NVML_SUCCESS) {
    //         powerSamples.push_back(tempPower);
    //     }
    //     std::this_thread::sleep_for(std::chrono::milliseconds(1)); // optional: to avoid hammering NVML
    //     if (donePolling) break;
    // }
    // });

    auto [timeElapsed, avgPower, energy] = measureCUDAPowerAndElapsedTime([&]() {
        vecAdd<<<numBlock, blockSize>>>(x, y, z);
        cudaDeviceSynchronize(); // ensure all are done
    });

    // Check for kernel launch errors
    errorCode = cudaGetLastError();
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }
    
    // while (true) {
    //     unsigned tempPower;
    //     if (nvmlDeviceGetPowerUsage(device, &tempPower) == NVML_SUCCESS) {
    //         powerSamples.push_back(tempPower);
    //     }
    //     if (cudaEventQuery(stop) == cudaSuccess) {
    //         break;
    //     }
    // }

    // End timer
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&timeElapsed, start, stop);
    // donePolling = true;
    // powerPoller.join();

    // double totalPower = 0;
    // for (size_t i = 0; i < powerSamples.size(); ++i) {
    //     totalPower += powerSamples[i];
    //     std::cout << "total Power: " << totalPower << "\t";
    // }

    // double avgPower_mW = totalPower / powerSamples.size();  // Average power in mW

    // // **Compute Energy in milliJoules (mJ)**
    // float energy_mJ = avgPower_mW * timeElapsed / 1000;

    printf("\nExecution time of vector addition = %f ms\n", timeElapsed);
    printf("Average Power = %f mW\n", avgPower);
    printf("Energy Consumption = %f mJ\n", energy);

    errorCode = cudaMemcpy(C.data(), z, vectorSize * sizeof(float), cudaMemcpyDeviceToHost);
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    // for (uint64_t i = 0; i < vectorSize; i++)
    // {
    //     if (C[i] != A[i] + B[i])
    //     {
    //         cout << "Addition failed at index: " << i << " value: " << z[i] << endl;
    //         break;
    //     }
    // }

    /* Free memory */
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    nvmlShutdown();

    return 0;
} /* main */
