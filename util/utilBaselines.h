#ifndef PIM_FUNC_SIM_BASELINE_UTIL_H
#define PIM_FUNC_SIM_BASELINE_UTIL_H

#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <thread>
#include <cstdint>
#include <random>
#include <functional>
#include <chrono>
#include <tuple>
#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace std;

#define WARMUP 4
#define MAX_NUMBER 102

#if defined(ENABLE_CUDA)
#include <cuda_runtime.h>
#include <nvml.h>
std::tuple<double, double, double> measureCUDAPowerAndElapsedTime(std::function<void()> kernelLaunch)
{
    nvmlReturn_t result;
    nvmlDevice_t device;
    cudaEvent_t start, stop;

    std::vector<unsigned> powerSamples;
    bool donePolling = false;

    int cudaDevice;
    cudaError_t errorCode = cudaGetDevice(&cudaDevice);
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    // Initialize NVML
    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        return { 0.0, 0.0, 0.0 };
    }

    result = nvmlDeviceGetHandleByIndex(cudaDevice, &device);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get GPU handle: " << nvmlErrorString(result) << std::endl;
        return { 0.0, 0.0, 0.0 };
    }

    // Start CUDA event timer
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Launch power polling in background
    std::thread powerPoller([&]() {
        while (!donePolling) {
            unsigned tempPower = 0;
            if (nvmlDeviceGetPowerUsage(device, &tempPower) == NVML_SUCCESS) {
                powerSamples.push_back(tempPower);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    // Launch kernel(s)
    kernelLaunch();

    // Wait for kernel completion
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    donePolling = true;
    powerPoller.join();

    // Measure kernel time
    float elapsedTimeMs = 0;
    cudaEventElapsedTime(&elapsedTimeMs, start, stop); // ms

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    nvmlShutdown();
    
    // Compute average power
    double avgPower = 0.0;
    for (auto p : powerSamples) avgPower += p;
    if (!powerSamples.empty()) avgPower /= powerSamples.size(); // in mW
    std::cout << "Power Sample Collected: " << powerSamples.size() << std::endl;
    // Convert to energy in mJ
    double energy = avgPower * (elapsedTimeMs / 1000.0); // mW × s = mJ

    return { elapsedTimeMs, avgPower, energy };
}
#endif

/**
 * @brief Initializes a vector with random values.
 *
 * This function creates a vector of the specified size and fills it with random
 * values ranging from 0 to MAX_NUMBER. It uses parallelism to speed up the initialization.
 *
 * @param vectorSize The size of the vector to be created.
 * @param dataVector A reference to the vector that will be initialized.
 */
template <typename T>
void getVector(uint64_t vectorSize, std::vector<T>& dataVector) {
    // Resize the vector to the specified size
    dataVector.resize(vectorSize);

    #pragma omp parallel for
    for (uint64_t i = 0; i < vectorSize; ++i) {
        dataVector[i] = static_cast<T>(i % MAX_NUMBER);
    }
}

/**
 * @brief Initializes a matrix with random values.
 *
 * This function creates a matrix with the specified number of rows and columns
 * and fills it with random values ranging from 0 to MAX_NUMBER. It uses parallelism
 * to speed up the initialization.
 *
 * @param numRows The number of rows in the matrix.
 * @param numCols The number of columns in the matrix.
 * @param matrix A reference to the matrix that will be initialized.
 */
template <typename T>
void getMatrix(uint64_t numRows, uint64_t numCols, vector<vector<T>> &matrix) {

    // Resize the matrix to the specified number of rows and columns
    matrix.resize(numRows, vector<T>(numCols));

    #pragma omp parallel for
    for (uint64_t row = 0; row < numRows; ++row) {
        for (uint64_t col = 0; col < numCols; ++col) {
            matrix[row][col] = static_cast<T>((row + 1) % MAX_NUMBER);
        }
    }
}

/**
 * @brief Initializes a matrix with random bool values.
 *
 * This function creates a matrix with the specified number of rows and columns
 * and fills it with random values ranging from true to false. It uses parallelism
 * to speed up the initialization. The matrix is stored values are stored as uint8_t
 * because the PIM API expects std::vector<uint8_t> instead of std::vector<bool>.
 *
 * @param numRows The number of rows in the matrix.
 * @param numCols The number of columns in the matrix.
 * @param matrix A reference to the matrix that will be initialized.
 */
void getMatrixBool(uint64_t numRows, uint64_t numCols, std::vector<std::vector<uint8_t>> &matrix)
{
  matrix.resize(numRows, std::vector<uint8_t>(numCols, 0));
#pragma omp parallel
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dist(0, 1);

    #pragma omp for collapse(2)
    for(size_t i = 0; i < numRows; ++i) {
      for(size_t j = 0; j < numCols; ++j) {
        matrix[i][j] = static_cast<bool>(dist(gen));
      }
    }
  }
}

#endif // _COMMON_H_
