/* File:     lr.cu
 * Purpose:  Implement linear regression on a gpu using cuda
 *
 */

#include <cstdint>
#include <iostream>
#include <math.h>
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "utilBaselines.h"

#define BLOCK_SIZE 512

using namespace std;

// Params ---------------------------------------------------------------------
typedef struct Params {
  uint64_t dataSize;
  std::string inputFile;
} Params;

void usage() {
  fprintf(stderr, "\nUsage:  ./lr.out [options]"
                  "\n"
                  "\n    -l    input size (default=2048 elements)"
                  "\n    -i    input file containing 2D matrix "
                  "(default=generates matrix with random numbers)"
                  "\n");
}

struct Params getInputParams(int argc, char **argv) {
  struct Params p;
  p.dataSize = 2048;
  p.inputFile = "";

  int opt;
  while ((opt = getopt(argc, argv, "h:l:i:")) >= 0) {
    switch (opt) {
    case 'h':
      usage();
      exit(0);
      break;
    case 'l':
      p.dataSize = strtoull(optarg, NULL, 0);
      break;
    case 'i':
      p.inputFile = optarg;
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

void initData(uint64_t numPoints, int32_t *dataPoints, uint64_t n_pad) {
  // Providing a seed value
  srand((unsigned)time(NULL));
  for (uint64_t i = 0; i < n_pad; i++) {
    for (uint64_t j = 0; j < 2; j++) {
      uint64_t index = j * n_pad + i;
      if (i >= numPoints) {
        dataPoints[index] = 0;
      } else {
        dataPoints[index] = rand() % 16;
      }
    }
  }
}

__device__ void warpReduce(uint64_t tid, volatile int32_t *dataArray) {
  dataArray[tid] += dataArray[tid + 32];
  dataArray[tid] += dataArray[tid + 16];
  dataArray[tid] += dataArray[tid + 8];
  dataArray[tid] += dataArray[tid + 4];
  dataArray[tid] += dataArray[tid + 2];
  dataArray[tid] += dataArray[tid + 1];
}

__global__ void LR(int32_t *points, int32_t *SX, int32_t *SY, int32_t *SXX,
                   int32_t *SYY, int32_t *SXY, uint64_t n_pad, uint64_t n) {
  uint64_t threadID = threadIdx.x;
  uint64_t index = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  __shared__ int32_t partialSumSX[BLOCK_SIZE], partialSumSY[BLOCK_SIZE],
      partialSumSXX[BLOCK_SIZE], partialSumSYY[BLOCK_SIZE],
      partialSumSXY[BLOCK_SIZE];

  if (index + blockDim.x < n) {
    partialSumSX[threadID] =
        points[index + n_pad * 0] + points[index + blockDim.x + n_pad * 0];
    partialSumSY[threadID] =
        points[index + n_pad * 1] + points[index + blockDim.x + n_pad * 1];
    partialSumSXX[threadID] =
        (points[index + n_pad * 0] * points[index + n_pad * 0]) +
        (points[index + blockDim.x + n_pad * 0] *
         points[index + blockDim.x + n_pad * 0]);
    partialSumSYY[threadID] =
        points[index + n_pad * 1] * points[index + n_pad * 1] +
        points[index + blockDim.x + n_pad * 1] *
            points[index + blockDim.x + n_pad * 1];
    partialSumSXY[threadID] =
        points[index + n_pad * 0] * points[index + n_pad * 1] +
        points[index + blockDim.x + n_pad * 0] *
            points[index + blockDim.x + n_pad * 1];
  } else {
    partialSumSX[threadID] = points[index + n_pad * 0];
    partialSumSY[threadID] = points[index + n_pad * 1];
    partialSumSXX[threadID] =
        (points[index + n_pad * 0] * points[index + n_pad * 0]);
    partialSumSYY[threadID] =
        points[index + n_pad * 1] * points[index + n_pad * 1];
    partialSumSXY[threadID] =
        points[index + n_pad * 0] * points[index + n_pad * 1];
  }

  __syncthreads();

  for (int i = blockDim.x / 2; i > 32; i >>= 1) {
    uint64_t currIndex = threadID;
    if (currIndex < i) {
      partialSumSX[currIndex] += partialSumSX[currIndex + i];
      partialSumSY[currIndex] += partialSumSY[currIndex + i];
      partialSumSXX[currIndex] += partialSumSXX[currIndex + i];
      partialSumSYY[currIndex] += partialSumSYY[currIndex + i];
      partialSumSXY[currIndex] += partialSumSXY[currIndex + i];
    }
    __syncthreads();
  }

  if (threadID < 32) {
    warpReduce(threadID, partialSumSX);
    warpReduce(threadID, partialSumSY);
    warpReduce(threadID, partialSumSXX);
    warpReduce(threadID, partialSumSYY);
    warpReduce(threadID, partialSumSXY);
  }

  if (threadID == 0) {
    SX[blockIdx.x] = partialSumSX[0];
    SY[blockIdx.x] = partialSumSY[0];
    SXX[blockIdx.x] = partialSumSXX[0];
    SYY[blockIdx.x] = partialSumSYY[0];
    SXY[blockIdx.x] = partialSumSXY[0];
  }
}

__global__ void LR_Wrap(int32_t *SX, int32_t *SY, int32_t *SXX, int32_t *SYY,
                        int32_t *SXY, uint64_t n) {
  int index = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  int gridSize = BLOCK_SIZE * 2 * gridDim.x;
  __shared__ int32_t partialSumSX[BLOCK_SIZE], partialSumSY[BLOCK_SIZE],
      partialSumSXX[BLOCK_SIZE], partialSumSYY[BLOCK_SIZE],
      partialSumSXY[BLOCK_SIZE];

  partialSumSX[threadIdx.x] = 0;
  partialSumSY[threadIdx.x] = 0;
  partialSumSXX[threadIdx.x] = 0;
  partialSumSYY[threadIdx.x] = 0;
  partialSumSXY[threadIdx.x] = 0;

  while (index < n) {
    if (index + blockDim.x < n) {
      partialSumSX[threadIdx.x] += SX[index] + SX[index + blockDim.x];
      partialSumSY[threadIdx.x] += SY[index] + SY[index + blockDim.x];
      partialSumSXX[threadIdx.x] += SXX[index] + SXX[index + blockDim.x];
      partialSumSYY[threadIdx.x] += SYY[index] + SYY[index + blockDim.x];
      partialSumSXY[threadIdx.x] += SXY[index] + SXY[index + blockDim.x];
    } else {
      partialSumSX[threadIdx.x] += SX[index];
      partialSumSY[threadIdx.x] += SY[index];
      partialSumSXX[threadIdx.x] += SXX[index];
      partialSumSYY[threadIdx.x] += SYY[index];
      partialSumSXY[threadIdx.x] += SXY[index];
    }
    index += gridSize;
  }

  __syncthreads();

  for (int i = blockDim.x / 2; i > 32; i >>= 1) {
    if (threadIdx.x < i) {
      partialSumSX[threadIdx.x] += partialSumSX[threadIdx.x + i];
      partialSumSY[threadIdx.x] += partialSumSY[threadIdx.x + i];
      partialSumSXX[threadIdx.x] += partialSumSXX[threadIdx.x + i];
      partialSumSYY[threadIdx.x] += partialSumSYY[threadIdx.x + i];
      partialSumSXY[threadIdx.x] += partialSumSXY[threadIdx.x + i];
    }
    __syncthreads();
  }

  if (threadIdx.x < 32) {
    warpReduce(threadIdx.x, partialSumSX);
    warpReduce(threadIdx.x, partialSumSY);
    warpReduce(threadIdx.x, partialSumSXX);
    warpReduce(threadIdx.x, partialSumSYY);
    warpReduce(threadIdx.x, partialSumSXY);
  }

  if (threadIdx.x == 0) {
    SX[blockIdx.x] = partialSumSX[0];
    SY[blockIdx.x] = partialSumSY[0];
    SXX[blockIdx.x] = partialSumSXX[0];
    SYY[blockIdx.x] = partialSumSYY[0];
    SXY[blockIdx.x] = partialSumSXY[0];
  }
}

int main(int argc, char *argv[]) {
  struct Params params = getInputParams(argc, argv);

  if (params.inputFile != "") {
    std::cout << "Need work reading in files" << std::endl;
    return 1;
  }

  uint64_t n = params.dataSize;
  int32_t *dataPoints_h, *dataPoints_d;
  int32_t *SX, *SY, *SXX, *SYY, *SXY;

  int numBlock = ceil((ceil((n * 1.0) / BLOCK_SIZE)) / 2);

  uint64_t n_pad = numBlock * BLOCK_SIZE * 2;
  cudaError_t errorCode;

  dataPoints_h = (int32_t*)malloc(sizeof(int32_t) * n_pad * 2);
  initData(n, dataPoints_h, n_pad);
  errorCode = cudaMalloc(&dataPoints_d, n_pad * 2 * sizeof(int32_t));
  if (errorCode != cudaSuccess) {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }
  errorCode = cudaMemcpy(dataPoints_d, dataPoints_h, n_pad * 2 * sizeof(int32_t), cudaMemcpyHostToDevice);
  if (errorCode != cudaSuccess) {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }
  errorCode = cudaMalloc(&SX, numBlock * sizeof(int32_t));
  if (errorCode != cudaSuccess) {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }
  errorCode = cudaMalloc(&SY, numBlock * sizeof(int32_t));
  if (errorCode != cudaSuccess) {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }
  errorCode = cudaMalloc(&SXX, numBlock * sizeof(int32_t));
  if (errorCode != cudaSuccess) {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }
  errorCode = cudaMalloc(&SYY, numBlock * sizeof(int32_t));
  if (errorCode != cudaSuccess) {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }
  errorCode = cudaMalloc(&SXY, numBlock * sizeof(int32_t));
  if (errorCode != cudaSuccess) {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  cout << "Num Block: " << numBlock << "\tn_pad: " << n_pad << "\n";

  // Start timer
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float timeElapsed = 0;
  // **Get active CUDA device**
  int cudaDevice;
  errorCode = cudaGetDevice(&cudaDevice);
  if (errorCode != cudaSuccess) {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  nvmlReturn_t result;
  nvmlDevice_t device;
  result = nvmlInit();
  if (result != NVML_SUCCESS) {
    std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result)
              << std::endl;
    return 1;
  }

  result = nvmlDeviceGetHandleByIndex(cudaDevice, &device);
  if (result != NVML_SUCCESS) {
    std::cerr << "Failed to get GPU handle: " << nvmlErrorString(result)
              << std::endl;
    return 1;
  }

  // Vector for power sampling
  std::vector<unsigned> powerSamples;

  cudaEventRecord(start, 0);

  /* Kernel Call */
  LR<<<numBlock, BLOCK_SIZE>>>(dataPoints_d, SX, SY, SXX, SYY, SXY, n_pad, n);
  LR_Wrap<<<1, BLOCK_SIZE>>>(SX, SY, SXX, SYY, SXY, numBlock);
  while (true) {
    unsigned tempPower;
    if (nvmlDeviceGetPowerUsage(device, &tempPower) == NVML_SUCCESS) {
      powerSamples.push_back(tempPower);
    }
    if (cudaEventQuery(stop) == cudaSuccess) {
      break;
    }
  }
  cudaDeviceSynchronize();

  // End timer
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timeElapsed, start, stop);
  double totalPower = 0;
  for (size_t i = 0; i < powerSamples.size(); ++i) {
    totalPower += powerSamples[i];
  }

  printf("Total Power: %lu\n\n\n", totalPower);

  double avgPower_mW = totalPower / powerSamples.size(); // Average power in mW

  // **Compute Energy in milliJoules (mJ)**
  float energy_mJ = avgPower_mW * timeElapsed / 1000;

  printf("Execution time = %f ms\n", timeElapsed);
  printf("Average Power = %f mW\n", avgPower_mW);
  printf("Energy Consumption = %f mJ\n", energy_mJ);

  // Calculate slope and intercept
  int32_t SX_host, SY_host, SXX_host, SXY_host;
  // Copy the first element from device to host
  cudaMemcpy(&SX_host, SX, sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&SY_host, SY, sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&SXX_host, SXX, sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&SXY_host, SXY, sizeof(int32_t), cudaMemcpyDeviceToHost);
  auto slopeD = (n * SXY_host - SX_host * SY_host) / (n * SXX_host - SX_host * SX_host);
  auto interceptD = (SY_host - slopeD * SX_host) / n;

  int32_t SX_ll = 0, SY_ll = 0, SXX_ll = 0, SYY_ll = 0, SXY_ll = 0;
  for (int i = 0; i < n; i++) {
    SX_ll += dataPoints_h[i + n_pad * 0];
    SXX_ll += dataPoints_h[i + n_pad * 0] * dataPoints_h[i + n_pad * 0];
    SY_ll += dataPoints_h[i + n_pad * 1];
    SYY_ll += dataPoints_h[i + n_pad * 1] * dataPoints_h[i + n_pad * 1];
    SXY_ll += dataPoints_h[i + n_pad * 0] * dataPoints_h[i + n_pad * 1];
  }

  auto slopeH = (n * SXY_ll - SX_ll * SY_ll) / (n * SXX_ll - SX_ll * SX_ll);
  auto interceptH = (SY_ll - slopeH * SX_ll) / n;

  if (interceptH != interceptD) {
    std::cout << "Wrong Answer" << std::endl;
  }

  // Free memory
  cudaFree(SX);
  cudaFree(SY);
  cudaFree(SXX);
  cudaFree(SYY);
  cudaFree(SXY);
  cudaFree(dataPoints_d);

  return 0;
}
