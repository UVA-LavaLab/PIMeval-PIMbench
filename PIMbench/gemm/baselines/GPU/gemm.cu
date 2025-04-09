/* File:     gemm.cu
 * Purpose:  Implement gemm on a gpu
 *
 */
// TODO: support different data type

#include <cublas_v2.h>
#include <iostream>
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "utilBaselines.h"

using namespace std;

#define TOLERANCE 200.0f

vector<float> A;
vector<float> B;
vector<float> C;

// Params ---------------------------------------------------------------------
typedef struct Params {
  uint64_t row, column_A, column_B;
} Params;

void usage() {
  fprintf(stderr, "\nUsage:  ./gemm.out [options]"
                  "\n"
                  "\n    -r <R>    row size"
                  "\n    -c <C>    MatA column size"
                  "\n    -d <C>    MatB column size"
                  "\n");
}

struct Params input_params(int argc, char **argv) {
  struct Params p;
  p.row = 1024;
  p.column_A = 1024;
  p.column_B = 1024;

  int opt;
  while ((opt = getopt(argc, argv, ":r:c:d:")) >= 0) {
    switch (opt) {
    case 'h':
      usage();
      exit(0);
      break;
    case 'r':
      p.row = atoll(optarg);
      break;
    case 'c':
      p.column_A = atoll(optarg);
      break;
    case 'd':
      p.column_B = atoll(optarg);
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

int main(int argc, char *argv[]) {
  struct Params p = input_params(argc, argv);

  uint64_t row = p.row, col_A = p.column_A, col_B = p.column_B;

  getVector(row * col_A, A);
  getVector(col_A * col_B, B);

  float *x, *y, *z;

  cudaError_t errorCode;

  errorCode = cudaMalloc(&x, row * col_A * sizeof(int32_t));
  if (errorCode != cudaSuccess) {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }
  errorCode = cudaMalloc(&y, col_A * col_B * sizeof(int32_t));
  if (errorCode != cudaSuccess) {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }
  errorCode = cudaMalloc(&z, row * col_B * sizeof(int32_t));
  if (errorCode != cudaSuccess) {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  errorCode = cudaMemcpy(x, A.data(), row * col_A * sizeof(float),
                         cudaMemcpyHostToDevice);
  if (errorCode != cudaSuccess) {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  errorCode = cudaMemcpy(y, B.data(), col_A * col_B * sizeof(float),
                         cudaMemcpyHostToDevice);
  if (errorCode != cudaSuccess) {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  const float alpha = 1.0;
  const float beta = 1.0;
  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate_v2(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CUBLAS initialization failed\n";
    exit(1);
  }

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

  // Variables for power sampling
  std::vector<unsigned int> powerSamples;

  // Event creation
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float timeElapsed = 0;

  // Start timer
  cudaEventRecord(start, 0);
  /* Kernel Call */
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, row, col_B, col_A, &alpha, x,
              row, y, col_A, &beta, z, row);

  while (true) {
    unsigned int power;
    if (nvmlDeviceGetPowerUsage(device, &power) == NVML_SUCCESS) {
      powerSamples.push_back(power);
    }
    if (cudaEventQuery(stop) == cudaSuccess) {
      break;
    }
  }

  // End timer
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timeElapsed, start, stop);

  double totalPower = 0;
  for (size_t i = 0; i < powerSamples.size(); ++i) {
    totalPower += powerSamples[i]; // Convert mW to W * time
  }

  float avgPower_mW = totalPower / powerSamples.size(); // Average power in mW

  // **Compute Energy in milliJoules (mJ)**
  float energy_mJ = avgPower_mW * timeElapsed / 1000;

  printf("Execution time for gemm = %f ms\n", timeElapsed);
  printf("Average Power = %f mW\n", avgPower_mW);
  printf("Energy Consumption = %f mJ\n", energy_mJ);

  /* Free memory */
  cudaFree(x);
  cudaFree(y);
  cudaFree(z);

  return 0;
} /* main */
