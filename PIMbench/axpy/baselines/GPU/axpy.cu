/* File:     axpy.cu
 * Purpose:  Implement axpy on a gpu using cuda
 *
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cublas_v2.h>
#include <nvml.h>

#include "utilBaselines.h"

using namespace std;

vector<int32_t> A;
vector<int32_t> B;
vector<int32_t> C;

#define TOLERANCE	200.0f

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t vector_size;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./axpy.out [options]"
          "\n"
          "\n    -i    vector size (default=65536)"
          "\n");
}

struct Params input_params(int argc, char **argv)
{
  struct Params p;
  p.vector_size = 65536;

  int opt;
  while ((opt = getopt(argc, argv, ":h:i:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'i':
      p.vector_size = atoll(optarg);
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

int main(int argc, char *argv[])
{
  struct Params p = input_params(argc, argv);

  uint64_t vector_size = p.vector_size;
  getVector(vector_size, A);
  getVector(vector_size, B);
  const float a = rand() % 5;

  float *x, *y;
  std::cout << "Running axpy on vector of size: " << vector_size << std::endl;
  cudaError_t errorCode;

  errorCode = cudaMalloc((void **)&x, vector_size * sizeof(int32_t));
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }
  errorCode = cudaMalloc((void **)&y, vector_size * sizeof(int32_t));
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  errorCode = cudaMemcpy(x, A.data(), vector_size * sizeof(float), cudaMemcpyHostToDevice);
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  errorCode = cudaMemcpy(y, B.data(), vector_size * sizeof(float), cudaMemcpyHostToDevice);
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS)
  {
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
  status = cublasSaxpy(handle, vector_size, &a, x, 1, y, 1);

  if (status != CUBLAS_STATUS_SUCCESS)
  {
    std::cerr << "CUBLAS SGEMV failed\n";
    exit(1);
  }

  // Check for kernel launch errors
  errorCode = cudaGetLastError();
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

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

  printf("Execution time for AXPY = %f ms\n", timeElapsed);
  printf("Average Power = %f mW\n", avgPower_mW);
  printf("Energy Consumption = %f mJ\n", energy_mJ);

  vector<int32_t> C(vector_size);
  errorCode = cudaMemcpy(C.data(), y, vector_size * sizeof(int32_t), cudaMemcpyDeviceToHost);
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error Copy from device to host: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  cout.precision(0);
  for (size_t i = 0; i < A.size(); ++i)
  {
    int32_t sum = a * B[i] + A[i];
    if (abs(C[i] - sum) > TOLERANCE)
    {
      cout << fixed << "AXPY failed at index: " << i << "\t" << C[i] << "\t" << sum << endl;
      break;
    }
  }

  /* Free memory */
  cublasDestroy(handle);
  cudaFree(x);
  cudaFree(y);

  return 0;
} /* main */
