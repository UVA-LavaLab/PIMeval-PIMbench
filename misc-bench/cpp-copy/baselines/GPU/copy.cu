/* File:     copy.cu
 * Purpose:  Implement vector copying on a GPU
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <stdint.h>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <iostream>
#include <vector>

#include "utilBaselines.h"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t vectorLength;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./copy.out [options]"
          "\n"
          "\n    -l    input size (default=2048 elements)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params input_params(int argc, char **argv)
{
  struct Params p;
  p.vectorLength = 2048;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:l:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'l':
      p.vectorLength = strtoull(optarg, NULL, 0);
      break;
    case 'v':
      p.shouldVerify = (*optarg == 't') ? true : false;
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

/**
 * @brief gpu copy kernel
 */
__global__ void copy(int* gpu_x, int* gpu_y, size_t sz) {
  size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx < sz) {
    gpu_y[idx] = gpu_x[idx];
  }
}

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char **argv)
{
  cudaGetLastError();

  struct Params params = input_params(argc, argv);

  uint64_t vectorSize = params.vectorLength;

  std::vector<int> X;
  std::vector<int> Y;

  getVector(vectorSize, X);
  Y.resize(X.size());

  int* gpu_x;
  int* gpu_y;

  size_t vector_sz = sizeof(int)*vectorSize;

  cudaError_t cuda_error;
  cuda_error = cudaMalloc((void**)&gpu_x, vector_sz);

  if(cuda_error != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << "\n";
    exit(1);
  }

  cuda_error = cudaMemcpy(gpu_x, X.data(), vector_sz, cudaMemcpyHostToDevice);

  if(cuda_error != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << "\n";
    exit(1);
  }

  cuda_error = cudaMalloc((void**)&gpu_y, vector_sz);

  if(cuda_error != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << "\n";
    exit(1);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float timeElapsed = 0;

  cudaEventRecord(start, 0);

  copy<<<(vectorSize + 1023) / 1024, 1024>>>(gpu_x, gpu_y, vectorSize);
  
  cuda_error = cudaGetLastError();
  if (cuda_error != cudaSuccess)
  {
      std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << "\n";
      exit(1);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timeElapsed, start, stop);

  printf("Execution time of copy = %f ms\n", timeElapsed);

  cuda_error = cudaMemcpy(Y.data(), gpu_y, vector_sz, cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess)
  {
      cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << "\n";
      exit(1);
  }
  cudaFree(gpu_x);
  cudaFree(gpu_y);

  if (params.shouldVerify) 
  {

    // verify result
    #pragma omp parallel for
    for (unsigned i = 0; i < params.vectorLength; ++i)
    {
      int result = X[i];
      if (Y[i] != result)
      {
        std::cout << "Wrong answer: " << Y[i] << " (expected " << result << ")" << std::endl;
        exit(-1);
      }
    }
    std::cout << "Correct for copy!" << std::endl;
  }

  return 0;
}