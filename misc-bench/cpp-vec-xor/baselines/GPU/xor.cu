/* File:     xor.cu
 * Purpose:  Implement xor on a gpu using Thrust
 *
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <vector>

#include "../../../../PIMbench/utilBaselines.h"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t dataSize;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./xor.out [options]"
          "\n"
          "\n    -l    vector length (default=2048 elements)"
          "\n    -i    input file containing two vectors (default=generates vectors with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.dataSize = 2048;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:l:i:v:")) >= 0)
  {
    switch (opt)
    {
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

__global__ void xorKernel(uint64_t vectorLength, const int *src1, const int *src2, int *dst)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < vectorLength)
  {
    dst[idx] = src1[idx] ^ src2[idx];
  }
}

int main(int argc, char *argv[]) 
{      
  struct Params params = getInputParams(argc, argv);
  std::vector<int> h_src1, h_src2, h_dst;
  uint64_t vectorLength = params.dataSize;
  
  if (params.inputFile == nullptr)
  {
    h_src1.resize(vectorLength);
    h_src2.resize(vectorLength);
    h_dst.resize(vectorLength);
    
    getVector(vectorLength, h_src1);
    getVector(vectorLength, h_src2);
  }
  else
  {
    std::cerr << "Reading from the input file is not implemented yet for the input matrix" << std::endl;
    return 1;
  }

  printf("Performing XOR GPU baseline with %lu data points\n", vectorLength);

  int *d_src1, *d_src2, *d_dst;

  cudaError_t errorCode = cudaMalloc(&d_src1, vectorLength * sizeof(int)); 
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }
  errorCode = cudaMalloc(&d_src2, vectorLength * sizeof(int)); 
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }
  errorCode = cudaMalloc(&d_dst, vectorLength * sizeof(int)); 
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  errorCode = cudaMemcpy(d_src1, h_src1.data(), vectorLength * sizeof(int), cudaMemcpyHostToDevice);
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }
  errorCode = cudaMemcpy(d_src2, h_src2.data(), vectorLength * sizeof(int), cudaMemcpyHostToDevice);
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  int threadsPerBlock = 256;
  int blocksPerGrid = (vectorLength + threadsPerBlock - 1) / threadsPerBlock;
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float timeElapsed = 0;

  // Start timer
  cudaEventRecord(start, 0);

  xorKernel<<<blocksPerGrid, threadsPerBlock>>>(vectorLength, d_src1, d_src2, d_dst);

  // End timer
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timeElapsed, start, stop);

  printf("Execution time = %f ms\n", timeElapsed);

  errorCode = cudaMemcpy(h_dst.data(), d_dst, vectorLength * sizeof(int), cudaMemcpyDeviceToHost);
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  if (params.shouldVerify)
  {  
    int errorFlag = 0;
    for (uint64_t i = 0; i < vectorLength; ++i) 
    {     
      if ((h_src1[i] ^ h_src2[i]) != h_dst[i])
      {
        std::cout << "Wrong answer at index " << i << " | Wrong CUDA answer = " << h_dst[i] << " (GPU expected = " << (h_src1[i] ^ h_src2[i]) << ")" << std::endl;
        errorFlag = 1;
      }
    }
    if (!errorFlag)
    {
      std::cout << "Correct!" << std::endl;
    }
  }

  cudaFree(d_src1);
  cudaFree(d_src2);
  cudaFree(d_dst);
   
  return 0;
}