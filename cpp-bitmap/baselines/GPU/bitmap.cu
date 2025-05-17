/* File:     bitmap.cu
 * Purpose:  Implement bitmap on a gpu using Thrust
 *
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fcntl.h>
#include <vector>
#include <random>

#include "../../../../PIMbench/utilBaselines.h"

// Static definition, should be made dynamic in future work
#define numBitmapIndices 8

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
          "\nUsage:  ./bitmap.out [options]"
          "\n"
          "\n    -l    number of data entries (default=2048 elements)"
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

__global__ void bitmapKernel(uint8_t* d_database, uint8_t* d_result, uint8_t* d_validEntries, uint64_t numDatabaseEntries) 
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < numDatabaseEntries) 
  {
    for (int i = 0; i < numBitmapIndices; ++i) 
    {
      d_result[idx + i * numDatabaseEntries] = (d_database[idx] == d_validEntries[i]);
    }
  }
}

int main(int argc, char *argv[]) 
{      
  struct Params params = getInputParams(argc, argv);
  std::vector<uint8_t> h_database;
  uint64_t numDatabaseEntries = params.dataSize;
  std::vector<uint8_t> h_validEntries;

  std::vector<uint8_t> h_result (numDatabaseEntries * numBitmapIndices);
  
  if (params.inputFile == nullptr)
  {
    h_database.resize(numDatabaseEntries);

    // Assuming 8 unique bitmap indicies, no h_database entries for 0x00
    h_validEntries = {
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80
    };
    for (uint64_t i = 0; i < numDatabaseEntries; ++i)
    {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> dis(0, h_validEntries.size() - 1);
      
      int idx = dis(gen);

      h_database[i] = h_validEntries[idx];
    }
  }
  else
  {
    std::cerr << "Reading from the input file is not implemented yet for bitmap" << std::endl;
    return 1;
  }

  printf("Performing bitmap GPU baseline with %lu data points and 8 unique bitmap indices\n", numDatabaseEntries);
  
  uint8_t *d_database, *d_result, *d_validEntries;
    
  cudaError_t errorCode = cudaMalloc(&d_database, numDatabaseEntries * sizeof(uint8_t));
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }
  errorCode = cudaMalloc(&d_result, numDatabaseEntries * numBitmapIndices * sizeof(uint8_t));
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }
  errorCode = cudaMalloc(&d_validEntries, numBitmapIndices * sizeof(uint8_t));
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  errorCode = cudaMemcpy(d_database, h_database.data(), numDatabaseEntries * sizeof(uint8_t), cudaMemcpyHostToDevice);
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }
  errorCode = cudaMemcpy(d_validEntries, h_validEntries.data(), numBitmapIndices * sizeof(uint8_t), cudaMemcpyHostToDevice);
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  int threadsPerBlock = 256;
  int blocksPerGrid = (numDatabaseEntries + threadsPerBlock - 1) / threadsPerBlock;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float timeElapsed = 0;

  // Start timer
  cudaEventRecord(start, 0);

  bitmapKernel<<<blocksPerGrid, threadsPerBlock>>>(d_database, d_result, d_validEntries, numDatabaseEntries);

  // End timer
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timeElapsed, start, stop);

  printf("Execution time = %f ms\n", timeElapsed);

  errorCode = cudaMemcpy(h_result.data(), d_result, numDatabaseEntries * numBitmapIndices * sizeof(uint8_t), cudaMemcpyDeviceToHost);
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  if (params.shouldVerify)
  {  
    int errorFlag = 0;
    std::vector<std::vector<uint8_t>> baselineResult(numBitmapIndices, std::vector<uint8_t> (numDatabaseEntries));

    for (int i = 0; i < numBitmapIndices; ++i) 
    {   
      
      for (uint64_t j = 0; j < numDatabaseEntries; ++j)
      {
        baselineResult[i][j] = (h_database[j] == h_validEntries[i]);
        if (baselineResult[i][j] != h_result[(i * numDatabaseEntries) + j]) 
        {
          std::cout << "Wrong answer at index [" << i << "," << j << "] | Wrong PIM answer = " << static_cast<int> (h_result[(i * numBitmapIndices) + j]) << " (GPU baseline expected = " << static_cast<int> (baselineResult[i][j]) << ")" << std::endl;
          errorFlag = 1;
        }
      }
    }
    
    if (!errorFlag)
    {
      std::cout << "Correct!" << std::endl;
    }
  }

  cudaFree(d_database);
  cudaFree(d_result);
  cudaFree(d_validEntries);
   
  return 0;
}
