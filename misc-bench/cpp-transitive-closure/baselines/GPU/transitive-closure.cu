/* File:     transitive-closure.cu
 * Purpose:  Implement transitive closure on a gpu using Thrust
 *
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <fstream>

#include "../../../utilBaselines.h"
#include "../../transitive-closure.hpp"

// Thrust error-checking macro
#define THRUST_CHECK() \
do \
{ \
  cudaError_t err = cudaDeviceSynchronize(); \
  if (err != cudaSuccess) \
  { \
    fprintf(stderr, "Thrust error in file '%s' in line %i : %s.\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
} while(0)

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t dataSize;
  int sparsityRate;
  char *configFile;
  std::string inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./transitive-closure.out [options]"
          "\n"
          "\n    -l    number of vertices for a generated dataset (default=2^8 vertices)"
          "\n    -r    sparsity rate n for a generated dataset where 0 < n < 100. (default=50 percent)"
          "\n    -c    dramsim config file"
          "\n    -i    specify a .csv input file containing the number of vertices followed by a 2D array, used for verification with CPU/GPU benchmarks"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.dataSize = 256; 
  p.sparsityRate = 50;
  p.configFile = nullptr;
  p.inputFile = "";
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:l:r:c:i:v:")) >= 0)
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
    case 'r':
      p.sparsityRate = strtoull(optarg, NULL, 0);
      break;  
    case 'c':
      p.configFile = optarg;
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

struct floyd_warshall_function
{
  int *d_adjMatrix;
  int k, numVertices;

  floyd_warshall_function(int *_d_adjMatrix, int _k, int _numVertices) : d_adjMatrix(_d_adjMatrix), k(_k), numVertices(_numVertices) {}

  __device__ void operator()(int idx)
  {
    int i = idx / numVertices;
    int j = idx % numVertices;
    int ik = d_adjMatrix[i * numVertices + k];
    int kj = d_adjMatrix[k * numVertices + j];
    int &ij = d_adjMatrix[i * numVertices + j];
    if (ij > ik + kj)
    {
      ij = ik + kj;
    }
  }
};

void transitiveClosure(thrust::device_vector<int> &d_adjMatrix, int numVertices)
{
  int *d_adjMatrix_ptr = thrust::raw_pointer_cast(d_adjMatrix.data());

  for (int k = 0; k < numVertices; ++k)
  {
    thrust::counting_iterator<int> indices(0);
    THRUST_CHECK();

    floyd_warshall_function fw(d_adjMatrix_ptr, k, numVertices);

    thrust::for_each(indices, indices + numVertices * numVertices, fw);
    THRUST_CHECK();
  }
}

int main(int argc, char *argv[]) 
{      
  struct Params params = getInputParams(argc, argv);
  
  std::vector<int> inputList;
  int numVertices;
  std::vector<std::vector<int>> generatedMatrix;

  if (params.inputFile.compare("") == 0)
  {
    numVertices = params.dataSize;
    getMatrix(numVertices, numVertices, generatedMatrix);
    srand(time(NULL));
    for (int i = 0; i < numVertices; ++i)
    {
      for (int j = 0; j < numVertices; ++j)
      {
        int sparsityChance = rand() % 100 + 1;
        if ((generatedMatrix[i][j] == 0 || sparsityChance <= params.sparsityRate) && (i != j))
        {
          generatedMatrix[i][j] = MAX_EDGE_VALUE; 
        }
        else if (i == j)
        {
          generatedMatrix[i][j] = 0;
        }
      }
    }

    for (int i = 0; i < numVertices; ++i)
    {
      for (int j = 0; j < numVertices; ++j)
      {
        inputList.push_back(generatedMatrix[j][i]);
      }
    }

    std::cout << "Sparsity rate: " << params.sparsityRate << "%" << std::endl;
  }
  else 
  {
    if (!readCSV(params.inputFile, inputList, numVertices))
    {
      std::cout << "Failed to read input file or does not exist" << std::endl;
      return 1;
    }

    std::cout << "Input file: '" << params.inputFile << "'" << std::endl;
  }

  std::cout << "Number of vertices: " << numVertices << std::endl;

  thrust::host_vector<int> h_adjList(inputList.begin(), inputList.end());
  thrust::device_vector<int> d_adjList = h_adjList;
  THRUST_CHECK();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float timeElapsed = 0;

  // Start timer
  cudaEventRecord(start, 0);

  transitiveClosure(d_adjList, numVertices);

  // End timer
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timeElapsed, start, stop);

  printf("Execution time = %f ms\n", timeElapsed);
  
  h_adjList = d_adjList;


  for (int i = 0; i < numVertices; ++i)
  {
    h_adjList[(i * numVertices) + i] = 0;
  }

  if (params.shouldVerify)
  {
    std::vector<std::vector<int>> cpuAdjMatrix (numVertices, std::vector<int>(numVertices));

    for (int i = 0; i < numVertices; ++i)
    {
      for (int j = 0; j < numVertices; ++j)
      {
        cpuAdjMatrix[i][j] = inputList[(i * numVertices) + j];
      }
    }

    for (int k = 0; k < numVertices; ++k) 
    {
      for (int i = 0; i < numVertices; ++i) 
      {
        for (int j = 0; j < numVertices; ++j) 
        {
          if (cpuAdjMatrix[i][j] > (cpuAdjMatrix[i][k] + cpuAdjMatrix[k][j]))
          {
            cpuAdjMatrix[i][j] = cpuAdjMatrix[i][k] + cpuAdjMatrix[k][j];
          }
        }
      }
    }

    int errorFlag = 0;
    for (int i = 0; i < numVertices; ++i)
    {
      for (int j = 0; j < numVertices; ++j)
      {
        if (cpuAdjMatrix[i][j] != h_adjList[(i * numVertices) + j])
        {
          std::cout << "Incorrect result at index [" << i << "," << j << "] = " << h_adjList[(i * numVertices) + j] << " (expected " << cpuAdjMatrix[i][j] << ")" << endl;
          errorFlag = 1; 
        }
      }
    }

    if (!errorFlag) 
    {
      std::cout << "Correct!" << endl;
    }
  }

  return 0;
}
