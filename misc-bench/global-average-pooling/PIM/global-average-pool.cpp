// Test: C++ version of global average pooling
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include <iostream>
#include <vector>
#include <getopt.h>
#include <cmath>
#include <cassert>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "util.h"
#include "libpimeval.h"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t row, column, dim;
  char *dramConfigFile;
  char *imageMatrixFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./global-average-pool.out [options]"
          "\n"
          "\n    -r    row (default=224)"
          "\n    -c    column (default=224)"
          "\n    -d    dimension (default=64)"
          "\n    -f    dramsim config file"
          "\n    -i    input image file containing matrices (default=generates matrix with random numbers)" 
          "\n    -v    should verify result with CPU"       
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.row = 224;
  p.column = 224;
  p.dim = 64;
  p.dramConfigFile = nullptr;
  p.imageMatrixFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:r:c:d:f:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'r':
      p.row = atoi(optarg);
      break;
    case 'c':
      p.column = atoi(optarg);
      break;
    case 'd':
      p.dim = atoi(optarg);
      break;
    case 'f':
      p.dramConfigFile = optarg;
      break;
    case 'i':
      p.imageMatrixFile = optarg;
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

void decomposeMatrix(const std::vector<std::vector<int>> &inputMatrix, std::vector<int> &decompMatrix)
{
  for (uint64_t i = 0; i < inputMatrix.size(); ++i)
  {
    for (uint64_t j = 0; j < inputMatrix[0].size(); ++j)
    {
      decompMatrix.push_back(inputMatrix[i][j]);
    }
  }
}

void globalAveragePool(const std::vector<std::vector<int>> &inputMatrix, std::vector<int> &outputMatrix)
{
  if (inputMatrix.empty())
  {
    std::cerr << "Abort: input matrix is empty" << std::endl;
    exit(1);
  }

  int numDepth = inputMatrix.size();
  int kernelSize = inputMatrix[0].size();

  PimObjId obj1 = pimAlloc(PIM_ALLOC_AUTO, kernelSize, PIM_INT32);
  assert(obj1 != -1);

  PimStatus status;
  for (int i = 0; i < numDepth; i++)
  {
    status = pimCopyHostToDevice((void *) inputMatrix[i].data(), obj1);
    assert(status == PIM_OK);

    status = pimRedSum(obj1, static_cast<void*>(&outputMatrix[i]));
    assert(status == PIM_OK);
  }

  PimObjId obj2 = pimAlloc(PIM_ALLOC_AUTO, numDepth, PIM_INT32);
  assert(obj2 != -1);

  status = pimCopyHostToDevice((void *) outputMatrix.data(), obj2);
  assert(status == PIM_OK);

  status = pimDivScalar(obj2, obj2, kernelSize);
  assert(status == PIM_OK);

  status = pimCopyDeviceToHost(obj2, outputMatrix.data());
  assert(status == PIM_OK);

  pimFree(obj1);
  pimFree(obj2);
}

// Function to perform global average pooling on CPU with configurable kernel size 
void verifyWithCPU(std::vector<std::vector<std::vector<int>>> &inputMatrix, std::vector<int> &PIMResult)
{
  int numDepth = inputMatrix.size();
  int numRows = inputMatrix[0].size();
  int numCols = inputMatrix[0][0].size();

  // Initialize the output matrix with zeros
  std::vector<int> outputMatrix(numDepth, 0);

  int kernelSize = numRows * numCols;    
  int mismatch_counter = 0;

  // Perform global average pooling with the specified kernel size
  for (int d = 0; d < numDepth; ++d) {
    int sumVal = 0;
    for (int i = 0; i < numRows; ++i) {
      for (int j = 0; j < numCols; ++j) {
        sumVal += inputMatrix[d][i][j];
      }
    }
    outputMatrix[d] = sumVal / kernelSize;
    if (outputMatrix[d] != PIMResult[d]) {
      std::cout << "Mismatch between PIM and CPU results at depth layer: " << d << std::endl;
      mismatch_counter += 1;
    }
  }

  if (!mismatch_counter) {
    std::cout << "Success: PIM results match with CPU results" << std::endl; 
  } 
  else {
    std::cerr << "Failure: PIM results do not match with CPU results" << std::endl;
    exit(1);
  }
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);

  std::vector<std::vector<std::vector<int>>> inputMatrix;
  inputMatrix.resize(params.dim, std::vector<std::vector<int>>(params.row, std::vector<int>(params.column)));

  if (params.imageMatrixFile == nullptr)
  {
    for (auto &mat : inputMatrix)
    {
      getMatrix(params.row, params.column, 0, mat);
    }
  }
  else
  {
    std::cerr << "Reading from the input file is not implemented yet for the input matrix" << std::endl;
    exit(1);
  }

  if (!createDevice(params.dramConfigFile))
    exit(1);

  std::vector<int> resultMatrix (params.dim);
  std::vector<std::vector<int>> decomposedMatrix (params.dim);
  
  for (uint64_t i = 0; i < params.dim; ++i)
  {
    decomposeMatrix(inputMatrix[i], decomposedMatrix[i]);
  }

  globalAveragePool(decomposedMatrix, resultMatrix);

  if (params.shouldVerify)
  {
    verifyWithCPU(inputMatrix, resultMatrix);
  }
  
  pimShowStats();

  return 0;
}
