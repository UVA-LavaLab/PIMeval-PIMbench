// Test: C++ version of linear regression
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <iomanip>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "../../util.h"
#include "libpimeval.h"

using namespace std;

std::chrono::duration<double, std::milli> hostElapsedTime = std::chrono::duration<double, std::milli>::zero();

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t dataSize;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./lr.out [options]"
          "\n"
          "\n    -l    input size (default=2048 elements)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing 2D matrix (default=generates matrix with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.dataSize = 2048;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:l:c:i:v:")) >= 0)
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

void linearRegression(uint64_t dataSize, const std::vector<int> &X, const std::vector<int> &Y, int64_t &SX, int64_t &SY, int64_t &SXX, int64_t &SXY)
{
  PimObjId srcObj1 = pimAlloc(PIM_ALLOC_AUTO, dataSize, PIM_INT32);
  if (srcObj1 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimStatus status = pimCopyHostToDevice((void *)X.data(), srcObj1);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimRedSumInt(srcObj1, &SX);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimObjId srcObj2 = pimAllocAssociated(srcObj1, PIM_INT32);
  if (srcObj2 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimMul(srcObj1, srcObj1, srcObj2);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimRedSumInt(srcObj2, &SXX);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimCopyHostToDevice((void *)Y.data(), srcObj2);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimRedSumInt(srcObj2, &SY);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimMul(srcObj1, srcObj2, srcObj1);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimRedSumInt(srcObj1, &SXY);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  pimFree(srcObj1);
  pimFree(srcObj2);
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "Running Linear Regression on PIM for data size: " << params.dataSize << "\n";
  std::vector<int> dataPointsX, dataPointsY;
  if (params.inputFile == nullptr)
  {
    getVector(params.dataSize, dataPointsX);
    getVector(params.dataSize, dataPointsY);
  }
  else
  {
    std::cout << "Reading from input file is not implemented yet." << std::endl;
    return 1;
  }

  if (!createDevice(params.configFile))
    return 1;

  int64_t SX_device = 0, SY_device = 0, SXX_device = 0, SXY_device = 0;

  // TODO: Check if vector can fit in one iteration. Otherwise need to run addition in multiple iteration.
  linearRegression(params.dataSize, dataPointsX, dataPointsY, SX_device, SY_device, SXX_device, SXY_device);

  auto start = std::chrono::high_resolution_clock::now();
  auto slope_device = (double)(params.dataSize * SXY_device - SX_device * SY_device) / (params.dataSize * SXX_device - SX_device * SX_device);
  auto intercept_device = (double)(SY_device - slope_device * SX_device) / params.dataSize;
  auto end = std::chrono::high_resolution_clock::now();
  hostElapsedTime += (end - start);

  if (params.shouldVerify)
  {
    // verify result
    int64_t SX = 0, SY = 0, SXX = 0, SXY = 0;
    
    #pragma omp parallel for reduction(+ : SX, SXX, SY, SXY)
    for (uint64_t i = 0; i < params.dataSize; ++i)
    {
      SX += dataPointsX[i];
      SXX += dataPointsX[i] * dataPointsX[i];
      SY += dataPointsY[i];
      SXY += dataPointsX[i] * dataPointsY[i];
    }

    // Calculate slope and intercept
    auto slope = (double)(params.dataSize * SXY - SX * SY) / (params.dataSize * SXX - SX * SX);
    auto intercept = (double)(SY - slope * SX) / params.dataSize;
    if (intercept != intercept_device)
    {
      cout << "\nWrong answer. Expected: " << intercept << " . Calculated: " << intercept_device << "\n";
    }
    else
    {
      cout << "\n\nCorrect Answer!\n\n";
    }
  }

  pimShowStats();
  cout << "Host elapsed time: " << std::fixed << std::setprecision(3) << hostElapsedTime.count() << " ms." << endl;

  return 0;
}
