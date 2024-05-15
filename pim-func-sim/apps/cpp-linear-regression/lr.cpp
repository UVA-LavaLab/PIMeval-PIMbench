// Test: C++ version of linear regression
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <iomanip>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "../util.h"
#include "libpimsim.h"

using namespace std;

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
          "\nUsage:  ./lr [options]"
          "\n"
          "\n    -l    input size (default=65536 elements)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing 2D matrix (default=generates matrix with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.dataSize = 65536;
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

void linearRegression(uint64_t dataSize, const std::vector<int> &X, const std::vector<int> &Y, int &SX, int &SY, int &SXX, int &SXY, int &SYY)
{
  unsigned bitsPerElement = sizeof(int) * 8;

  PimObjId srcObj1 = pimAlloc(PIM_ALLOC_V1, dataSize, bitsPerElement, PIM_INT32);
  if (srcObj1 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  PimObjId srcObj2 = pimAllocAssociated(PIM_ALLOC_V1, dataSize, bitsPerElement, srcObj1, PIM_INT32);
  if (srcObj2 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  PimObjId dstObj = pimAllocAssociated(PIM_ALLOC_V1, dataSize, bitsPerElement, srcObj1, PIM_INT32);
  if (dstObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimStatus status = pimCopyHostToDevice(PIM_COPY_V, (void *)X.data(), srcObj1);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimCopyHostToDevice(PIM_COPY_V, (void *)X.data(), srcObj2);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimMul(srcObj1, srcObj2, dstObj);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  SXX = pimRedSum(dstObj);

  status = pimCopyHostToDevice(PIM_COPY_V, (void *)Y.data(), srcObj2);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimMul(srcObj1, srcObj2, dstObj);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  SX = pimRedSum(srcObj1);
  SXX = pimRedSum(dstObj);

  status = pimCopyHostToDevice(PIM_COPY_V, (void *)Y.data(), srcObj1);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimMul(srcObj1, srcObj2, dstObj);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  SYY = pimRedSum(dstObj);
  SY = pimRedSum(srcObj1);

  pimFree(srcObj1);
  pimFree(srcObj2);
  pimFree(dstObj);
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "Data Size: " << params.dataSize << "\n";
  std::vector<int> dataPointsX, dataPointsY;
  if (params.inputFile == nullptr)
  {
    getVector(params.dataSize, dataPointsX);
    getVector(params.dataSize, dataPointsY);
  }
  else
  {
    // TODO: Read from files
  }

  if (!createDevice(params.configFile))
    return 1;

  int SX_device = 0, SY_device = 0, SXX_device = 0, SYY_device = 0, SXY_device = 0;
  std::vector<int> XX_device, XY_device;

  // TODO: Check if vector can fit in one iteration. Otherwise need to run addition in multiple iteration.
  linearRegression(params.dataSize, dataPointsX, dataPointsY, SX_device, SY_device, SXX_device, SXY_device, SYY_device);

  auto slope_device = (params.dataSize * SXY_device - SX_device * SY_device) / (params.dataSize * SXX_device - SX_device * SX_device);
  auto intercept_device = (SY_device - slope_device * SX_device) / params.dataSize;

  if (params.shouldVerify)
  {
    // verify result
    int SX = 0, SY = 0, SXX = 0, SYY = 0, SXY = 0;
#pragma omp parallel for reduction(+ : SX, SXX, SY, SYY, SXY)
    for (uint64_t i = 0; i < params.dataSize; ++i)
    {
      SX += dataPointsX[i];
      SXX += dataPointsX[i] * dataPointsX[i];
      SY += dataPointsY[i];
      SYY += dataPointsY[i] * dataPointsY[i];
      SXY += dataPointsX[i] * dataPointsY[i];
    }
    // Calculate slope and intercept
    auto slope = (params.dataSize * SXY - SX * SY) / (params.dataSize * SXX - SX * SX);
    auto intercept = (SY - slope * SX) / params.dataSize;
    if (intercept != intercept_device) {
      cout << "Wrong answer. Expected: " << intercept << " . Calculated: " << intercept_device << "\n";
    }
  }

  pimShowStats();

  return 0;
}
