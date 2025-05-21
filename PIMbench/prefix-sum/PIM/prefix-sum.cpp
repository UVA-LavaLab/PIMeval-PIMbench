// Test: C++ version of prefix sum
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

#include "util.h"
#include "libpimeval.h"

using namespace std;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t vectorLength;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./prefix-sum.out [options]"
          "\n"
          "\n    -l    input size (default=2048 elements)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing a vector (default=generates vector with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.vectorLength = 2048;
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
      p.vectorLength = strtoull(optarg, NULL, 0);
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

std::chrono::duration<double, std::milli> hostElapsedTime = std::chrono::duration<double, std::milli>::zero();

void prefixSum(uint64_t vectorLength, std::vector<int> &src, std::vector<int> &dst)
{
  PimObjId srcObj = pimAlloc(PIM_ALLOC_AUTO, vectorLength, PIM_INT32);
  if (srcObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimStatus status = pimCopyHostToDevice((void *)src.data(), srcObj);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimDeviceProperties deviceProp;
  status = pimGetDeviceProperties(&deviceProp);
  if (deviceProp.isHLayoutDevice) {
    PimObjId dstObj = pimAllocAssociated(srcObj, PIM_INT32);
    if (dstObj == -1)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
    status = pimPrefixSum(srcObj, dstObj);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
    dst.resize(vectorLength);
    status = pimCopyDeviceToHost(dstObj, (void *)dst.data());
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
    }
    pimFree(dstObj);
  } else {
    std::vector <int> tempVec = src;
    PimObjId maskObj = pimAllocAssociated(srcObj, PIM_INT32);
    if (maskObj == -1)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
    std::vector<int> maskVec (vectorLength, 0);
    for (uint64_t i = 0; (1 << i) < vectorLength; ++i) {
      auto start_cpu = std::chrono::high_resolution_clock::now();
      #pragma omp parallel for
      for (uint64_t j = 0; j < vectorLength; ++j) {
        if (j < (1 << i)) maskVec[j] = 0;
        else maskVec[j] = tempVec[j - (1 << i)];
      }
      auto stop_cpu = std::chrono::high_resolution_clock::now();
      hostElapsedTime += (stop_cpu - start_cpu);
      status = pimCopyHostToDevice((void *)maskVec.data(), maskObj);
      if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        return;
      }
      status = pimAdd(srcObj, maskObj, srcObj);
      status = pimCopyDeviceToHost(srcObj, tempVec.data());
      if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        return;
      }
    }
    dst = tempVec;
  }
  pimFree(srcObj);
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "Running Prefix Sum on PIM for vector length: " << params.vectorLength << "\n\n";
  std::vector<int> src(params.vectorLength, 1), dst;
  if (params.shouldVerify) {  
    if (params.inputFile == nullptr)
    {
      getVector(params.vectorLength, src);
    } else {
      std::cout << "Reading from input file is not implemented yet." << std::endl;
      return 1;
    }
  }
  if (!createDevice(params.configFile)) return 1;

  //TODO: Check if vector can fit in one iteration. Otherwise need to run addition in multiple iteration.
  prefixSum(params.vectorLength, src, dst);
  if (params.shouldVerify) {
    // verify result
    int sum = 0;
    for (unsigned i = 1; i < params.vectorLength; ++i)
    {
      sum += src[i];
      if (dst[i] != sum)
      {
        std::cout << "Wrong answer for prefix sum: " << dst[i] << " (expected " << sum << ")" << std::endl;
      }
    }
  }

  pimShowStats();

  return 0;
}
