// Test: C++ version of dot product
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

#include "../util.h"
#include "libpimeval.h"

using namespace std;

std::chrono::duration<double, std::milli> hostElapsedTime = std::chrono::duration<double, std::milli>::zero();

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
          "\nUsage:  ./dot-prod.out [options]"
          "\n"
          "\n    -l    input size (default=65536 elements)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing two vectors (default=generates vector with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.vectorLength = 65536;
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

void dotProduct(uint64_t vectorLength, std::vector<int> &src1, std::vector<int> &src2, int64_t &result)
{
  PimObjId srcObj1 = pimAlloc(PIM_ALLOC_AUTO, vectorLength, PIM_INT32);
  if (srcObj1 == -1)
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

  PimStatus status = pimCopyHostToDevice((void *)src1.data(), srcObj1);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimCopyHostToDevice((void *)src2.data(), srcObj2);
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

  std::vector<int> dst(vectorLength);
  status = pimCopyDeviceToHost(srcObj1, (void *)dst.data());
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimRedSumInt(srcObj1, &result);
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
  std::cout << "Vector length: " << params.vectorLength << "\n";
  std::vector<int> src1, src2;
  if (params.inputFile == nullptr)
  {
    getVector(params.vectorLength, src1);
    getVector(params.vectorLength, src2);
  }
  else
  {
    std::cout << "Reading from input file is not implemented yet." << std::endl;
    return 0;
  }
  if (!createDevice(params.configFile))
    return 1;
  // TODO: Check if vector can fit in one iteration. Otherwise need to run addition in multiple iteration.
  int64_t deviceValue = 0;
  dotProduct(params.vectorLength, src1, src2, deviceValue);
  if (params.shouldVerify)
  {
    // verify result
    int sum = 0;
    for (unsigned i = 0; i < params.vectorLength; ++i)
    {
      sum += src1[i] * src2[i];
    }
    if (deviceValue != sum)
    {
      std::cout << "Wrong answer for dot product: " << deviceValue << " (expected " << sum << ")" << std::endl;
    }
    else
    {
      std::cout << "Correct answer for dot product!" << std::endl;
    }
  }

  pimShowStats();

  cout << "Host elapsed time: " << std::fixed << std::setprecision(3) << hostElapsedTime.count() << " ms." << endl;
  return 0;
}
