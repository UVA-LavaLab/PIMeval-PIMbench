// Test: C++ version of vector multiplication
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
          "\nUsage:  ./vec-red.out [options]"
          "\n"
          "\n    -l    input size (default=65536 elements)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing two vectors (default=generates vector with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n    -h    shows help text"
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

void vectorRed(uint64_t vectorLength, std::vector<int> src, int64_t &reductionValue)
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

  status = pimRedSumInt(srcObj, &reductionValue);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  pimFree(srcObj);
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "Vector length: " << params.vectorLength << "\n";
  std::vector<int> src;
  if (params.inputFile == nullptr)
  {
    getVector(params.vectorLength, src);
  }
  else
  {
    // TODO: Read from files
  }
  if (!createDevice(params.configFile))
    return 1;
  // TODO: Check if vector can fit in one iteration. Otherwise need to run addition in multiple iteration.
  int64_t reductionValue = 0;
  vectorRed(params.vectorLength, src, reductionValue);
  if (params.shouldVerify)
  {
    int sum = 0;
// verify result
#pragma omp parallel for
    for (unsigned i = 0; i < params.vectorLength; ++i)
    {
      sum += src[i];
    }

    if (reductionValue != sum)
    {
      std::cout << "Wrong answer for reduction: "<< reductionValue << " (expected " << sum << ")" << std::endl;
    }
    else {
      std::cout << "Correct Answer!" << std::endl;
    }
  }

  pimShowStats();

  return 0;
}
