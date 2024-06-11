// Test: C++ version of vector multiplication
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
  uint64_t vectorLength;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./mul [options]"
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

void vectorRed(uint64_t vectorLength, std::vector<int> src, int &reductionValue)
{
  unsigned bitsPerElement = sizeof(int) * 8;
  PimObjId srcObj = pimAlloc(PIM_ALLOC_AUTO, vectorLength, bitsPerElement, PIM_INT32);
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

  status = pimRedSum(srcObj, &reductionValue);
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
  int reductionValue = 0;
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
