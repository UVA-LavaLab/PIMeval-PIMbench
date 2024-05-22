// Test: C++ version of vector addition
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
          "\nUsage:  ./add [options]"
          "\n"
          "\n    -l    input size (default=8M elements)"
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

void axpy(uint64_t vectorLength, const std::vector<int> &X, const std::vector<int> &Y, int A, std::vector<int> &dst)
{
  unsigned bitsPerElement = sizeof(int) * 8;
  PimObjId obj1 = pimAlloc(PIM_ALLOC_V1, vectorLength, bitsPerElement, PIM_INT32);
  if (obj1 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimObjId obj2 = pimAllocAssociated(PIM_ALLOC_V1, vectorLength, bitsPerElement, obj1, PIM_INT32);
  if (obj1 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimStatus status = pimCopyHostToDevice(PIM_COPY_V, (void *)X.data(), obj1);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimBroadCast(PIM_COPY_V, obj2, A);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimMul(obj1, obj2, obj1);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimCopyHostToDevice(PIM_COPY_V, (void *)Y.data(), obj2);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimAdd(obj1, obj2, obj2);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  dst.resize(vectorLength);
  status = pimCopyDeviceToHost(PIM_COPY_V, obj2, (void *)dst.data());
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
  }

  pimFree(obj1);
  pimFree(obj2);
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "Vector length: " << params.vectorLength << "\n";
  std::vector<int> X, Y, Y_device;
  if (params.inputFile == nullptr)
  {
    getVector(params.vectorLength, X);
    getVector(params.vectorLength, Y);
  } else {
    //TODO: Read from files
  }
  if (!createDevice(params.configFile)) return 1;
  //TODO: Check if vector can fit in one iteration. Otherwise need to run in multiple iteration.
  int A = rand() % 50;
  axpy(params.vectorLength, X, Y, A, Y_device);

  if (params.shouldVerify) {
    // verify result
    #pragma omp parallel for
    for (unsigned i = 0; i < params.vectorLength; ++i)
    {
      int result = A * X[i] + Y[i];
      if (Y_device[i] != result)
      {
        std::cout << "Wrong answer: " << Y_device[i] << " (expected " << result << ")" << std::endl;
      }
    }
  }

  pimShowStats();

  return 0;
}
