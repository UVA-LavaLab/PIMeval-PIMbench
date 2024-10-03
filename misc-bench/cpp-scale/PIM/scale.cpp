// Test: C++ version of scale
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <iomanip>
#include <cassert>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "../../util.h"
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
          "\nUsage:  ./scale.out [options]"
          "\n"
          "\n    -l    input size (default=2048 elements)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing one vector (default=generates vector with random numbers)"
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

void scale(uint64_t vectorLength, const std::vector<int> &src_host, int A, std::vector<int> &dst_host)
{
  PimObjId src_pim = pimAlloc(PIM_ALLOC_AUTO, vectorLength, PIM_INT32);
  assert(src_pim != -1);

  PimObjId dst_pim = pimAllocAssociated(src_pim, PIM_INT32);
  assert(dst_pim != -1);

  PimStatus status = pimCopyHostToDevice((void *)src_host.data(), src_pim);
  assert (status == PIM_OK);

  status = pimMulScalar(src_pim, dst_pim, A);
  assert (status == PIM_OK);

  dst_host.resize(vectorLength);
  status = pimCopyDeviceToHost(dst_pim, (void *)dst_host.data());
  assert (status == PIM_OK);

  pimFree(src_pim);
  pimFree(dst_pim);
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "Running PIM scale for vector length: " << params.vectorLength << "\n";
  std::vector<int> X, Y_device;
  if (params.inputFile == nullptr)
  {
    getVector(params.vectorLength, X);
  } 
  else 
  {
    std::cout << "Reading from input file is not implemented yet." << std::endl;
    return 1;
  }
  
  if (!createDevice(params.configFile))
  {
    return 1;
  }

  //TODO: Check if vector can fit in one iteration. Otherwise need to run in multiple iteration.
  int A = rand() % 50;
  scale(params.vectorLength, X, A, Y_device);

  if (params.shouldVerify) 
  {
    // verify result
    #pragma omp parallel for
    bool is_correct = true;
    for (unsigned i = 0; i < params.vectorLength; ++i)
    {
      int result = A * X[i];
      if (Y_device[i] != result)
      {
        std::cout << "Wrong answer: " << Y_device[i] << " (expected " << result << ")" << std::endl;
        is_correct = false;
      }
    }
    if(is_correct) {
      std::cout << "Correct for scale!" << std::endl;
    }
  }

  pimShowStats();

  return 0;
}
