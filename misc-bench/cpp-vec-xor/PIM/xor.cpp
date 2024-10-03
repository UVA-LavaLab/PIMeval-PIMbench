// Test: C++ version of XOR
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
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
  uint64_t dataSize;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./xor.out [options]"
          "\n"
          "\n    -l    vector length (default=2048 elements)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing two vectors (default=generates vectors with random numbers)"
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

void XOR(uint64_t vectorLength, const std::vector<int> src1, const std::vector<int> src2, std::vector<int> &dst)
{
  PimObjId srcObj1 = pimAlloc(PIM_ALLOC_AUTO, vectorLength, PIM_INT32);
  assert(srcObj1 != -1);
  PimObjId srcObj2 = pimAllocAssociated(srcObj1, PIM_INT32);
  assert(srcObj2 != -1);
  PimObjId dstObj = pimAllocAssociated(srcObj1, PIM_INT32);
  assert(dstObj != -1);

  PimStatus status = pimCopyHostToDevice((void *) src1.data(), srcObj1);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void *) src2.data(), srcObj2);
  assert(status == PIM_OK);

  status = pimXor(srcObj1, srcObj2, dstObj);
  assert(status == PIM_OK);

  dst.resize(vectorLength);
  status = pimCopyDeviceToHost(dstObj, (void *) dst.data());
  assert(status == PIM_OK);

  pimFree(srcObj1);
  pimFree(srcObj2);
  pimFree(dstObj);
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::vector<int> src1, src2, dst;
  uint64_t vectorLength = params.dataSize;
  
  if (params.inputFile == nullptr)
  {
    src1.resize(vectorLength);
    src2.resize(vectorLength);
    
    getVector(vectorLength, src1);
    getVector(vectorLength, src2);
  }
  else
  {
    std::cerr << "Reading from the input file is not implemented yet for the input matrix" << std::endl;
    return 1;
  }

  printf("Performing XOR with %llu data points\n", vectorLength);

  if (!createDevice(params.configFile))
  {
    return 1;
  }

  XOR(vectorLength, src1, src2, dst);

  if (params.shouldVerify)
  {  
    int errorFlag = 0;

    #pragma omp parallel for 
    for (uint64_t i = 0; i < vectorLength; ++i) 
    {    
      if ((src1[i] ^ src2[i]) != dst[i])
      {
        std::cout << "Wrong answer at index " << i << " | Wrong PIM answer = " << dst[i] << " (Baseline expected = " << (src1[i] ^ src2[i]) << ")" << std::endl;
        errorFlag = 1;
      }
    }
    if (!errorFlag)
    {
      std::cout << "Correct!" << std::endl;
    }
  }

  pimShowStats();

  return 0;
}
