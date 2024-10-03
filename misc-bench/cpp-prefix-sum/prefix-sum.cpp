// Test: C++ version of prefix sum
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "../util.h"
#include "libpimeval.h"
#include <getopt.h>
#include <stdint.h>
#include <iomanip>

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace std;

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
          "\n    -i    input file containing interger vector (default=generates datapoints with random numbers)"
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

void prefixSum(vector<int> &input, vector<int> &deviceoutput, uint64_t len)
{
  std::vector<PimObjId> temp(len);
  std::vector<PimObjId> acc(len);

  PimObjId inputObj = pimAlloc(PIM_ALLOC_AUTO, len, PIM_INT32);
  if (inputObj == -1)
  {
    std::cerr << "Abort: Failed to allocate memory on PIM." << std::endl;
    return;
  }

  PimStatus status = pimCopyHostToDevice((void *)input.data(), inputObj);
  if (status != PIM_OK)
  {
    std::cerr << "Abort: Failed to copy data to PIM." << std::endl;
    return;
  }
  PimObjId tempObj = pimAllocAssociated(inputObj, PIM_INT32);
  if (tempObj == -1)
  {
    std::cerr << "Abort: Failed to allocate memory on PIM." << std::endl;
    return;
  }
  status = pimCopyHostToDevice((void *)input.data(), tempObj);
  if (status != PIM_OK)
  {
    std::cerr << "Abort: Failed to copy data to PIM." << std::endl;
    return;
  }

  PimObjId accObj = pimAllocAssociated(inputObj, PIM_INT32);
  if (accObj == -1)
  {
    std::cerr << "Abort: Failed to allocate memory on PIM." << std::endl;
    return;
  }

  status = pimCopyHostToDevice((void *)input.data(), accObj);
  if (status != PIM_OK)
  {
    std::cerr << "Abort: Failed to copy data to PIM." << std::endl;
    return;
  }

  PimObjId outputObj = pimAllocAssociated(inputObj, PIM_INT32);
  if (outputObj == -1)
  {
    std::cerr << "Abort: Failed to allocate memory on PIM." << std::endl;
    return;
  }

  status = pimCopyHostToDevice((void *)deviceoutput.data(), outputObj);
  if (status != PIM_OK)
  {
    std::cerr << "Abort: Failed to copy data to PIM." << std::endl;
    return;
  }

  while (len > 0)
  {
    pimShiftElementsRight(tempObj);
    status = pimAdd(tempObj, accObj, accObj);
    if (status != PIM_OK)
    {
      std::cerr << "Abort: Failed to perform PIM addition." << std::endl;
      return;
    }
    len--;
  }

  status = pimCopyDeviceToHost(accObj, (void *)deviceoutput.data());
  if (status != PIM_OK)
  {
    std::cerr << "Abort: Failed to copy prefix sum result from PIM." << std::endl;
    return;
  }

  // Clean up PIM objects
  pimFree(accObj);
  pimFree(tempObj);
  pimFree(inputObj);
  pimFree(outputObj);
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  vector<int> input;
  if (params.inputFile == nullptr)
  {
    getVector(params.vectorLength, input);
  }
  else
  {
    std::cout << "Reading from input file is not implemented yet." << std::endl;
    return 1;
  }

  int len = input.size();
  vector<int> deviceoutput;
  vector<int> hostoutput(len);

  for (int i = 0; i < input.size(); i++)
  {
    deviceoutput.push_back(0);
  }

  hostoutput[0] = input[0];
  for (int i = 0; i < input.size(); i++)
  {
    hostoutput[i + 1] = hostoutput[i] + input[i + 1];
  }

  if (!createDevice(params.configFile))
    return 1;
  prefixSum(input, deviceoutput, len);

  // Verification of Results hostresults vs deviceresults
  if (params.shouldVerify)
  {
    // verify result
#pragma omp parallel for
    for (unsigned i = 0; i < len; ++i)
    {
      if (hostoutput[i] != deviceoutput[i])
      {
        std::cout << "Wrong answer for Prefixsum: " << hostoutput[i] << " != " << deviceoutput[i] << std::endl;
      }
    }
  }

  pimShowStats();
  return 0;
}
