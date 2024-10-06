// A trivial database filtering implementation on bitSIMD
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include "../../util.h"
#include <iostream>
#include <vector>
#include <getopt.h>
#include <cstdlib>
#include <time.h>
#include <algorithm>
#include <chrono>
#if defined(_OPENMP)
#include <omp.h>
#endif

#define MY_RANGE 100

using namespace std;

std::chrono::duration<double, std::milli> hostElapsedTime = std::chrono::duration<double, std::milli>::zero();

typedef struct Params
{
  uint64_t inVectorSize;
  int key;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\n    Usage:  ./filter.out [options]"
          "\n"
          "\n    -n    database size (default=2048 elements)"
          "\n    -k    value of key (default = 1)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing the database (default=generates datapoints with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=true)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.inVectorSize = 2048;
  p.key = 1;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = true;

  int opt;
  while ((opt = getopt(argc, argv, "h:n:k:c:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'n':
      p.inVectorSize = strtoull(optarg, NULL, 0);
      break;
    case 'k':
      p.key = strtoull(optarg, NULL, 0);
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

void filterByKey(std::vector<int> &Vector, uint64_t vector_size, int key, std::vector<int> &bitMap)
{
  for (uint64_t i = 0; i < vector_size; ++i)
  {
    if (key > Vector[i])
      bitMap[i] = true;
  }
}

int main(int argc, char **argv)
{

  struct Params p = getInputParams(argc, argv);
  std::cout << "PIM test: database-filtering" << std::endl;
  uint64_t inVectorSize;
  int key;

  if (p.inputFile == nullptr)
  {
    inVectorSize = p.inVectorSize;
    key = p.key;
  }
  else
  {
    std::cout << "Reading from input file is not implemented yet." << std::endl;
    return 1;
  }

  if (!createDevice(p.configFile))
  {
    return 1;
  }

  // TODO: Check if vector can fit in one iteration. Otherwise need to run in multiple iteration.

  vector<int> inVector(inVectorSize);
  vector<int> outVector;
  vector<int> outVectorHost;

  std::cout << "DB element size: " << inVectorSize << std::endl;

  // randomly initialize input vector with value between 0 and MY_RANGE
  srand((unsigned)time(NULL));
  for (uint64_t i = 0; i < inVectorSize; i++)
  {
    inVector[i] = rand() % MY_RANGE;
  }

  // initialize the vector on the host to hold the output bitmap
  std::vector<int> bitMap(inVectorSize, 0);
  std::vector<int> bitMapHost(inVectorSize, 0);

  PimObjId srcObj1 = pimAlloc(PIM_ALLOC_AUTO, inVector.size(), PIM_INT32);
  if (srcObj1 == -1)
  {
    std::cout << "Abort" << std::endl;
    return 1;
  }
  PimObjId srcObj2 = pimAllocAssociated(srcObj1, PIM_INT32);
  if (srcObj2 == -1)
  {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  PimStatus status = pimCopyHostToDevice((void *)inVector.data(), srcObj1);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  status = pimLTScalar(srcObj1, srcObj2, key);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  status = pimCopyDeviceToHost(srcObj2, (void *)bitMap.data());
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
  }

  auto start_cpu = std::chrono::high_resolution_clock::now();

  // select data whose bitmap is '1'
  // Do parallel reduction with a degree of (32) by creating 32 subarrays and then combining the results in these 32 array serially.
  uint64_t nThreads = 16;
  cout << "nThreads = " << nThreads << endl;
  uint64_t outSubSize = inVectorSize / nThreads;
  vector<vector<int32_t>> outSubVector(nThreads);

#pragma omp parallel num_threads(nThreads)
  for (uint64_t i = 0; i < outSubSize; i++)
  {
#if defined(_OPENMP)
    int tid = omp_get_thread_num();
#else
    int tid = 0; // Fallback for when OpenMP is not available
#endif
    uint64_t index = tid * outSubSize + i;
    if (index < inVectorSize)
    {
      if (bitMap[index] == true)
      {
        outSubVector[tid].push_back(inVector[index]);
      }
    }
  }
  int outSize = 0;

#pragma omp critical
  for (uint64_t i = 0; i < nThreads; i++)
  {
    outSize += outSubVector[i].size();
  }
#pragma omp critical
  for (uint64_t i = 0; i < nThreads; i++)
  {
    outVector.insert(outVector.end(), outSubVector[i].begin(), outSubVector[i].end());
  }

  auto stop_cpu = std::chrono::high_resolution_clock::now();
  hostElapsedTime += (stop_cpu - start_cpu);

  // run scan on host
  filterByKey(inVector, inVectorSize, p.key, bitMapHost);

  // select data whose bitmap is '1' on host based on bitmap from host
  for (uint64_t i = 0; i < inVectorSize; i++)
  {
    if (bitMapHost[i] == 1)
    {
      outVectorHost.push_back(inVector[i]);
    }
  }
  // !check results and print it like km
  if (p.shouldVerify)
  {
    bool ok = true;
    if (outVector != outVectorHost)
    {
      std::cout << "Wrong answer!" << std::endl;
      ok = false;
    }
    if (ok)
    {
      std::cout << "All correct!" << std::endl;
    }
  }
  pimShowStats();

  cout << "Host elapsed time: " << std::fixed << std::setprecision(3) << hostElapsedTime.count() << " ms." << endl;
  return 0;
}
