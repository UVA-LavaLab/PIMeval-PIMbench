// Test: C++ version of matrix vector multiplication
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

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t row, column;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./gemv.out [options]"
          "\n"
          "\n    -r    matrix row (default=8M elements)"
          "\n    -d    matrix column (default=8M elements)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing two vectors (default=generates vector with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.row = 65536;
  p.column = 65536;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:r:d:c:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'r':
      p.row = strtoull(optarg, NULL, 0);
      break;
    case 'd':
      p.column = strtoull(optarg, NULL, 0);
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

void gemv(uint64_t row, uint64_t col, std::vector<int> &srcVector, std::vector<std::vector<int>> &srcMatrix, std::vector<int64_t> &dst)
{
  PimObjId srcObj1 = pimAlloc(PIM_ALLOC_AUTO, col, PIM_INT32);
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

  PimObjId dstObj = pimAllocAssociated(srcObj1, PIM_INT32);
  if (dstObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimStatus status = pimCopyHostToDevice((void *)srcVector.data(), srcObj2);
  dst.reserve(row);
  for (int i = 0; i < row; ++i)
  {
    status = pimCopyHostToDevice((void *)srcMatrix[i].data(), srcObj1);
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

    status = pimRedSumInt(dstObj, &dst[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
  }

  pimFree(srcObj1);
  pimFree(srcObj2);
  pimFree(dstObj);
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "Row: " << params.row << " Column: " << params.column << "\n";

  std::vector<int> srcVector;
  std::vector<int64_t> resultVector;
  std::vector<std::vector<int>> srcMatrix; // matrix should lay out in colXrow format for bitserial PIM
  if (params.inputFile == nullptr)
  {
    getVector(params.column, srcVector);
    getMatrix(params.row, params.column, 0, srcMatrix);
  }
  else
  {
    std::cout << "Reading from input file is not implemented yet." << std::endl;
    return 0;
  }

  if (!createDevice(params.configFile))
    return 1;

  // TODO: Check if vector can fit in one iteration. Otherwise need to run in multiple iteration.
  gemv(params.row, params.column, srcVector, srcMatrix, resultVector);

  if (params.shouldVerify)
  {
    bool shouldBreak = false; // shared flag variable
// verify result
#pragma omp parallel for
    for (size_t i = 0; i < params.row; ++i)
    {
      if (shouldBreak) continue;
      int result = 0;
      for (size_t j = 0; j < params.column; ++j)
      {
        result += srcMatrix[i][j] * srcVector[j];
      }
      if (result != resultVector[i])
      {
#pragma omp critical
        {
          if (!shouldBreak)
          { // check the flag again in a critical section
            std::cout << "Wrong answer: " << resultVector[i] << " (expected " << result << ")" << std::endl;
            shouldBreak = true; // set the flag to true
          }
        }
      }
    }
    if (!shouldBreak) {
      std::cout << "\n\nCorrect Answer!!\n\n";
    }
  }

  pimShowStats();

  return 0;
}
