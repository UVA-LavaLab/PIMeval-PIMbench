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

#include "../../util.h"
#include "libpimeval.h"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t row, column, batchSize;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./gemv.out [options]"
          "\n"
          "\n    -r    matrix row (default=2048 elements)"
          "\n    -d    matrix column (default=64 elements)"
          "\n    -b    batch size (default=1)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing two vectors (default=generates vector with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.row = 2048;
  p.column = 64;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;
  p.batchSize = 1;

  int opt;
  while ((opt = getopt(argc, argv, "h:r:d:b:c:i:v:")) >= 0)
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
    case 'b':
      p.batchSize = strtoull(optarg, NULL, 0);
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

template <typename T>
void printVector(const std::vector<T>& vec) {
    for (const auto& element : vec) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
}

void gemv(uint64_t row, uint64_t col, uint64_t batchSize, const std::vector<std::vector<int>>& srcVectors, const std::vector<std::vector<std::vector<int>>>& srcMatrices, std::vector<std::vector<int64_t>>& dstBatch)
{
  PimStatus status = PIM_OK;
  unsigned bitsPerElement = sizeof(int) * 8;
  PimObjId srcObj1 = pimAlloc(PIM_ALLOC_AUTO, col * batchSize, bitsPerElement, PIM_INT32);
  if (srcObj1 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  PimObjId srcObj2 = pimAllocAssociated(bitsPerElement, srcObj1, PIM_INT32);
  if (srcObj2 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimObjId dstObj = pimAllocAssociated(bitsPerElement, srcObj1, PIM_INT32);
  if (dstObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  for (uint64_t batchId = 0; batchId < batchSize; batchId++)
  {
    status = pimCopyHostToDevice((void *)srcVectors[batchId].data(), srcObj2, batchId * col, (batchId + 1) * col);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
    dstBatch[batchId].reserve(row);
  }

  for (uint64_t i = 0; i < row; ++i)
  {
    for (uint64_t batchId = 0; batchId < batchSize; batchId++)
    {
      status = pimCopyHostToDevice((void *)srcMatrices[batchId][i].data(), srcObj1, batchId * col, (batchId + 1) * col);
      if (status != PIM_OK)
      {
        std::cout << "Abort" << std::endl;
        return;
      }
    }
    
    status = pimMul(srcObj1, srcObj2, dstObj);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }

    for (uint64_t batchId = 0; batchId < batchSize; batchId++)
    {
      status = pimRedSumRangedInt(dstObj, batchId * col, (batchId + 1) * col, &dstBatch[batchId][i]);
      if (status != PIM_OK)
      {
        std::cout << "Abort" << std::endl;
        return;
      }
    }
  }

  pimFree(srcObj1);
  pimFree(srcObj2);
  pimFree(dstObj);
}

int main(int argc, char *argv[])
{
    struct Params params = getInputParams(argc, argv);
    std::cout << "Running batch of GEMVs for matrix row: " << params.row << " column: " << params.column << " with batch size: " << params.batchSize << std::endl;

    std::vector<std::vector<int>> srcVectors(params.batchSize, std::vector<int>(params.column));
    std::vector<std::vector<int64_t>> resultVectors(params.batchSize, std::vector<int64_t>(params.row));
    std::vector<std::vector<std::vector<int>>> srcMatrices(params.batchSize, std::vector<std::vector<int>>(params.column, std::vector<int>(params.row)));
        
    if (params.inputFile == nullptr)
    {
        for (size_t i = 0; i < params.batchSize; ++i) {
            getVector(params.column, srcVectors[i]);
            getMatrix(params.row, params.column, 0, srcMatrices[i]);
        }
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

    // Process the batch of GEMVs
    gemv(params.row, params.column, params.batchSize, srcVectors, srcMatrices, resultVectors);

    if (params.shouldVerify)
    {
        bool shouldBreak = false;

        // Verify the result
        #pragma omp parallel for
        for (size_t b = 0; b < params.batchSize; ++b) {
            for (size_t i = 0; i < params.row; ++i)
            {
                if (shouldBreak) continue;
                int result = 0;
                for (size_t j = 0; j < params.column; ++j)
                {
                    result += srcMatrices[b][i][j] * srcVectors[b][j];
                }
                if (result != resultVectors[b][i])
                {
                    #pragma omp critical
                    {
                        if (!shouldBreak)
                        {
                            std::cout << "Wrong answer in batch " << b << ": " << resultVectors[b][i] << " (expected " << result << ")" << std::endl;
                            shouldBreak = true;
                        }
                    }
                }
            }
        }

        if (!shouldBreak) {
            std::cout << "\n\nCorrect Answer for All Batch!!\n\n";
        }
    }

    pimShowStats();

    return 0;
}

