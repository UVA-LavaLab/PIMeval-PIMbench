// Test: C++ version of matrix matrix multiplication with batched matrix vector multiplication
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
  uint64_t row, columnA, columnB;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./gemm.out [options]"
          "\n"
          "\n    -r    matrix1 row (default=1024 elements)"
          "\n    -d    matrix1 column (default=256 elements)"
          "\n    -z    matrix2 column (default=64 elements)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing two vectors (default=generates vector with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.row = 1024;
  p.columnA = 256;
  p.columnB = 64;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:r:d:z:c:i:v:")) >= 0)
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
      p.columnA = strtoull(optarg, NULL, 0);
      break;
    case 'z':
      p.columnB = strtoull(optarg, NULL, 0);
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

void gemvBatched(uint64_t row, uint64_t col, std::vector<std::vector<int>> &srcVector, std::vector<std::vector<int>> &srcMatrix, std::vector<int> &dst)
{
  PimObjId srcObj1 = pimAlloc(PIM_ALLOC_AUTO, row, PIM_INT32);
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

  PimObjId dstObj1 = pimAllocAssociated(srcObj1, PIM_INT32);
  if (dstObj1 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimStatus status = pimBroadcastInt(dstObj, 0);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  
  for (uint64_t i = 0; i < col; ++i)
  {
    status = pimCopyHostToDevice((void *)srcMatrix[i].data(), srcObj1);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }

    status = pimCopyHostToDevice((void *)srcVector[i].data(), srcObj2);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }

    status = pimMul(srcObj1, srcObj2, dstObj1);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }

    status = pimAdd(dstObj1, dstObj, dstObj);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
  }

  //dst.reserve(row);
  status = pimCopyDeviceToHost(dstObj, (void *)dst.data());
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
  }
  pimFree(srcObj1);
  pimFree(srcObj2);
  pimFree(dstObj);
  pimFree(dstObj1);
}

void transposeMatrix(uint64_t row, uint64_t col, std::vector<std::vector<int>> &srcMatrix, std::vector<std::vector<int>> &dstMatrix)
{
#pragma omp parallel for
  for (uint64_t i = 0; i < col; ++i)
  {
    for (uint64_t j = 0; j < row; ++j)
    {
      dstMatrix[i][j] = srcMatrix[j][i];
    }
  }
}

void verifyResult(uint64_t row, uint64_t colA, uint64_t colB, std::vector<std::vector<int>> &srcMatrixAT, std::vector<std::vector<int>> &srcMatrixBT, std::vector<std::vector<int>> &dstMatrix)
{
  printMatrix(dstMatrix);
  cout << "Starting verification......\n";
  std::vector<std::vector<int>> C(row, std::vector<int>(colB, 0));
  for (uint64_t i = 0; i < row; ++i)
  {
    for (uint64_t j = 0; j < colB; ++j)
    {
      for (uint64_t k = 0; k < colA; ++k)
      {
        C[i][j] += srcMatrixAT[k][i] * srcMatrixBT[j][k];
      }
    }
  }
  printMatrix(C);
  bool shouldContinue = true;
  for (uint64_t i = 0; i < row && shouldContinue; ++i)
  {
    for (uint64_t j = 0; j < colB; ++j)
    {
      if (C[i][j] != dstMatrix[i][j])
      {
        std::cout << "Error: Incorrect Result for:" << i << ".\nHost: " << C[i][j] << "\t PIM: " << dstMatrix[i][j] << "\n";
        shouldContinue = false;
        break;
      }
    }
  }
}

void gemm(uint64_t row, uint64_t colA, uint64_t colB, std::vector<std::vector<int>> &srcMatrixA, std::vector<std::vector<int>> &srcMatrixB, std::vector<std::vector<int>> &dstMatrix, bool shouldVerify)
{
  dstMatrix.resize(row, std::vector<int>(colB, 0));
  std::vector<std::vector<int>> transposedDstMat(colB, std::vector<int>(row, 0));
  vector<std::vector<int>> srcMatrixAT(colA, std::vector<int>(row, 0)), srcMatrixBT(colB, std::vector<int>(colA, 0));

  // TODO: Do we actually need to transpose matrices
  transposeMatrix(row, colA, srcMatrixA, srcMatrixAT);
  transposeMatrix(colA, colB, srcMatrixB, srcMatrixBT);

  PimDeviceProperties deviceProp;
  PimStatus status = pimGetDeviceProperties(&deviceProp);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  // How many GEMVs can be processed in parallel?
  // Following calculation is done based on bit serial data layout
  uint64_t totalAvailableBitsPerItr = (uint64_t)deviceProp.numRanks * (uint64_t)deviceProp.numBankPerRank * (uint64_t)deviceProp.numSubarrayPerBank * (uint64_t)deviceProp.numRowPerSubarray * (uint64_t)deviceProp.numColPerSubarray;
  uint64_t totalBitsRequired = colB * row * 32 * 4; // assuming 32bits INT
  uint64_t batchSize = std::min(colB, (uint64_t)std::ceil(totalAvailableBitsPerItr * 1.0 / totalBitsRequired));

  // Concatenate matrices. Needs to be done once
  std::vector<std::vector<int>> batchedSrcMatA;
  for (uint64_t i = 0; i < colA; ++i)
  {
    std::vector<int> temp;
    for (uint64_t j = 0; j < batchSize; ++j) {
      // TODO: Add replicate a pattern API in the simulator
      for (uint64_t k = 0; k < row; ++k) {
        temp.push_back(srcMatrixAT[i][k]);
      }
    }
    batchedSrcMatA.push_back(temp);
  }

  std::cout << "Done concatenating A." << std::endl;

  for (uint64_t i = 0; i < colB; i += batchSize)
  {
    uint64_t currBatchSize = std::min(batchSize, colB - i);

    std::cout << "Starting Gemv for batch size: " << currBatchSize << std::endl;

    std::vector<std::vector<int>> batchedSrcMatB(colA, std::vector<int>(row * batchSize));
    for (uint64_t j = 0; j < colA; ++j)
    {
      for (uint64_t k = 0; k < currBatchSize; ++k) {
        int valueT = srcMatrixBT[i + k][j];
        uint64_t idx = k * row;
        // replicate valueT row times
        for (uint64_t l = 0; l < row; ++l) {
          batchedSrcMatB[j][idx + l] = valueT;
        }
      }
    }

    std::cout << "Done concatenating B." << std::endl;
    std::vector<int> tempDst(row * currBatchSize, 0);
    gemvBatched(row*currBatchSize, colA, batchedSrcMatB, batchedSrcMatA, tempDst);
    std::cout << "Done Running GEMV." << std::endl;

    for (uint64_t j = 0; j < currBatchSize; ++j)
    {
      for (uint64_t k = 0; k < row; ++k) {
        transposedDstMat[i+j][k] = tempDst[j * row + k];
      }
    }
  }
  transposeMatrix(colB, row, transposedDstMat, dstMatrix);
  
  if (shouldVerify)
  {
    //verifyResult(row, colA, colB, srcMatrixAT, srcMatrixBT, dstMatrix);
    cout << "Starting verification......\n";
    std::vector<std::vector<int>> C(row, std::vector<int>(colB, 0));
    for (uint64_t i = 0; i < row; ++i)
    {
      for (uint64_t j = 0; j < colB; ++j)
      {
        for (uint64_t k = 0; k < colA; ++k)
        {
          C[i][j] += srcMatrixAT[k][i] * srcMatrixBT[j][k];
        }
      }
    }
    bool shouldContinue = true;
    for (uint64_t i = 0; i < row && shouldContinue; ++i)
    {
      for (uint64_t j = 0; j < colB; ++j)
      {
        if (C[i][j] != dstMatrix[i][j])
        {
          std::cout << "Error: Incorrect Result.\nHost: " << C[i][j] << "\t PIM: " << dstMatrix[i][j] << "\n";
          shouldContinue = false;
          break;
        }
      }
    }
  }
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "Running GEMM on PIM for matrix of row: " << params.row << " column: " << params.columnA << " and matrix of row: " << params.columnA << " column: " << params.columnB << std::endl;

  std::vector<int> srcVector, resultVector;
  std::vector<std::vector<int>> srcMatrixA, srcMatrixB, dstMatrix;
  if (params.inputFile == nullptr)
  {
    getMatrix(params.row, params.columnA, 0, srcMatrixA);
    getMatrix(params.columnA, params.columnB, 0, srcMatrixB);
  }
  else
  {
    std::cout << "Reading from input file is not implemented yet." << std::endl;
    return 1;
  }

  if (!createDevice(params.configFile))
    return 1;

  // TODO: Check if vector can fit in one iteration. Otherwise need to run in multiple iteration.
  gemm(params.row, params.columnA, params.columnB, srcMatrixA, srcMatrixB, dstMatrix, params.shouldVerify);

  pimShowStats();

  return 0;
}
