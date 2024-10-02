// Test: C++ version of matrix matrix multiplication
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

void gemv(uint64_t row, uint64_t col, std::vector<int> &srcVector, std::vector<std::vector<int>> &srcMatrix, std::vector<int> &dst)
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

    status = pimMulScalar(srcObj1, srcObj2, srcVector[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }

    status = pimAdd(srcObj2, dstObj, dstObj);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
  }

  dst.reserve(row);
  status = pimCopyDeviceToHost(dstObj, (void *)dst.data());
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
  }
  pimFree(srcObj1);
  pimFree(srcObj2);
  pimFree(dstObj);
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

void gemm(uint64_t row, uint64_t colA, uint64_t colB, std::vector<std::vector<int>> &srcMatrixA, std::vector<std::vector<int>> &srcMatrixB, std::vector<std::vector<int>> &dstMatrix, bool shouldVerify)
{
  dstMatrix.resize(row, std::vector<int>(colB, 0));
  std::vector<std::vector<int>> transposedDstMat(colB, std::vector<int>(row, 0));
  vector<std::vector<int>> srcMatrixAT(colA, std::vector<int>(row, 0)), srcMatrixBT(colB, std::vector<int>(colA, 0));
  // TODO: Do we actually need to transpose matrices
  transposeMatrix(row, colA, srcMatrixA, srcMatrixAT);
  transposeMatrix(colA, colB, srcMatrixB, srcMatrixBT);
  for (uint64_t i = 0; i < colB; ++i)
  {
    gemv(row, colA, srcMatrixBT[i], srcMatrixAT, transposedDstMat[i]);
  }
  transposeMatrix(colB, row, transposedDstMat, dstMatrix);
  if (shouldVerify)
  {
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
