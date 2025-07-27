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

#include "util.h"
#include "libpimeval.h"
#include <assert.h>

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
          "\n    -r    matrix row (default=2048 elements)"
          "\n    -d    matrix column (default=64 elements)"
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

void gemv(uint64_t row, uint64_t col, std::vector<int> &srcVector, std::vector<std::vector<int>> &srcMatrix, std::vector<int> &dst)
{
  PimObjId srcObj1 = pimAlloc(PIM_ALLOC_AUTO, row, PIM_INT32);
  if (srcObj1 == -1)
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

    status = pimScaledAdd(srcObj1, dstObj, dstObj, srcVector[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
  }

  dst.resize(row);
  status = pimCopyDeviceToHost(dstObj, (void *)dst.data());
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
  }
  pimFree(srcObj1);
  pimFree(dstObj);
}

void gemv_aim(uint64_t row, uint64_t col, std::vector<int> &srcVector, std::vector<std::vector<int>> &srcMatrix, std::vector<int> &dst, PimDeviceProperties &deviceProps)
{
  unsigned elementPerRow = deviceProps.numColPerSubarray / 32; // 32 bits per elements
  uint64_t rowItr = std::ceil(static_cast<double>(row) / deviceProps.numPIMCores); // Number of rows per core
  dst.resize(rowItr * deviceProps.numPIMCores , 0); // Initialize result vector

  uint64_t numChunks = std::ceil(static_cast<double>(col) / elementPerRow);
  for (uint64_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
    uint64_t chunkStart = chunkIdx * elementPerRow;
    uint64_t chunkSize = std::min(elementPerRow, static_cast<unsigned>(col - chunkStart));
    uint64_t rowChunk = std::ceil(static_cast<double>(row) / deviceProps.numPIMCores); // Number of rows per core

    std::vector<int> matrixChunkData(row * chunkSize);

    // Allocate PIM objects
    PimObjId vectorChunkObj = pimAllocBuffer(chunkSize, PIM_INT32);
    assert(vectorChunkObj != -1);

    PimObjId matrixChunkObj = pimAlloc(PIM_ALLOC_AUTO, chunkSize * deviceProps.numPIMCores, PIM_INT32);
    if (matrixChunkObj == -1) {
      std::cout << "Abort" << std::endl;
      return;
    }

    // Prepare vector chunk
    std::vector<int> vectorChunk(chunkSize, 0);
    for (uint64_t i = 0; i < chunkSize; ++i) {
      vectorChunk[i] = srcVector[chunkStart + i];
    }
    
    // Prepare matrix chunk: for each row, get the corresponding columns
    for (uint64_t rowIdx = 0; rowIdx < row; ++rowIdx) {
      for (uint64_t colIdx = chunkStart; colIdx < chunkSize; ++colIdx) {
        matrixChunkData[rowIdx * chunkSize + colIdx] = srcMatrix[rowIdx][chunkStart + colIdx];
      }
    }
    // Copy data to PIM
    PimStatus status = pimCopyHostToDevice(vectorChunk.data(), vectorChunkObj);
    assert(status == PIM_OK);

    for (uint64_t idx = 0; idx < rowChunk; ++idx) {
      status = pimCopyHostToDevice(matrixChunkData.data() + idx*deviceProps.numPIMCores, matrixChunkObj);
      assert(status == PIM_OK);
    
      // Perform MAC operation: compute partial dot products
      status = pimMAC(matrixChunkObj, vectorChunkObj, dst.data() + idx * deviceProps.numPIMCores);
      assert(status == PIM_OK);
    }
    pimFree(vectorChunkObj);
    pimFree(matrixChunkObj);
  }
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "Running GEMV for matrix row: " << params.row << " column: " << params.column << " and vector of size: " << params.column << std::endl;

  std::vector<int> srcVector (params.column, 1), resultVector;
  std::vector<std::vector<int>> srcMatrix (params.column, std::vector<int>(params.row, 1)); // matrix should lay out in colXrow format for bitserial PIM

  if (params.shouldVerify) {
    if (params.inputFile == nullptr)
    {
      getVector(params.column, srcVector);
      getMatrix(params.column, params.row, 0, srcMatrix);
    }
    else
    {
      std::cout << "Reading from input file is not implemented yet." << std::endl;
      return 1;
    }
  }

  if (!createDevice(params.configFile))
  {
    return 1;
  }

  PimDeviceProperties deviceProps;
  PimStatus status = pimGetDeviceProperties(&deviceProps);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return 1;
  }
  if (deviceProps.simTarget == PIM_DEVICE_AIM)
  {
    std::vector<std::vector<int>> tempMatrix (params.row, std::vector<int>(params.column, 1));
    for (size_t i = 0; i < params.row; ++i)
    {
      for (size_t j = 0; j < params.column; ++j)
      {
        tempMatrix[i][j] = srcMatrix[j][i];
      }
    }
    gemv_aim(params.row, params.column, srcVector, tempMatrix, resultVector, deviceProps);
  } else {
    gemv(params.row, params.column, srcVector, srcMatrix, resultVector);
  }

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
        result += srcMatrix[j][i] * srcVector[j];
      }
      if (result != resultVector[i])
      {
        #pragma omp critical
        {
          if (!shouldBreak)
          { // check the flag again in a critical section
            std::cout << "idx: " << i << " Wrong answer: " << resultVector[i] << " (expected " << result << ")" << std::endl;
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
