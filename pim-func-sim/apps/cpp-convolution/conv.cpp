// Test: C++ version of convolution
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "libpimsim.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <getopt.h>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "../util.h"

using namespace std;

void vectorAddition(uint64_t vectorLength, const std::vector<int> &src1, const std::vector<int> &src2, std::vector<int> &dst)
{
  unsigned bitsPerElement = sizeof(int) * 8;
  PimObjId srcObj1 = pimAlloc(PIM_ALLOC_V1, vectorLength, bitsPerElement, PIM_INT32);
  if (srcObj1 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  PimObjId srcObj2 = pimAllocAssociated(PIM_ALLOC_V1, vectorLength, bitsPerElement, srcObj1, PIM_INT32);
  if (srcObj2 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  PimObjId dstObj = pimAllocAssociated(PIM_ALLOC_V1, vectorLength, bitsPerElement, srcObj1, PIM_INT32);
  if (dstObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimStatus status = pimCopyHostToDevice(PIM_COPY_V, (void *)src1.data(), srcObj1);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimCopyHostToDevice(PIM_COPY_V, (void *)src2.data(), srcObj2);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimAdd(srcObj1, srcObj2, dstObj);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  dst.resize(vectorLength);
  status = pimCopyDeviceToHost(PIM_COPY_V, dstObj, (void *)dst.data());
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
  }
  pimFree(srcObj1);
  pimFree(srcObj2);
  pimFree(dstObj);
// verify result
#pragma omp parallel for
  for (unsigned i = 0; i < vectorLength; ++i)
  {
    int sum = src1[i] + src2[i];
    if (dst[i] != sum)
    {
      std::cout << "Wrong answer for addition: " << src1[i] << " + " << src2[i] << " = " << dst[i] << " (expected " << sum << ")" << std::endl;
    }
  }
}

void getDecomposedMatrix(int matrixRow, int matrixColumn, int filterRow, int filterColumn, int stride, std::vector<std::vector<int>>& inputMatrix, std::vector<std::vector<int>>& decompMatrix) {

  decompMatrix.resize(filterRow*filterColumn, std::vector<int>(matrixRow*matrixColumn, 0));
  int colIdx = 0;
  for (int i = 0; i < (inputMatrix.size() - filterRow + 1); i += stride)
  {
    for (int j = 0; j < (inputMatrix[i].size() - filterColumn + 1); j += stride)
    {
      int rowIDX = 0;
      for (int k = i; k < i + filterRow; k++)
      {
        for (int l = j; l < j + filterColumn; l++)
        {
          decompMatrix[rowIDX++][colIdx] = inputMatrix[k][l];
        }
      }
      ++colIdx;
    }
  }
}

void performConv(std::vector<std::vector<int>>& filterMatrix, std::vector<std::vector<int>>& inputMatrix, std::vector<int>& outputMatrix, int numRequiredPIMRows, int numRequiredPIMCol) {


  unsigned bitsPerElement = 32;
  std::vector<PimObjId> filterObjects;

  PimObjId obj1 = pimAlloc(PIM_ALLOC_V1, numRequiredPIMCol, bitsPerElement, PIM_INT32);
  if (obj1 == -1)
  {
    std::cout << "Abort" << std::endl;
    return ;
  }
  filterObjects.push_back(obj1);
  for (int i = 1; i < numRequiredPIMRows; i++)
  {
    PimObjId obj = pimAllocAssociated(PIM_ALLOC_V1, numRequiredPIMCol, bitsPerElement, filterObjects[0], PIM_INT32);
    if (obj == -1)
    {
      std::cout << "Abort" << std::endl;
      return ;
    }
    filterObjects.push_back(obj);
  }

  int idx = 0;
  for (int i = 0; i < filterMatrix.size(); ++i)
  {
    for (int j = 0; j < filterMatrix[i].size(); ++j)
    {
      PimStatus status = pimBroadCast(PIM_COPY_V, filterObjects[idx], filterMatrix[i][j]);
      if (status != PIM_OK)
      {
        std::cout << "Abort" << std::endl;
        return;
      }
    }
  }

  std::vector<PimObjId> matrixObjects;
  for (int i = 0; i < numRequiredPIMRows; i++)
  {
    PimObjId obj = pimAllocAssociated(PIM_ALLOC_V1, inputMatrix[i].size(), bitsPerElement, filterObjects[0], PIM_INT32);
    if (obj == -1)
    {
      std::cout << "Abort" << std::endl;
      return ;
    }
    matrixObjects.push_back(obj);
  }

  for (int i = 0; i < inputMatrix.size(); i++)
  {
    PimStatus status = pimCopyHostToDevice(PIM_COPY_V, (void *)inputMatrix[i].data(), matrixObjects[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return ;
    }
  }

  for (int i = 0; i < inputMatrix.size(); i++)
  {
    PimStatus status = pimMul(matrixObjects[i], filterObjects[i], filterObjects[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return ;
    }
  }

  for (int i = 1; i < numRequiredPIMRows; i++)
  {
    PimStatus status = pimAdd(filterObjects[0], filterObjects[i], filterObjects[0]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return ;
    }
  }
  outputMatrix.resize(numRequiredPIMCol);

  PimStatus status = pimCopyDeviceToHost(PIM_COPY_V, filterObjects[0], outputMatrix.data());
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return ;
  }

  for (auto elem : filterObjects) {
    pimFree(elem);
  }

  for (auto elem : matrixObjects) {
    pimFree(elem);
  }
}

// Params ---------------------------------------------------------------------
typedef struct Params
{
  int row, column, dim, stride, kernelSize, kernelDim, padding;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./conv [options]"
          "\n"
          "\n    -r    row (default=224)"
          "\n    -c    column (default=224)"
          "\n    -d    dimension (default=3)"
          "\n    -s    stride (default=1)"
          "\n    -k    kernel size (default=3X3)"
          "\n    -z    kernel dimension (default=64)"
          "\n    -v    should verify result with CPU"
          "\n    -p    padding (default = 1)"
          "\n    -f    input file containing matrices (default=generates matrix with random numbers)"
          "\n    -i    input file containing matrices (default=generates matrix with random numbers)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.row = 224;
  p.column = 224;
  p.dim = 3;
  p.stride = 1;
  p.kernelSize = 3;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;
  p.kernelDim = 64;
  p.padding = 1;

  int opt;
  while ((opt = getopt(argc, argv, "r:c:d:s:k:v:z:f:i:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'r':
      p.row = atoi(optarg);
      break;
    case 'c':
      p.column = atoi(optarg);
      break;
    case 'd':
      p.dim = atoi(optarg);
      break;
    case 's':
      p.stride = atoi(optarg);
      break;
    case 'k':
      p.kernelSize = atoi(optarg);
      break;
    case 'z':
      p.kernelDim = atoi(optarg);
      break;
    case 'p':
      p.padding = atoi(optarg);
      break;
    case 'f':
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

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::vector<std::vector<std::vector<int>>> inputMatrix;
  std::vector<std::vector<std::vector<int>>> kernelMatrix;

  if (params.inputFile == nullptr)
  {
    inputMatrix.resize(params.dim);
    for (int i = 0; i < params.dim; i++)
    {
      getMatrix(params.row, params.column, params.padding, inputMatrix[i]);
    }
    kernelMatrix.resize(params.kernelDim);
    for (auto &mat : kernelMatrix)
    {
      getMatrix(params.kernelSize, params.kernelSize, 0, mat);
    }
  }
  else
  {
    // TODO: read Matrix from file
  }

  if (!createDevice(params.configFile)) return 1;

  // TODO: get number of columns after creating the device. Maybe support an API like getDeviceConfig. Besides 65536 is too large.
  unsigned numCols = 65536;

  int outMatDim = params.kernelDim;
  // TODO: this will not work if padding is not equal to 1
  int outMatRow = params.row;
  int outMatCol = params.column;
  int numOfMatPerRow = floor((1.0*numCols)/(outMatRow*outMatCol)) < params.dim ? floor((1.0*numCols)/(outMatRow*outMatCol)) : params.dim;
  int numOfPIMRow = params.kernelSize * params.kernelSize;

  std::vector<std::vector<std::vector<int>>> resultMatrix;
  resultMatrix.resize(outMatDim, std::vector<std::vector<int>>(outMatRow, std::vector<int>(outMatCol)));
  for (int i = 0; i < params.kernelDim; i++) {
    std::vector<int> srcVec, dstVec;
    for (int j = 0; j < params.dim; j+=numOfMatPerRow) {
      int matChunk = (numOfMatPerRow + j) <= params.dim ? (numOfMatPerRow + j) : params.dim;
      std::vector<std::vector<int>> mergedMat(numOfPIMRow);
      for (int k = j; k < matChunk; k++) {
        std::vector<std::vector<int>> decompMat;
        getDecomposedMatrix(params.row, params.column, kernelMatrix[i].size(), kernelMatrix[i][0].size(), params.stride, inputMatrix[k], decompMat);
        for (int idx = 0; idx < mergedMat.size(); idx++) {
          mergedMat[idx].reserve(mergedMat[idx].size() + decompMat[idx].size());
          mergedMat[idx].insert(mergedMat[idx].end(), make_move_iterator(decompMat[idx].begin()), make_move_iterator(decompMat[idx].end()));
        }
      }
      std::vector<int> outVector;
      performConv(kernelMatrix[i], mergedMat, outVector, numOfPIMRow, outMatCol*outMatRow);

      //For architectures that don't support reduction, either perfor reduction in host or a mix of host and device.
      if (!srcVec.empty()) {
        vectorAddition(outVector.size(), outVector, srcVec, dstVec);
        srcVec = dstVec;
      } else {
        srcVec = outVector;
      }
    }
    // performing the reduction on host
    for (int rdx = 0; rdx < outMatRow*outMatCol; ++rdx) {
      for (int cdx = outMatRow*outMatCol; cdx < dstVec.size(); cdx+=(outMatRow*outMatCol)-1){
        dstVec[rdx] += dstVec[cdx+rdx]; 
      }
    } 
    int ddx = 0;
    for (int rdx = 0; rdx < outMatRow; ++rdx) {
      for (int cdx = 0; cdx < outMatCol; ++cdx) {
        resultMatrix[i][rdx][cdx] = dstVec[ddx++];
      }
    }
  }

  pimShowStats();

  return 0;
}

