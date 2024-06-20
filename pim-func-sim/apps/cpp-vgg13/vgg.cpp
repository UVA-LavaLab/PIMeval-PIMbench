// Test: C++ version of vgg16
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "libpimsim.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <getopt.h>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "../util.h"
#include <iomanip>
#include <chrono>

using namespace std;

std::chrono::duration<double, std::milli> hostElapsedTime = std::chrono::duration<double, std::milli>::zero();

void getDecomposedMatrix(int matrixRow, int matrixColumn, int filterRow, int filterColumn, int stride, std::vector<std::vector<int>> &inputMatrix, std::vector<std::vector<int>> &decompMatrix)
{
  decompMatrix.resize(filterRow * filterColumn, std::vector<int>(matrixRow * matrixColumn, 0));
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
          if (rowIDX >= decompMatrix.size() || colIdx >= decompMatrix[0].size())
          {
            std::cerr << "Error: Out of bounds access detected. rowIdx: " << rowIDX << ", colIdx: " << colIdx << std::endl;
            return;
          }
          decompMatrix[rowIDX++][colIdx] = inputMatrix[k][l];
        }
      }
      ++colIdx;
    }
  }
}

void performConv(std::vector<std::vector<int>> &filterMatrix, std::vector<std::vector<int>> &inputMatrix, std::vector<int> &outputMatrix, int numRequiredPIMRows, int numRequiredPIMCol)
{

  unsigned bitsPerElement = 32;
  std::vector<PimObjId> filterObjects;

  PimObjId obj1 = pimAlloc(PIM_ALLOC_AUTO, numRequiredPIMCol, bitsPerElement, PIM_INT32);
  if (obj1 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  filterObjects.push_back(obj1);

  for (int i = 1; i < numRequiredPIMRows; i++)
  {
    PimObjId obj = pimAllocAssociated(bitsPerElement, filterObjects[0], PIM_INT32);
    if (obj == -1)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
    filterObjects.push_back(obj);
  }

  int idx = 0;
  for (int i = 0; i < filterMatrix.size(); ++i)
  {
    for (int j = 0; j < filterMatrix[i].size(); ++j)
    {
      PimStatus status = pimBroadcastSignedInt(filterObjects[idx], filterMatrix[i][j]);
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
    PimObjId obj = pimAllocAssociated(bitsPerElement, filterObjects[0], PIM_INT32);
    if (obj == -1)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
    matrixObjects.push_back(obj);
  }

  for (int i = 0; i < inputMatrix.size(); i++)
  {
    PimStatus status = pimCopyHostToDevice((void *)inputMatrix[i].data(), matrixObjects[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
  }

  for (int i = 0; i < inputMatrix.size(); i++)
  {
    PimStatus status = pimMul(matrixObjects[i], filterObjects[i], filterObjects[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
  }

  for (int i = 1; i < numRequiredPIMRows; i++)
  {
    PimStatus status = pimAdd(filterObjects[0], filterObjects[i], filterObjects[0]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
  }
  outputMatrix.resize(numRequiredPIMCol);

  PimStatus status = pimCopyDeviceToHost(filterObjects[0], outputMatrix.data());
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  for (auto elem : filterObjects)
  {
    pimFree(elem);
  }

  for (auto elem : matrixObjects)
  {
    pimFree(elem);
  }
}

// Params ---------------------------------------------------------------------
typedef struct Params
{
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./vgg [options]"
          "\n"
          "\n    -v    should verify result with CPU"
          "\n    -i    input file containing matrices (default=generates matrix with random numbers)"
          "\n    -c    input file containing dramsim config"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "c:v:i:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
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

void conv2(std::vector<std::vector<std::vector<int>>> &inputMatrix, std::vector<std::vector<std::vector<int>>> &kernelMatrix, std::vector<std::vector<std::vector<int>>> &resultMatrix, int stride, int padding, int imageSize, int kernelSize)
{
  // TODO: get number of columns after creating the device. Maybe support an API like getDeviceConfig.
  unsigned numCols = 8192, numOfCore = 4096;
  int kernelDim = kernelMatrix.size(), inputDim = inputMatrix.size();
  int outMatDim = kernelMatrix.size();
  int outMatSize = imageSize;
  int numOfMatPerRow = floor((1.0 * numCols * numOfCore) / (outMatSize * outMatSize)) < inputDim ? floor((1.0 * numCols * numOfCore) / (outMatSize * outMatSize)) : inputDim;
  int numOfPIMRow = kernelSize * kernelSize;

  resultMatrix.resize(outMatDim, std::vector<std::vector<int>>(outMatSize, std::vector<int>(outMatSize)));
  for (int i = 0; i < kernelDim; i++)
  {
    std::vector<int> dstVec(outMatSize * outMatSize);
    for (int j = 0; j < inputDim; j += numOfMatPerRow)
    {
      int matChunk = (numOfMatPerRow + j) <= inputDim ? (numOfMatPerRow + j) : inputDim;
      int tempcol = 0;
      std::vector<std::vector<int>> mergedMat(numOfPIMRow);
      for (int k = j; k < matChunk; ++k)
      {
        std::vector<std::vector<int>> decompMat;
        getDecomposedMatrix(imageSize, imageSize, kernelMatrix[i].size(), kernelMatrix[i][0].size(), stride, inputMatrix[k], decompMat);
        for (int idx = 0; idx < mergedMat.size(); idx++)
        {
          mergedMat[idx].reserve(mergedMat[idx].size() + decompMat[idx].size());
          mergedMat[idx].insert(mergedMat[idx].end(), make_move_iterator(decompMat[idx].begin()), make_move_iterator(decompMat[idx].end()));
          tempcol = mergedMat[idx].size();
        }
      }
      std::vector<int> outVector;
      performConv(kernelMatrix[i], mergedMat, outVector, numOfPIMRow, tempcol);

      auto start = std::chrono::high_resolution_clock::now();

      int hopSize = outMatSize * outMatSize;
      std::vector<int> localSums(hopSize, 0);

#pragma omp parallel
      {
        std::vector<int> threadLocalSums(hopSize, 0);

#pragma omp for
        for (int m = 0; m < hopSize; ++m)
        {
          for (int n = m + hopSize; n < outVector.size(); n += hopSize)
          {
            threadLocalSums[m] += outVector[n];
          }
        }

#pragma omp critical
        {
          for (int m = 0; m < hopSize; ++m)
          {
            localSums[m] += threadLocalSums[m];
          }
        }
      }

      for (int m = 0; m < hopSize; ++m)
      {
        dstVec[m] += localSums[m];
      }
      auto end = std::chrono::high_resolution_clock::now();
      hostElapsedTime += (end - start);
    }
    int ddx = 0;
    for (int rdx = 0; rdx < outMatSize; ++rdx)
    {
      for (int cdx = 0; cdx < outMatSize; ++cdx)
      {
        resultMatrix[i][rdx][cdx] = dstVec[ddx++];
      }
    }
  }
}

/*
  This should work for bitSIMD or any PIM that requires vertical data layout.
*/
void maxPool(const std::vector<std::vector<int>> &inputMatrix, std::vector<int> &outputMatrix)
{

  unsigned bitsPerElement = 32;

  if (inputMatrix.empty())
  {
    return;
  }
  int numRows = inputMatrix.size();
  int numCols = inputMatrix[0].size();

  std::vector<PimObjId> pimObjectList(numRows);
  PimObjId obj1 = pimAlloc(PIM_ALLOC_AUTO, numCols, bitsPerElement, PIM_INT32);
  if (obj1 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  pimObjectList[0] = obj1;
  for (int i = 1; i < numRows; i++)
  {
    PimObjId obj = pimAllocAssociated(bitsPerElement, pimObjectList[0], PIM_INT32);
    if (obj == -1)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
    pimObjectList[i] = obj;
  }

  for (int i = 0; i < pimObjectList.size(); i++)
  {
    PimStatus status = pimCopyHostToDevice((void *)inputMatrix[i].data(), pimObjectList[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
  }

  for (int i = 1; i < pimObjectList.size(); i++)
  {
    PimStatus status = pimMax(pimObjectList[0], pimObjectList[i], pimObjectList[0]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
  }
  outputMatrix.resize(numCols);
  PimStatus status = pimCopyDeviceToHost(pimObjectList[0], outputMatrix.data());
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  for (auto elem : pimObjectList)
  {
    pimFree(elem);
  }
}

void getDecomposedMatrixPool(int matrixRow, int matrixColumn, int kernelSize, int stride, const std::vector<std::vector<int>> &inputMatrix, std::vector<std::vector<int>> &decompMatrix)
{

  int numRows = kernelSize * kernelSize;
  // The following won't work if the kernel is not square and if stride != kernel size
  int numCols = matrixRow * matrixColumn / numRows;
  decompMatrix.resize(numRows, std::vector<int>(numCols, 0));
  int colIdx = 0;
  for (int i = 0; i < matrixRow - kernelSize + 1; i += stride)
  {
    for (int j = 0; j < (matrixColumn - kernelSize + 1); j += stride)
    {
      int rowIDX = 0;
      for (int k = i; k < i + kernelSize; k++)
      {
        for (int l = j; l < j + kernelSize; l++)
        {
          decompMatrix[rowIDX++][colIdx] = inputMatrix[k][l];
        }
      }
      ++colIdx;
    }
  }
}

void pool(std::vector<std::vector<std::vector<int>>> &inputMatrix, int kernelSize, int stride, int imageSize, std::vector<std::vector<std::vector<int>>> &resultMatrix)
{
  unsigned numCols = 8192, numOfCore = 4096;

  // TODO: currently considers square shape kernel. But it could be rectangle. In that case take kernel row and column as an input and modify this code accordingly.
  int numOfPIMRow = kernelSize * kernelSize;
  int numOfPIMColumn = imageSize * imageSize / numOfPIMRow;
  int dim = inputMatrix.size();
  int numOfMatPerRow = floor((1.0 * numCols * numOfCore) / numOfPIMColumn) < dim ? floor((1.0 * numCols * numOfCore) / (numOfPIMColumn)) : dim;

  cout << "Input Dim: " << dim << endl;

  // TODO: this won't work for all the cases but will work for vgg
  resultMatrix.resize(dim, std::vector<std::vector<int>>(imageSize / kernelSize, std::vector<int>(imageSize / kernelSize)));

  for (int i = 0; i < dim; i += numOfMatPerRow)
  {
    // This vector packs all the matrices that can be fit into one PIM iteration
    std::vector<std::vector<int>> mergedMat(numOfPIMRow);
    int matChunk = (numOfMatPerRow + i) <= dim ? (numOfMatPerRow + i) : dim;
    for (int j = i; j < matChunk; j++)
    {
      std::vector<std::vector<int>> tempMat;
      getDecomposedMatrixPool(imageSize, imageSize, kernelSize, stride, inputMatrix[j], tempMat);
      for (int idx = 0; idx < mergedMat.size(); idx++)
      {
        mergedMat[idx].reserve(mergedMat[idx].size() + tempMat[idx].size());
        mergedMat[idx].insert(mergedMat[idx].end(), make_move_iterator(tempMat[idx].begin()), make_move_iterator(tempMat[idx].end()));
      }
    }
    std::vector<int> outMatrix;
    maxPool(mergedMat, outMatrix);
    int idx = 0;
    for (int j = i; j < matChunk; ++j)
    {
      for (int r = 0; r < resultMatrix[j].size(); ++r)
      {
        for (int c = 0; c < resultMatrix[j][r].size(); ++c)
        {
          resultMatrix[j][r][c] = outMatrix[idx++];
        }
      }
    }
  }
}

void gemv(uint64_t row, uint64_t col, std::vector<int> &srcVector, std::vector<std::vector<int>> &srcMatrix, std::vector<int> &dst)
{
  unsigned bitsPerElement = sizeof(int) * 8;
  PimObjId srcObj1 = pimAlloc(PIM_ALLOC_AUTO, row, bitsPerElement, PIM_INT32);
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

  PimStatus status = pimBroadcastSignedInt(dstObj, 0);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  for (int i = 0; i < col; ++i)
  {
    status = pimCopyHostToDevice((void *)srcMatrix[i].data(), srcObj1);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }

    status = pimBroadcastSignedInt(srcObj2, srcVector[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }

    status = pimMul(srcObj1, srcObj2, srcObj2);
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

  dst.resize(row);
  status = pimCopyDeviceToHost(dstObj, (void *)dst.data());
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
  }
  pimFree(srcObj1);
  pimFree(srcObj2);
  pimFree(dstObj);
}

void softmax(const std::vector<int> &input, std::vector<int> &output)
{
    // Find the maximum value in the input vector for numerical stability
    int max_input = *std::max_element(input.begin(), input.end());

    // Compute the exponentials of each element (subtracting the max value for stability)
    std::vector<double> exponentials(input.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i)
    {
        exponentials[i] = std::exp(static_cast<double>(input[i] - max_input));
    }

    // Compute the sum of exponentials
    double sum_exponentials = 0.0;
    
    #pragma omp parallel for reduction(+:sum_exponentials)
    for (size_t i = 0; i < exponentials.size(); ++i)
    {
        sum_exponentials += exponentials[i];
    }

    // Compute the softmax values
    output.resize(input.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i)
    {
        output[i] = exponentials[i] / sum_exponentials;
    }
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::vector<std::vector<std::vector<int>>> inputMatrix;
  std::vector<std::vector<std::vector<int>>> kernelMatrix;

  if (params.inputFile == nullptr)
  {
    inputMatrix.resize(3);
    for (int i = 0; i < 3; i++)
    {
      getMatrix(224, 224, 1, inputMatrix[i]);
    }
    kernelMatrix.resize(64);
    for (auto &mat : kernelMatrix)
    {
      getMatrix(3, 3, 0, mat);
    }
  }
  else
  {
    // TODO: read Matrix from file
  }

  if (!createDevice(params.configFile))
    return 1;

  // conv1-1
  std::cout << "........starting conv1-1........\n";
  std::vector<std::vector<std::vector<int>>> resultMatrix1;
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 1, 224, 3);
  std::cout << "........ending conv1-1........\n";

  // conv1-2
  kernelMatrix.clear();
  if (params.inputFile == nullptr)
  {
    kernelMatrix.resize(64);
    for (auto &mat : kernelMatrix)
    {
      getMatrix(3, 3, 0, mat);
    }
  }
  inputMatrix.clear();
  inputMatrix.resize(resultMatrix1.size());
  for (int i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(224, 224, 1, resultMatrix1[i], inputMatrix[i]);
  }

  std::cout << "........starting conv1-2........\n";
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 1, 224, 3);
  std::cout << "........ending conv1-2........\n";

  // pool
  std::cout << "........starting pooling........\n";
  std::vector<std::vector<std::vector<int>>> resultMatrix2;
  pool(resultMatrix1, 2, 2, 224, resultMatrix2);
  std::cout << "........ending pooling........\n";

  // conv2-1
  kernelMatrix.clear();
  if (params.inputFile == nullptr)
  {
    kernelMatrix.resize(128);
    for (auto &mat : kernelMatrix)
    {
      getMatrix(3, 3, 0, mat);
    }
  }
  inputMatrix.clear();
  inputMatrix.resize(resultMatrix2.size());
  for (int i = 0; i < resultMatrix2.size(); ++i)
  {
    addPadding(112, 112, 1, resultMatrix2[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv2-1........\n";
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 1, 112, 3);
  std::cout << "........ending conv2-1........\n";

  // conv2-2
  kernelMatrix.clear();
  if (params.inputFile == nullptr)
  {
    kernelMatrix.resize(128);
    for (auto &mat : kernelMatrix)
    {
      getMatrix(3, 3, 0, mat);
    }
  }
  inputMatrix.clear();
  inputMatrix.resize(128);
  for (int i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(112, 112, 1, resultMatrix1[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv2-2........\n";
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 1, 112, 3);
  std::cout << "........ending conv2-2........\n";

  // pool

  std::cout << "........starting pooling........\n";
  resultMatrix2.clear();
  pool(resultMatrix1, 2, 2, 112, resultMatrix2);
  std::cout << "........ending pooling........\n";

  // conv3-1
  kernelMatrix.clear();
  if (params.inputFile == nullptr)
  {
    kernelMatrix.resize(256);
    for (auto &mat : kernelMatrix)
    {
      getMatrix(3, 3, 0, mat);
    }
  }
  inputMatrix.clear();
  inputMatrix.resize(128);
  for (int i = 0; i < resultMatrix2.size(); ++i)
  {
    addPadding(56, 56, 1, resultMatrix2[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv3-1........\n";
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 1, 56, 3);
  std::cout << "........starting conv3-1........\n";

  // conv3-2
  kernelMatrix.clear();
  if (params.inputFile == nullptr)
  {
    kernelMatrix.resize(256);
    for (auto &mat : kernelMatrix)
    {
      getMatrix(3, 3, 0, mat);
    }
  }
  inputMatrix.clear();
  inputMatrix.resize(256);
  for (int i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(56, 56, 1, resultMatrix1[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv3-2........\n";
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 1, 56, 3);
  std::cout << "........ending conv3-2........\n";

  // pool
  resultMatrix2.clear();
  std::cout << "........starting pooling........\n";
  pool(resultMatrix1, 2, 2, 56, resultMatrix2);
  std::cout << "........ending pooling........\n";

  // conv4-1
  kernelMatrix.clear();
  if (params.inputFile == nullptr)
  {
    kernelMatrix.resize(512);
    for (auto &mat : kernelMatrix)
    {
      getMatrix(3, 3, 0, mat);
    }
  }
  inputMatrix.clear();
  inputMatrix.resize(256);
  for (int i = 0; i < resultMatrix2.size(); ++i)
  {
    addPadding(28, 28, 1, resultMatrix2[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv4-1........\n";
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 1, 28, 3);
  std::cout << "........ending conv4-1........\n";

  // conv4-2
  kernelMatrix.clear();
  if (params.inputFile == nullptr)
  {
    kernelMatrix.resize(512);
    for (auto &mat : kernelMatrix)
    {
      getMatrix(3, 3, 0, mat);
    }
  }
  inputMatrix.clear();
  inputMatrix.resize(512);
  for (int i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(28, 28, 1, resultMatrix1[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv4-2........\n";
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 1, 28, 3);
  std::cout << "........ending conv4-2........\n";

  // pool
  resultMatrix2.clear();
  std::cout << "........starting pooling........\n";
  pool(resultMatrix1, 2, 2, 28, resultMatrix2);
  std::cout << "........ending pooling........\n";

  // conv5-1
  kernelMatrix.clear();
  if (params.inputFile == nullptr)
  {
    kernelMatrix.resize(512);
    for (auto &mat : kernelMatrix)
    {
      getMatrix(3, 3, 0, mat);
    }
  }
  inputMatrix.clear();
  inputMatrix.resize(512);
  for (int i = 0; i < resultMatrix2.size(); ++i)
  {
    addPadding(14, 14, 1, resultMatrix2[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv5-1........\n";
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 1, 14, 3);
  std::cout << "........ending conv5-1........\n";

  // conv5-2
  kernelMatrix.clear();
  if (params.inputFile == nullptr)
  {
    kernelMatrix.resize(512);
    for (auto &mat : kernelMatrix)
    {
      getMatrix(3, 3, 0, mat);
    }
  }
  inputMatrix.clear();
  inputMatrix.resize(512);
  for (int i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(14, 14, 1, resultMatrix1[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv5-2........\n";
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 1, 14, 3);
  std::cout << "........ending conv5-2........\n";

  // pool
  resultMatrix2.clear();
  std::cout << "........starting pooling........\n";
  pool(resultMatrix1, 2, 2, 14, resultMatrix2);
  std::cout << "........ending pooling........\n";

  // dense layer 1
  std::vector<int> flattenedMat;
  flatten3DMat(resultMatrix2, flattenedMat);
  std::vector<std::vector<int>> desnseWeight;
  std::vector<int> denseOutput1;
  if (params.inputFile == nullptr)
  {
    getMatrix(25088, 4096, 0, desnseWeight);
  }
  std::cout << "........starting dense1........\n";
  gemv(4096, 25088, flattenedMat, desnseWeight, denseOutput1);
  std::cout << "........ending dense1........\n";

  // dense layer 2
  desnseWeight.clear();
  std::vector<int> denseOutput2;
  if (params.inputFile == nullptr)
  {
    getMatrix(4096, 4096, 0, desnseWeight);
  }
  std::cout << "........starting dense2........\n";
  gemv(4096, 4096, denseOutput1, desnseWeight, denseOutput2);
  std::cout << "........ending dense2........\n";

  // dense layer 3
  desnseWeight.clear();
  std::vector<int> denseOutput3;
  if (params.inputFile == nullptr)
  {
    getMatrix(1000, 4096, 0, desnseWeight);
  }
  std::cout << "........starting dense3........\n";
  gemv(4096, 1000, denseOutput2, desnseWeight, denseOutput3);
  std::cout << "........ending dense3........\n";

  // perform softmax in host
  std::vector<int> resultVector;
  auto start = std::chrono::high_resolution_clock::now();
  softmax(denseOutput3, resultVector);
  auto end = std::chrono::high_resolution_clock::now();
  hostElapsedTime += (end - start);

  pimShowStats();
  cout << "Host elapsed time: " << std::fixed << std::setprecision(3) << hostElapsedTime.count() << " ms." << endl;

  return 0;
}
