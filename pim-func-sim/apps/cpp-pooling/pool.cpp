// Test: C++ version of max pool. This works for vgg max pool. The code may not work if the matrix is not square.
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "libpimsim.h"
#include <iostream>
#include <vector>
#include <getopt.h>
#include <cmath>
#include "../util.h"

using namespace std;

void getDecomposedMatrix(int matrixRow, int matrixColumn, int kernelSize, int stride, const std::vector<std::vector<int>> &inputMatrix, std::vector<std::vector<int>> &decompMatrix)
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
  PimObjId obj1 = pimAlloc(PIM_ALLOC_V1, numCols, bitsPerElement, PIM_INT32);
  if (obj1 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  pimObjectList[0] = obj1;
  for (int i = 1; i < numRows; i++)
  {
    PimObjId obj = pimAllocAssociated(PIM_ALLOC_V1, numCols, bitsPerElement, pimObjectList[0], PIM_INT32);
    if (obj == -1)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
    pimObjectList[i] = obj;
  }

  for (int i = 0; i < pimObjectList.size(); i++)
  {
    PimStatus status = pimCopyHostToDevice(PIM_COPY_V, (void *)inputMatrix[i].data(), pimObjectList[i]);
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
  PimStatus status = pimCopyDeviceToHost(PIM_COPY_V, pimObjectList[0], outputMatrix.data());
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

// Params ---------------------------------------------------------------------
typedef struct Params
{
  int row, column, dim, stride, kernelSize;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./pool [options]"
          "\n"
          "\n    -r    row (default=224)"
          "\n    -c    column (default=224)"
          "\n    -d    dimension (default=64)"
          "\n    -d    dimension (default=64)"
          "\n    -s    stride (default=2)"
          "\n    -k    kernel size (default=2X2)"
          "\n    -v    should verify result with CPU"
          "\n    -f    input file containing matrices (default=generates matrix with random numbers)"
          "\n    -i    input file containing matrices (default=generates matrix with random numbers)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.row = 224;
  p.column = 224;
  p.dim = 64;
  p.row = 224;
  p.column = 224;
  p.dim = 64;
  p.stride = 2;
  p.kernelSize = 2;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:r:c:d:s:k:v:f:i")) >= 0)
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

// Function to perform max pooling (VGG style)
std::vector<std::vector<int>> maxPoolingVGG(const std::vector<std::vector<int>> &inputMatrix)
{
  int numRows = inputMatrix.size();
  int numCols = inputMatrix[0].size();

  // Initialize the output matrix with zeros
  std::vector<std::vector<int>> outputMatrix(numRows / 2, std::vector<int>(numCols / 2, 0));

  // Perform max pooling (VGG style) with pool size 2x2 and stride 2
  for (int i = 0; i < numRows; i += 2)
  {
    for (int j = 0; j < numCols; j += 2)
    {
      int maxVal = std::max(inputMatrix[i][j], std::max(inputMatrix[i][j + 1],
                                                        std::max(inputMatrix[i + 1][j], inputMatrix[i + 1][j + 1])));

      // Assign the maximum value to the corresponding position in the output matrix
      outputMatrix[i / 2][j / 2] = maxVal;
    }
  }

  return outputMatrix;
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);

  std::vector<std::vector<std::vector<int>>> inputMatrix;
  inputMatrix.resize(params.dim, std::vector<std::vector<int>>(params.row, std::vector<int>(params.column)));

  if (params.inputFile == nullptr)
  {
    for (auto &mat : inputMatrix)
    {
      getMatrix(params.row, params.column, 0, mat);
    }
  }
  else
  {
    // TODO: read Matrix from file
  }

  
  if (!createDevice(params.configFile)) return 1;
  
  // TODO: get number of columns after creating the device. Maybe support an API like getDeviceConfig. Besides 65536 is too large.
  unsigned numCols = 65536;

  // TODO: currently considers square shape kernel. But it could be rectangle. In that case take kernel row and column as an input and modify this code accordingly.
  // TODO: currently considers square shape kernel. But it could be rectangle. In that case take kernel row and column as an input and modify this code accordingly.
  int numOfPIMRow = params.kernelSize * params.kernelSize;
  int numOfPIMColumn = params.row * params.column / numOfPIMRow;
  int numOfMatPerRow = floor((1.0 * numCols) / numOfPIMColumn) < params.dim ? floor((1.0 * numCols) / (numOfPIMColumn)) : params.dim;

  cout << "Matrix per core: " << numOfMatPerRow << endl;

  // TODO: this won't work for all the cases but will work for vgg
  std::vector<std::vector<std::vector<int>>> resultMatrix;
  resultMatrix.resize(params.dim, std::vector<std::vector<int>>(params.row / params.kernelSize, std::vector<int>(params.column / params.kernelSize)));
  
  
  for (int i = 0; i < params.dim; i += numOfMatPerRow)
  {
    // This vector packs all the matrices that can be fit into one PIM iteration
    std::vector<std::vector<int>> mergedMat(numOfPIMRow);
    int matChunk = (numOfMatPerRow + i) <= params.dim ? (numOfMatPerRow + i) : params.dim;
    for (int j = i; j < matChunk; j++)
    {
      std::vector<std::vector<int>> tempMat;
      getDecomposedMatrix(params.row, params.column, params.kernelSize, params.stride, inputMatrix[j], tempMat);
      for (int idx = 0; idx < mergedMat.size(); idx++)
      for (int idx = 0; idx < mergedMat.size(); idx++)
      {
        mergedMat[idx].reserve(mergedMat[idx].size() + tempMat[idx].size());
        mergedMat[idx].insert(mergedMat[idx].end(), make_move_iterator(tempMat[idx].begin()), make_move_iterator(tempMat[idx].end()));
        mergedMat[idx].reserve(mergedMat[idx].size() + tempMat[idx].size());
        mergedMat[idx].insert(mergedMat[idx].end(), make_move_iterator(tempMat[idx].begin()), make_move_iterator(tempMat[idx].end()));
      }
    }
    std::vector<int> outMatrix;
    maxPool(mergedMat, outMatrix);
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
    if (params.shouldVerify)
    {
      for (int j = i; j < matChunk; ++j)
      {
        std::vector<std::vector<int>> cpuPoolMat = maxPoolingVGG(inputMatrix[j]);
        for (size_t tdx = 0; tdx < cpuPoolMat.size(); ++tdx)
        {
          for (size_t cdx = 0; cdx < cpuPoolMat[tdx].size(); ++cdx)
          {
            if (cpuPoolMat[tdx][cdx] != resultMatrix[j][tdx][cdx])
            {
              std::cout << "Did not matched." << j << " Actual: " << cpuPoolMat[tdx][cdx] << "\tGot: " << resultMatrix[j][tdx][cdx] << "\n";
            }
          }
        }
      }
    }
  }

  pimShowStats();

  return 0;
}
