// Test: C++ version of max pool. 
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "libpimsim.h"
#include <iostream>
#include <vector>
#include <getopt.h>
#include <cmath>
#include "../util.h"

using namespace std;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  int row, column, dim, stride, kernelHeight, kernelWidth;
  char *dramConfigFile;
  char *imageMatrixFile;
  bool shouldVerify;
  bool moreDebugPrints;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./pool [options]"
          "\n"
          "\n    -r    row (default=224)"
          "\n    -c    column (default=224)"
          "\n    -d    dimension (default=64)"
          "\n    -s    stride (default=2)"
          "\n    -l    kernel height (default=2)"
          "\n    -w    kernel width (default=2)"
          "\n    -v    should verify result with CPU"
          "\n    -f    input file containing kernel matrices (default=generates matrix with random numbers)"
          "\n    -i    input image file containing matrices (default=generates matrix with random numbers)"
	  "\n    -m    enable more debug prints (default = false)"          
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.row = 224;
  p.column = 224;
  p.dim = 64;
  p.stride = 2;
  p.kernelHeight = 2;
  p.kernelWidth = 2;
  p.dramConfigFile = nullptr;
  p.imageMatrixFile = nullptr;
  p.shouldVerify = false;
  p.moreDebugPrints = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:r:c:d:s:l:w:v:f:i:m:")) >= 0)
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
    case 'l':
      p.kernelHeight = atoi(optarg);
      break;
    case 'w':
      p.kernelWidth = atoi(optarg);
      break; 
    case 'f':
      p.dramConfigFile = optarg;
      break;
    case 'i':
      p.imageMatrixFile = optarg;
      break;
    case 'v':
      p.shouldVerify = (*optarg == 't') ? true : false;
      break;
    case 'm':
      p.moreDebugPrints = (*optarg == 't') ? true : false;
      break;  
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

// Function to print a 3D matrix
void print3DMatrix(const std::vector<std::vector<std::vector<int>>> &matrix) {
    for (int i = 0; i < matrix.size(); ++i) {
        std::cout << "Layer " << i << ":\n";
        for (int j = 0; j < matrix[i].size(); ++j) {
            for (int k = 0; k < matrix[i][j].size(); ++k) {
                std::cout << matrix[i][j][k] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

void getDecomposedMatrix(int matrixRow, int matrixColumn, int kernelHeight, int kernelWidth, int stride, const std::vector<std::vector<int>> &inputMatrix, std::vector<std::vector<int>> &decompMatrix)
{
  int numRows = kernelHeight * kernelWidth;
  int numCols = matrixRow * matrixColumn;
  decompMatrix.resize(numRows, std::vector<int>(numCols, 0));
  int colIdx = 0;
  for (int i = 0; i < (matrixRow - kernelHeight + 1); i += stride)
  {
    for (int j = 0; j < (matrixColumn - kernelWidth + 1); j += stride)
    {
      int rowIDX = 0;
      for (int k = i; k < i + kernelHeight; k++)
      {
        for (int l = j; l < j + kernelWidth; l++)
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

// Function to perform max pooling on CPU with configurable kernel size and stride
std::vector<std::vector<std::vector<int>>> verifyWithCPU(std::vector<std::vector<std::vector<int>>> &inputMatrix, std::vector<std::vector<std::vector<int>>> &PIMResult, int kernelHeight, int kernelWidth, int stride, bool moreDebugPrints)
{
  int numDepth = inputMatrix.size();
  int numRows = inputMatrix[0].size();
  int numCols = inputMatrix[0][0].size();
  
  // Calculate the dimensions of the output matrix
  int outputRows = (numRows - kernelHeight) / stride + 1;
  int outputCols = (numCols - kernelWidth) / stride + 1;

  // Initialize the output matrix with zeros
  std::vector<std::vector<std::vector<int>>> outputMatrix(numDepth, std::vector<std::vector<int>>(outputRows, std::vector<int>(outputCols, 0)));
       
  int mismatch_counter = 0;
  // Perform max pooling with the specified kernel size and stride
  for (int d = 0; d < numDepth; ++d) {
    for (int i = 0; i < outputRows; ++i) {
      for (int j = 0; j < outputCols; ++j) {
        int maxVal = std::numeric_limits<int>::min();
        for (int m = 0; m < kernelHeight; ++m) {
          for (int n = 0; n < kernelWidth; ++n) {
            int row = i * stride + m;
            int col = j * stride + n;
            if (row < numRows && col < numCols) {
              maxVal = std::max(maxVal, inputMatrix[d][row][col]);
            }
          }
        }
        outputMatrix[d][i][j] = maxVal;
        if (outputMatrix[d][i][j] != PIMResult[d][i][j]) {
          std::cout << "Mismatch between PIM and CPU results at depth: " << d << ", row: " << i << ", column: " << j << std::endl;
          mismatch_counter += 1;
        }
      }
    }
  }

  if (mismatch_counter == 0) {
    std::cout << "Success: PIM results match with CPU results" << std::endl; 
  } else {
    std::cout << "Failure: PIM results do not match with CPU results" << std::endl;
  }

  if (moreDebugPrints == true) {
    std::cout << "Stride: " << stride << ", Kernel size: " << kernelHeight << "x" << kernelWidth << std::endl;
    std::cout << "Input matrix:" << std::endl;
    print3DMatrix(inputMatrix);
    std::cout << "Output matrix from CPU:" << std::endl;
    print3DMatrix(outputMatrix);
    std::cout << "Output matrix from PIM:" << std::endl;
    print3DMatrix(PIMResult);    
  }
  
  return outputMatrix;
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);

  std::vector<std::vector<std::vector<int>>> inputMatrix;
  inputMatrix.resize(params.dim, std::vector<std::vector<int>>(params.row, std::vector<int>(params.column)));

  if (params.imageMatrixFile == nullptr)
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

  if (!createDevice(params.dramConfigFile))
    return 1;

  // TODO: get number of columns after creating the device. Maybe support an API like getDeviceConfig. Besides 65536 is too large.
  unsigned numCols = 8192, numOfCore = 4096;
  
  int numOfPIMRow = params.kernelHeight * params.kernelWidth;
  int numOfPIMColumn = (params.row * params.column / numOfPIMRow);
  int numOfMatPerRow = floor((1.0 * numCols * numOfCore) / numOfPIMColumn) < params.dim ? floor((1.0 * numCols * numOfCore) / (numOfPIMColumn)) : params.dim;

  int inputHeight = inputMatrix[0].size();
  int inputWidth = inputMatrix[0][0].size();
  int outputHeight = (inputHeight - params.kernelHeight) / params.stride + 1;
  int outputWidth = (inputWidth - params.kernelWidth) / params.stride + 1;
  
  std::vector<std::vector<std::vector<int>>> resultMatrix;
  resultMatrix.resize(params.dim, std::vector<std::vector<int>>(outputHeight, std::vector<int>(outputWidth)));

  for (int i = 0; i < params.dim; i += 1)
  {
    // This vector packs all the matrices that can be fit into one PIM iteration
    std::vector<std::vector<int>> mergedMat(numOfPIMRow);
    int matChunk = (numOfMatPerRow + i) <= params.dim ? (numOfMatPerRow + i) : params.dim;
    for (int j = i; j < matChunk; j++)
    {
      std::vector<std::vector<int>> tempMat;
      getDecomposedMatrix(params.row, params.column, params.kernelHeight, params.kernelWidth, params.stride, inputMatrix[j], tempMat);
      if (params.moreDebugPrints == true) { 
        // Debug print
        std::cout << "Decomposed Matrix:" << std::endl;
        for (const auto &row : tempMat)
        {
          for (const auto &val : row)
          {
            std::cout << val << " ";
          }
          std::cout << std::endl;
        }
      }
      for (int idx = 0; idx < mergedMat.size(); idx++) {
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

  if (params.shouldVerify == true)
  {
    // Perform max pooling on CPU and compare results with PIM
    verifyWithCPU(inputMatrix, resultMatrix, params.kernelHeight, params.kernelWidth, params.stride, params.moreDebugPrints);
  }
  
  pimShowStats();

  return 0;
}
