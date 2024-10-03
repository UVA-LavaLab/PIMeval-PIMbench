// Test: C++ version of max pool
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include <iostream>
#include <vector>
#include <getopt.h>
#include <cmath>
#include "../../util.h"

using namespace std;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  int row, column, dim, stride, padding, kernelHeight, kernelWidth;
  char *dramConfigFile;
  char *imageMatrixFile;
  bool shouldVerify;
  bool moreDebugPrints;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./pool.out [options]"
          "\n"
          "\n    -r    row (default=224)"
          "\n    -c    column (default=224)"
          "\n    -d    dimension (default=64)"
          "\n    -s    stride (default=2)"
          "\n    -p    input padding (default=0)"
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
  p.padding = 0;
  p.kernelHeight = 2;
  p.kernelWidth = 2;
  p.dramConfigFile = nullptr;
  p.imageMatrixFile = nullptr;
  p.shouldVerify = false;
  p.moreDebugPrints = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:r:c:d:s:p:l:w:v:f:i:m:")) >= 0)
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
    case 'p':
      p.padding = atoi(optarg);
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

// Decompose the input matrix by sliding the kernel dimensions (kernelHeight * kernelWidth) along the input matrix with a stride.
// Assume the input matrix is padded.
void decomposeMatrix(int matrixRow, int matrixColumn, int kernelHeight, int kernelWidth, int stride, int padding, const std::vector<std::vector<int>> &inputMatrix, std::vector<std::vector<int>> &decompMatrix)
{
  // Calculate the number of rows and columns for the decomposed matrix
  int numRows = kernelHeight * kernelWidth;
  int numCols = ((matrixRow - kernelHeight + 2 * padding) / stride + 1) * ((matrixColumn - kernelWidth + 2 * padding) / stride + 1);  
  // Initialize the decomposed matrix with the correct size
  decompMatrix.resize(numRows, std::vector<int>(numCols, 0));

  int colIdx = 0;
  for (int i = 0; i < (matrixRow + 2 * padding - kernelHeight + 1); i += stride)
  {
    for (int j = 0; j < (matrixColumn + 2 * padding - kernelWidth + 1); j += stride)
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
  if (inputMatrix.empty())
  {
    std::cerr << "Abort: input matrix is empty" << std::endl;
    exit(1);
  }
  int numRows = inputMatrix.size();
  int numCols = inputMatrix[0].size();

  std::vector<PimObjId> pimObjectList(numRows);
  PimObjId obj1 = pimAlloc(PIM_ALLOC_AUTO, numCols, PIM_INT32);
  if (obj1 == -1)
  {
    std::cerr << "Abort: pimAlloc failed for obj1" << std::endl;
    exit(1);
  }
  pimObjectList[0] = obj1;
  for (int i = 1; i < numRows; i++)
  {
    PimObjId obj = pimAllocAssociated(pimObjectList[0], PIM_INT32);
    if (obj == -1)
    {
      std::cerr << "Abort: pimAllocAssociated failed pimObjectList at iteration: " << i << std::endl;
      exit(1);
    }
    pimObjectList[i] = obj;
  }

  for (int i = 0; i < pimObjectList.size(); i++)
  {
    PimStatus status = pimCopyHostToDevice((void *)inputMatrix[i].data(), pimObjectList[i]);
    if (status != PIM_OK)
    {
      std::cerr << "Abort: pimCopyHostToDevice failed between inputMatrix and pimObjectList at iteration: " << i << std::endl;
      exit(1);
    }
  }

  for (int i = 1; i < pimObjectList.size(); i++)
  {
    PimStatus status = pimMax(pimObjectList[0], pimObjectList[i], pimObjectList[0]);
    if (status != PIM_OK)
    {
      std::cerr << "Abort: pimMax failed between pimObjectLists at iteration: " << i << std::endl;
      exit(1); 
    }
  }
  outputMatrix.resize(numCols);
  PimStatus status = pimCopyDeviceToHost(pimObjectList[0], outputMatrix.data());
  if (status != PIM_OK)
  {
    std::cerr << "Abort: pimCopyDeviceToHost failed between pimObjectList and outputMatrix" << std::endl;
    exit(1);
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

  if (moreDebugPrints == true) {
    std::cout << "Stride: " << stride << ", Kernel size: " << kernelHeight << "x" << kernelWidth << std::endl;
    std::cout << "Input matrix:" << std::endl;
    printMatrix(inputMatrix);
    std::cout << "Output matrix from CPU:" << std::endl;
    printMatrix(outputMatrix);
    std::cout << "Output matrix from PIM:" << std::endl;
    printMatrix(PIMResult);    
  }

  if (mismatch_counter == 0) {
    std::cout << "Success: PIM results match with CPU results" << std::endl; 
  } else {
    std::cerr << "Failure: PIM results do not match with CPU results" << std::endl;
    exit(1);
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
      getMatrix(params.row, params.column, params.padding, mat);
    }
  }
  else
  {
    std::cerr << "Reading from the input file is not implemented yet for the input matrix" << std::endl;
    exit(1);
  }

  if (!createDevice(params.dramConfigFile))
    exit(1);

  PimDeviceProperties deviceProp;
  PimStatus status = pimGetDeviceProperties(&deviceProp);
  if (status != PIM_OK) {
    std::cerr << "Abort: pimGetDeviceProperties failed" << std::endl;
    exit(1);
  }
  // Get the device parameters
  uint64_t numCols = deviceProp.numColPerSubarray;
  uint64_t numRows = deviceProp.numRowPerSubarray;
  uint64_t numOfBits = uint64_t(deviceProp.numRanks) * uint64_t(deviceProp.numBankPerRank) * uint64_t(deviceProp.numSubarrayPerBank) * numCols * numRows; 

  uint64_t inputHeight = inputMatrix[0].size();
  uint64_t inputWidth = inputMatrix[0][0].size();
  uint64_t outputHeight = (inputHeight - params.kernelHeight + 2 * params.padding) / params.stride + 1;
  uint64_t outputWidth = (inputWidth - params.kernelWidth + 2 * params.padding) / params.stride + 1;

  uint64_t numOfPIMRow = params.kernelHeight * params.kernelWidth;
  uint64_t numOfPIMColumn = outputHeight * outputWidth;
  uint64_t numOfMatPerRow = floor((1.0 * numOfBits) / numOfPIMColumn) < params.dim ? floor((1.0 * numOfBits) / (numOfPIMColumn)) : params.dim;

  std::vector<std::vector<std::vector<int>>> resultMatrix;
  resultMatrix.resize(params.dim, std::vector<std::vector<int>>(outputHeight, std::vector<int>(outputWidth)));

  for (uint64_t i = 0; i < params.dim; i += 1)
  {
    // This vector packs all the matrices that can be fit into one PIM iteration
    std::vector<std::vector<int>> mergedMat(numOfPIMRow);
    uint64_t matChunk = (numOfMatPerRow + i) <= params.dim ? (numOfMatPerRow + i) : params.dim;
    for (uint64_t j = i; j < matChunk; j++)
    {
      std::vector<std::vector<int>> decompMat;
      decomposeMatrix(params.row, params.column, params.kernelHeight, params.kernelWidth, params.stride, params.padding, inputMatrix[j], decompMat);
      if (params.moreDebugPrints == true) { 
        // Debug print
        std::cout << "Decomposed Matrix:" << std::endl;
        printMatrix(decompMat);
      }
      // Merge the matrices
      for (uint64_t idx = 0; idx < mergedMat.size(); idx++) {
        mergedMat[idx].insert(mergedMat[idx].end(),
                             std::make_move_iterator(decompMat[idx].begin()),
                             std::make_move_iterator(decompMat[idx].end()));
      }
    }

    std::vector<int> outVector;
    if (params.moreDebugPrints == true) { 
      // Debug print
      std::cout << "Merged matrix:" << std::endl;
      printMatrix(mergedMat);
    }    
    maxPool(mergedMat, outVector);
    if (params.moreDebugPrints == true) { 
      // Debug print
      std::cout << "outVector:" << std::endl;
      printVector(outVector);
    }

    uint64_t idx = 0;
    for (uint64_t j = i; j < matChunk; ++j)
    {
      for (uint64_t r = 0; r < outputHeight; ++r)
      {
        for (uint64_t c = 0; c < outputWidth; ++c)
        {
          resultMatrix[j][r][c] = outVector[idx++];
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
