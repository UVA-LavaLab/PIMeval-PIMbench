// Test: C++ version of RELU activation function: max(0, x) in batches.
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include <iostream>
#include <vector>
#include <getopt.h>
#include <cmath>
#include "../util.h"

using namespace std;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  int row, column, dim, batchSize;
  char *imageMatrixFile;
  char *dramConfigFile;
  bool shouldVerify;
  bool moreDebugPrints;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./relu_batch.out [options]"
          "\n"
          "\n    -r    row (default=224)"
          "\n    -c    column (default=224)"
          "\n    -d    dimension (default=64)"
          "\n    -b    batch size (Default=2)"
          "\n    -v    should verify result with CPU"
          "\n    -i    input image file containing matrices (default=generates matrix with random numbers)"
          "\n    -o    DRAM config file (default = nullptr)"
          "\n    -m    enable more debug prints (default = false)"          
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.row = 226;
  p.column = 226;
  p.dim = 128;
  p.batchSize = 2;
  p.imageMatrixFile = nullptr;
  p.dramConfigFile = nullptr;
  p.shouldVerify = false;
  p.moreDebugPrints = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:r:c:d:b:v:o:i:m:")) >= 0)
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
    case 'b':
      p.batchSize = atoi(optarg);
      break;  
    case 'i':
      p.imageMatrixFile = optarg;
      break;
    case 'v':
      p.shouldVerify = (*optarg == 't') ? true : false;
      break;
    case 'o':
      p.dramConfigFile = optarg;
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
void performRelu(const std::vector<std::vector<int>> &inputMatrix, std::vector<int> &outputMatrix)
{
  unsigned bitsPerElement = 32;

  if (inputMatrix.empty())
  {
    std::cout << "Function: " << __func__ << ", Abort: Input matrix is empty" << std::endl;
    return;   
  }
  int numRows = inputMatrix.size();
  int numCols = inputMatrix[0].size();
  // Initialize reluConst with zero for max(0, x) operation. Initialize with a different value 'y' for max(y, x) operation.
  uint64_t reluConst = 0;  

  std::vector<PimObjId> pimObjectList(numRows);
  PimObjId obj1 = pimAlloc(PIM_ALLOC_AUTO, numCols, bitsPerElement, PIM_INT32);
  if (obj1 == -1)
  {
    std::cout << "Abort: pimAlloc for PimObj obj1 failed" << std::endl;
    return;
  }
  pimObjectList[0] = obj1;
  for (int i = 1; i < numRows; i++)
  {
    PimObjId obj = pimAllocAssociated(bitsPerElement, pimObjectList[0], PIM_INT32);
    if (obj == -1)
    {
      std::cout << "Abort: pimAllocAssociated for PimObj obj failed" << std::endl;
      return;
    }
    pimObjectList[i] = obj;
  }
  
  for (int i = 0; i < pimObjectList.size(); i++)
  {
    PimStatus status = pimCopyHostToDevice((void *)inputMatrix[i].data(), pimObjectList[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort: pimCopyHostToDevice from inputMatrix[" << i << "] to pimObjectList[" << i << "] failed" << std::endl;
      return;
    }
  }

  for (int i = 0; i < pimObjectList.size(); i++)
  {
    PimStatus status = pimMaxScalar(pimObjectList[i], pimObjectList[0], reluConst);
    if (status != PIM_OK)
    {
      std::cout << "Abort: pimMax failed between RELUConstObj and pimObjectList[" << i << "]" << std::endl;
      return;
    }
  }
  outputMatrix.resize(numCols);
  PimStatus status = pimCopyDeviceToHost(pimObjectList[0], outputMatrix.data());
  if (status != PIM_OK)
  {
    std::cout << "Abort: pimCopyDeviceToHost from pimObjectList[0] to outputMatrix" << std::endl;
    return;
  }
  for (auto elem : pimObjectList)
  {
    pimFree(elem);
  }
}

// Function to perform RELU on CPU with configurable kernel size and stride
std::vector<std::vector<std::vector<int>>> verifyWithCPU(std::vector<std::vector<std::vector<int>>> &inputMatrix, std::vector<std::vector<std::vector<int>>> &PIMResult, bool moreDebugPrints)
{
  int numDepth = inputMatrix.size();
  int numRows = inputMatrix[0].size();
  int numCols = inputMatrix[0][0].size();
  
  // Initialize the output matrix with zeros
  std::vector<std::vector<std::vector<int>>> outputMatrix(numDepth, std::vector<std::vector<int>>(numRows, std::vector<int>(numCols, 0)));
       
  int mismatch_counter = 0;
  // Perform RELU with the specified kernel size and stride
  for (int d = 0; d < numDepth; ++d) {
    for (int i = 0; i < numRows; ++i) {
      for (int j = 0; j < numCols; ++j) {
        outputMatrix[d][i][j] = std::max(0, inputMatrix[d][i][j]);
        if (outputMatrix[d][i][j] != PIMResult[d][i][j]) {
          std::cout << "Mismatch between PIM and CPU results at depth: " << d << ", row: " << i << ", column: " << j << std::endl;
          mismatch_counter += 1;
        }        
      }
    }
  }
   
  if (mismatch_counter == 0) {
    std::cout << "Success: PIM results match with CPU results" << std::endl << std::endl; 
  } else {
    std::cout << "Failure: PIM results do not match with CPU results" << std::endl << std::endl;
  }

  if (moreDebugPrints == true) {
    std::cout << "Input matrix:" << std::endl;
    printMatrix(inputMatrix);
    std::cout << "Output matrix from CPU:" << std::endl;
    printMatrix(outputMatrix);
    std::cout << "Output matrix from PIM:" << std::endl;
    printMatrix(PIMResult);    
  }
  
  return outputMatrix;
}

int main(int argc, char *argv[]) {

  // Parse input parameters from command line
  Params params = getInputParams(argc, argv);

  // Initialize input matrix based on parsed dimensions
  std::vector<std::vector<std::vector<int>>> inputMatrix(params.dim, std::vector<std::vector<int>>(params.row, std::vector<int>(params.column * params.batchSize)));

  // Check if an image matrix file is provided
  if (params.imageMatrixFile == nullptr) {
    // Generate or retrieve matrix data if no file is provided
    for (auto &mat : inputMatrix) {
      getMatrix(params.row, (params.column * params.batchSize), 0, mat);
    }
  } else {
    std::cout << "Reading from input file is not implemented yet." << std::endl;
    return 0;    
  }

  // Create a device based on provided DRAM configuration file
  if (!createDevice(params.dramConfigFile))
    return 1;


  // Calculate matrix dimensions
  int inputDepth = inputMatrix.size();
  int inputHeight = inputMatrix[0].size();
  int inputWidth = inputMatrix[0][0].size();

  // Define parameters for processing
  PimDeviceProperties deviceProp;
  PimStatus status = pimGetDeviceProperties(&deviceProp);
  if (status != PIM_OK) {
    std::cout << "Abort: pimGetDeviceProperties failed" << std::endl;
    return 1;
  }
  // Get the device parameters
  uint64_t numCols = deviceProp.numColPerSubarray;
  uint64_t numRows = deviceProp.numRowPerSubarray;
  uint64_t numOfBits = uint64_t(deviceProp.numRanks) * uint64_t(deviceProp.numBankPerRank) * uint64_t(deviceProp.numSubarrayPerBank) * numCols * numRows;

  uint64_t numOfPIMRow = 1;
  uint64_t numOfMatPerRow = std::min(static_cast<uint64_t>(std::floor((1.0 * numOfBits) / (inputHeight * inputWidth * 32))), static_cast<uint64_t>(inputWidth));

  // Initialize result matrix with the same dimensions as input matrix
  std::vector<std::vector<std::vector<int>>> resultMatrix(inputDepth, std::vector<std::vector<int>>(inputHeight, std::vector<int>(inputWidth)));

  int tempcol = 0;
  std::vector<std::vector<int>> decompMat;
  std::vector<std::vector<int>> mergedMat(numOfPIMRow);
  std::vector<int> outVector;
  outVector.resize(inputDepth * inputHeight * inputWidth);
  
  // Loop through input depth in chunks
  for (uint64_t j = 0; j < inputDepth; j += numOfMatPerRow) {
    uint64_t matChunk = (numOfMatPerRow + j) <= inputDepth ? (numOfMatPerRow + j) : inputDepth;
    // Decompose and merge matrices
    for (uint64_t k = j; k < matChunk; k++) {
      decomposeMatrix(inputHeight, inputWidth, 1, 1, 1, 0, inputMatrix[k], decompMat);
      if (params.moreDebugPrints) { 
        // Debug print decomposed matrix
        std::cout << "Decomposed Matrix:" << std::endl;
        printMatrix(decompMat);
      }
      for (uint64_t idx = 0; idx < mergedMat.size(); idx++) {
        mergedMat[idx].reserve(mergedMat[idx].size() + decompMat[idx].size());
        mergedMat[idx].insert(mergedMat[idx].end(), make_move_iterator(decompMat[idx].begin()), make_move_iterator(decompMat[idx].end()));
        tempcol = mergedMat[idx].size();
      }
    }

    if (params.moreDebugPrints) {
      // Debug print merged matrix
      std::cout << "Merged Matrix:" << std::endl;            
      printMatrix(mergedMat);
    }
  }
  
  performRelu(mergedMat, outVector);
  if (params.moreDebugPrints) {
    // Debug print outVector
    std::cout << "outVector:" << std::endl;            
    printVector(outVector);    
  }

  uint64_t idx = 0;
  for (int i = 0; i < inputDepth; i += 1)
  {  
    for (int r = 0; r < inputHeight; ++r)
    {
      for (int c = 0; c < inputWidth; ++c)
        {
          resultMatrix[i][r][c] = outVector[idx++];
        }
    }
  }  
  
  // Verify results against CPU if specified
  if (params.shouldVerify) {
    verifyWithCPU(inputMatrix, resultMatrix, params.moreDebugPrints);
  }

  // Display PIM processing statistics
  pimShowStats();
  
  return 0;
}