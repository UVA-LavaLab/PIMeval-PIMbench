// Test: C++ version of convolution in batches
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <getopt.h>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "../util.h"
#include <iomanip>
#include <chrono>

using namespace std;
typedef vector<vector<vector<int>>> Image3D;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  int row, column, dim, stride, kernelSize, kernelDim, padding, batchSize;
  char *kernelMatrixFile;
  char *imageMatrixFile;
  char *dramConfigFile;
  bool shouldVerify;
  bool moreDebugPrints;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./conv-batch.out [options]"
          "\n"
          "\n    -r    row (default=224)"
          "\n    -c    column (default=224)"
          "\n    -d    dimension (default=3)"
          "\n    -s    stride (default=1)"
          "\n    -k    kernel size (default=3X3)"
          "\n    -z    kernel dimension (default=64)"
          "\n    -b    batch size (default=2)"
          "\n    -v    should verify result with CPU"
          "\n    -p    padding (default = 1)"
          "\n    -f    input file containing kernel matrices (default=generates matrix with random numbers)"
          "\n    -i    input file containing image matrices (default=generates matrix with random numbers)"
          "\n    -o    DRAM config file (default = false)"
          "\n    -m    enable more debug prints (default = false)"
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
  p.kernelMatrixFile = nullptr;
  p.imageMatrixFile = nullptr;
  p.dramConfigFile = nullptr;
  p.shouldVerify = false;
  p.kernelDim = 64;
  p.batchSize = 2;
  p.padding = 1;
  p.moreDebugPrints = false;

  int opt;
  while ((opt = getopt(argc, argv, "r:c:d:s:k:v:z:b:f:o:i:p:m:")) >= 0)
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
    case 'b':
      p.batchSize = atoi(optarg);
      break;  
    case 'p':
      p.padding = atoi(optarg);
      break;
    case 'f':
      p.kernelMatrixFile = optarg;
      break;
    case 'i':
      p.imageMatrixFile = optarg;
      break;
    case 'o':
      p.dramConfigFile = optarg;
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
          decompMatrix[rowIDX++][colIdx] = inputMatrix[k][l];
        }
      }
      ++colIdx;
    }
  }
}

void performConv(std::vector<std::vector<int>> &filterMatrix, std::vector<std::vector<int>> &inputMatrix, std::vector<int> &outputMatrix, int numRequiredPIMRows, int numRequiredPIMCol, bool moreDebugPrints)
{
  std::vector<PimObjId> filterObjects;
  std::vector<int> temp;
  PimObjId obj1 = pimAlloc(PIM_ALLOC_AUTO, numRequiredPIMCol, PIM_INT32);
  if (obj1 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  filterObjects.push_back(obj1);
  for (int i = 1; i < numRequiredPIMRows; i++)
  {
    PimObjId obj = pimAllocAssociated(filterObjects[0], PIM_INT32);
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
      PimStatus status = pimBroadcastInt(filterObjects[idx++], filterMatrix[i][j]);
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
    PimObjId obj = pimAllocAssociated(filterObjects[0], PIM_INT32);
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
  
    status = pimMul(matrixObjects[i], filterObjects[i], filterObjects[i]);
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

// Function to perform 3D convolution on CPU and compare the results with PIM results
void VerifyWithCPU(std::vector<std::vector<std::vector<int>>> &input,
                   std::vector<std::vector<std::vector<int>>> &PIMResult,
                   std::vector<std::vector<std::vector<int>>> &kernel,
                   int padding, int stride, int batchId, bool moreDebugPrints
                   ) {

  // Compute input, kernel and output dimensions	
  int inputDepth = input.size();
  int inputHeight = input[0].size();
  int inputWidth = input[0][0].size();
  int kernelDepth = kernel.size();
  int kernelHeight = kernel[0].size();
  int kernelWidth = kernel[0][0].size();
  int outputHeight = (inputHeight - kernelHeight) / stride + 1;
  int outputWidth = (inputWidth - kernelWidth) / stride + 1;
  int outputDepth = kernel.size(); // Output depth matches the number of filters in the kernel

  // Check if output dimensions are within reasonable limits
  if (outputHeight <= 0 || outputWidth <= 0 || outputDepth <= 0) {
      std::cerr << "Invalid output dimensions." << std::endl;
      exit(0);
  }

  // Properly initialize the output vector
  Image3D output(outputDepth, vector<vector<int>>(outputHeight, vector<int>(outputWidth, 0)));

  // Perform convolution
  std::cout << "Performing convolution on CPU for batchId: " << batchId << std::endl;
  #pragma omp parallel for collapse (3)
  for (int k = 0; k < kernelDepth; ++k) {
      for (int i = 0; i < outputHeight; ++i) {
          for (int j = 0; j < outputWidth; ++j) {
              int convSum = 0;
              for (int d = 0; d < inputDepth; ++d) {
                  for (int h = 0; h < kernelHeight; ++h) {
                      for (int w = 0; w < kernelWidth; ++w) {
                           convSum += kernel[k][h][w] * input[d][i * stride + h][j * stride + w];
                      }
                  }
              }
              output[k][i][j] = convSum;
          }
      }
  }

  int mismatch_counter = 0;
  std::cout << "Comparing PIM convolution results with CPU results for batchId: " << batchId << std::endl;
  for (int i = 0; i < output.size(); ++i) {
      for (int j = 0; j < output[0].size(); ++j) {
          for (int k = 0; k < output[0][0].size(); ++k) {
              if (output[i][j][k] !=  PIMResult[i][j][k]) {
		if (moreDebugPrints == true) {      
                  std::cout<< "Mismatch between PIM and CPU results at index: " << i << ", " << j << ", " << k << "; PIM result: " << PIMResult[i][j][k] << ", CPU result:" << output[i][j][k] << std::endl;
		}
		mismatch_counter += 1;
              }
          }
      }
  }

  if (moreDebugPrints) { 
    std::cout << "[INFO]: Ouput matrix from CPU for batchId: " << batchId << std::endl;
    printMatrix(output);
    std::cout << "[INFO]: Ouput matrix from PIM for batchId: " << batchId << std::endl;
    printMatrix(PIMResult);    
  }

  if (mismatch_counter == 0) {
    std::cout << "Success: PIM results match with CPU for batchId: " << batchId << std::endl;
  } else {
    std::cout << "Failure: PIM results do not match with CPU for batchId: " << batchId << ", mismatch at " << mismatch_counter << " indices" << std::endl;
    exit(0);
  }
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::vector<std::vector<std::vector<int>>> kernelMatrix;
  std::vector<std::vector<std::vector<std::vector<int>>>> inputMatrix(params.batchSize);

  if (params.imageMatrixFile == nullptr)
  {
    // Generate input matrices for the batch
    for (int b = 0; b < params.batchSize; ++b) {
      inputMatrix[b].resize(params.dim);
      for (auto &mat : inputMatrix[b]) {
        getMatrix(params.row, params.column, params.padding, mat);
      }       
    }
  } else {
      std::cout << "Reading from input file is not implemented yet." << std::endl;
      return 1;
  }

  if (params.kernelMatrixFile == nullptr)
  {
    kernelMatrix.resize(params.kernelDim);
    for (auto &mat : kernelMatrix)
    {
      getMatrix(params.kernelSize, params.kernelSize, 0, mat);
    }
  }
  else
  {
    std::cout << "Reading from input file is not implemented yet." << std::endl;
    return 1;    
  }

  if (!createDevice(params.dramConfigFile))
    return 1;

  PimDeviceProperties deviceProp;
  PimStatus status = pimGetDeviceProperties(&deviceProp);
  if (status != PIM_OK) {
    std::cout << "Abort: pimGetDeviceProperties failed" << std::endl;
    return 1;
  }
  // Get the device parameters
  uint64_t numCols = deviceProp.numColPerSubarray;
  uint64_t numRows = deviceProp.numRowPerSubarray;
  uint64_t numOfBits = deviceProp.numRanks * deviceProp.numBankPerRank * deviceProp.numSubarrayPerBank;  

  // Calculate the input, kernel and the output dimensions 
  uint64_t inputHeight = inputMatrix[0][0].size();
  uint64_t inputWidth = inputMatrix[0][0][0].size();
  uint64_t kernelHeight = kernelMatrix[0].size();
  uint64_t kernelWidth = kernelMatrix[0][0].size(); 
  uint64_t outMatDim = params.kernelDim;
  uint64_t outputHeight = std::floor((inputHeight - kernelHeight) / params.stride) + 1;
  uint64_t outputWidth = std::floor((inputWidth - kernelWidth) / params.stride) + 1;

  // Calculate the required number of PIM rows and number of matrices per row   
  uint64_t numOfPIMRow = params.kernelSize * params.kernelSize;
  uint64_t numOfMatPerRow = std::min(static_cast<int>(std::floor((1.0 * numCols * numOfBits) / (outputHeight * outputWidth * params.batchSize))), params.dim);  

  std::chrono::duration<double, std::milli> hostElapsedTime = std::chrono::duration<double, std::milli>::zero();

  std::vector<std::vector<std::vector<std::vector<int>>>> resultMatrix(params.batchSize);
  for (uint64_t b = 0; b < params.batchSize; ++b) {
    resultMatrix[b].resize(outMatDim, std::vector<std::vector<int>>(outputHeight, std::vector<int>(outputWidth)));
  }

  for (uint64_t i = 0; i < params.kernelDim; i++)
  {
    int tempcol = 0;
    std::vector<int> dstVec(outputHeight * outputWidth * params.batchSize);
    for (uint64_t j = 0; j < params.dim; j += numOfMatPerRow)
    {
      int matChunk = (numOfMatPerRow + j) <= params.dim ? (numOfMatPerRow + j) : params.dim;
      std::vector<std::vector<int>> mergedMat(numOfPIMRow);
      for (uint64_t k = j; k < matChunk; k++)
      { 
        for (int b = 0; b < params.batchSize; ++b) 
        {
          std::vector<std::vector<int>> decompMat;
          getDecomposedMatrix(params.row, params.column, kernelMatrix[i].size(), kernelMatrix[i][0].size(), params.stride, inputMatrix[b][k], decompMat);
          if (params.moreDebugPrints == true) { 
            // Debug print
            std::cout << "[INFO]: Decomposed Matrix:" << std::endl;
            printMatrix(decompMat);
          }
          for (u_int64_t idx = 0; idx < mergedMat.size(); idx++)
          {
            mergedMat[idx].reserve(mergedMat[idx].size() + decompMat[idx].size());
            mergedMat[idx].insert(mergedMat[idx].end(), make_move_iterator(decompMat[idx].begin()), make_move_iterator(decompMat[idx].end()));
            tempcol = mergedMat[idx].size();
          }
        }  
      }

      if (params.moreDebugPrints == true) {
        // Debug print
        std::cout << "[INFO]: Merged Matrix (Iteration " << i << ", Chunk " << j << "):" << std::endl;
        printMatrix(mergedMat);
      }      

      std::vector<int> outVector;
      performConv(kernelMatrix[i], mergedMat, outVector, numOfPIMRow, tempcol, params.moreDebugPrints);

      if (params.moreDebugPrints)
      {
        std::cout << "[INFO]: Output from the PIM at iteration (kernel): " << i << std::endl;
        printVector(outVector); 
      }

      auto start = std::chrono::high_resolution_clock::now();

      int hopSize = outputWidth * outputHeight * params.batchSize;
      if (j == 0)
      {
        std::copy(outVector.begin(), outVector.begin() + hopSize, dstVec.begin());
      }

      for (uint64_t m = 0; m < hopSize; ++m)
      {
        for (uint64_t n = m + hopSize; n < outVector.size(); n += hopSize)
        {
          dstVec[m] += outVector[n];
        }
      }
      auto end = std::chrono::high_resolution_clock::now();
      hostElapsedTime += (end - start);
    
      if (params.moreDebugPrints == true) {
        std::cout << "[INFO]: Intermediate dstVec (Iteration " << i << ", Chunk " << j << "):" << std::endl;
        printVector(dstVec);
      }
    }

    int ddx = 0;
    for (int b = 0; b < params.batchSize; ++b) 
    {
    for (uint64_t r = 0; r < outputHeight; ++r)
      {
        for (uint64_t c = 0; c < outputWidth; ++c)
        {
          resultMatrix[b][i][r][c] = dstVec[ddx++];
        }
      }
    }
  }
  
  if (params.shouldVerify  == true)
  {
    // Perform convolution on CPU and compare results with PIM for each matrix in the batch
    for (int b = 0; b < params.batchSize; ++b) 
    {
      VerifyWithCPU(inputMatrix[b], resultMatrix[b], kernelMatrix, params.padding, params.stride, b, params.moreDebugPrints);
    }
  }

  pimShowStats();
  std::cout << "Host elapsed time: " << std::fixed << std::setprecision(3) << hostElapsedTime.count() << " ms." << std::endl;

  return 0;

}

