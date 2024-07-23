// Test: C++ version of convolution
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
  int row, column, dim, stride, kernelHeight, kernelWidth, kernelDim, padding;
  char *kernelMatrixFile;
  char *imageMatrixFile;
  char *dramConfigFile;
  bool shouldVerify;
  bool moreDebugPrints;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./conv.out [options]"
          "\n"
          "\n    -r    row (default=224)"
          "\n    -c    column (default=224)"
          "\n    -d    dimension (default=3)"
          "\n    -s    stride (default=1)"
          "\n    -l    kernel height (default=3)"
          "\n    -w    kernel width (default=3)"
          "\n    -z    kernel dimension (default=64)"
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
  p.kernelHeight = 3;
  p.kernelWidth = 3;
  p.kernelMatrixFile = nullptr;
  p.imageMatrixFile = nullptr;
  p.dramConfigFile = nullptr;
  p.shouldVerify = false;
  p.kernelDim = 64;
  p.padding = 1;
  p.moreDebugPrints = false;

  int opt;
  while ((opt = getopt(argc, argv, "r:c:d:s:h:l:w:v:z:f:i:p:m:")) >= 0)
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
    case 'z':
      p.kernelDim = atoi(optarg);
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

void performConv(std::vector<std::vector<int>> &filterMatrix, std::vector<PimObjId> &filterObjects, std::vector<PimObjId> &matrixObjects, std::vector<int> &outputMatrix, int numRequiredPIMRows, int numRequiredPIMCol, bool moreDebugPrints)
{
  unsigned bitsPerElement = 32;
  std::vector<int> temp;

  int idx = 0;
  for (int i = 0; i < filterMatrix.size(); ++i)
  {
    for (int j = 0; j < filterMatrix[i].size(); ++j)
    {
      PimStatus status = pimBroadcastInt(filterObjects[idx++], filterMatrix[i][j]);
      if (status != PIM_OK)
      {
        std::cout << "Function: " << __func__  << ": " << "Abort: pimBroadCastInt failed for filterObjects at iteration, i, j:" << i << j << std::endl;
        return;
      }
    }
  }
    
  for (int i = 0; i < matrixObjects.size(); i++)
  {
    PimStatus status = pimMul(matrixObjects[i], filterObjects[i], filterObjects[i]);
    if (status != PIM_OK)
    {
      std::cout << "Function: " << __func__  << ": " << "Abort: pimMul failed between matrixObjects[i] and filterObjects[i] at itertion, i:" << i << std::endl;
      return;
    }
  }
  
  for (int i = 1; i < numRequiredPIMRows; i++)
  {
    PimStatus status = pimAdd(filterObjects[0], filterObjects[i], filterObjects[0]);
    if (status != PIM_OK)
    {
      std::cout << "Function: " << __func__  << ": " << "Abort: pimAdd failed between filterObjects[0] and filterObjects[i] at iteration, i:" << i << std::endl;
      return;
    }
  }

  outputMatrix.resize(numRequiredPIMCol);

  PimStatus status = pimCopyDeviceToHost(filterObjects[0], outputMatrix.data());
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__  << ": " << "Abort: pimCopyDeviceToHost failed between filterObjects[0] and outputMatrix" << std::endl;
    return;
  }

  if (moreDebugPrints == true) {
    // Debug print
    std::cout << "Output Matrix from performConv():" << std::endl;
    for (const auto &val : outputMatrix)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;
  }
}

// Function to perform 3D convolution on CPU and compare the results with PIM results
void VerifyWithCPU(std::vector<std::vector<std::vector<int>>> &input,
                   std::vector<std::vector<std::vector<int>>> &kernel,
                   int padding, int stride, bool moreDebugPrints,
                   std::vector<std::vector<std::vector<int>>> &PIMResult
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
  std::cout << "Performing convolution on CPU " << std::endl;
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
  std::cout << "Comparing PIM convolution results with CPU results " << std::endl;
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
 
  if (moreDebugPrints == true) { 
    std::cout << "Ouput matrix from CPU:" << std::endl;
    for (int i = 0; i < output.size(); ++i) {
        std::cout << "Layer " << i << ":\n";
        for (int j = 0; j < output[i].size(); ++j) { 
            for (int k = 0; k < output[i][j].size(); ++k) {
                std::cout << output[i][j][k] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
  }

  if (mismatch_counter == 0) {
    std::cout << "Success: PIM results match with CPU" << std::endl;
  } else {
    std::cout << "Failure: PIM results do not match with CPU, mismatch at " << mismatch_counter << " indices" << std::endl;
    exit(0);
  }
}

int main(int argc, char *argv[]) {

  struct Params params = getInputParams(argc, argv);
  std::vector<std::vector<std::vector<int>>> inputMatrix;
  std::vector<std::vector<std::vector<int>>> kernelMatrix;

  // Initialize inputMatrix
  if (params.imageMatrixFile == nullptr) {
    inputMatrix.resize(params.dim);
    for (int i = 0; i < params.dim; i++) {
      getMatrix(params.row, params.column, params.padding, inputMatrix[i]);
    }
  }
  else
  {
    std::cout << "Reading from input file is not implemented yet." << std::endl;
    return 0;
  }

  // Initialize kernelMatrix
  if (params.kernelMatrixFile == nullptr) {
    kernelMatrix.resize(params.kernelDim);
    for (auto &mat : kernelMatrix) {
      getMatrix(params.kernelHeight, params.kernelWidth, 0, mat);
    }
  } else {
    std::cout << "Reading from input file is not implemented yet." << std::endl;
    return 0;
  }

  // Create the device with DRAM configuration
  if (!createDevice(params.dramConfigFile))
    return 1;

  // TODO: Get number of columns after creating the device, maybe support an API like getDeviceConfig.
  unsigned numCols = 8192, numOfCore = 4096;

  // Calculate dimensions of the input and kernel matrices. The input matrix is already padded.
  int inputDepth = inputMatrix.size();
  int inputHeight = inputMatrix[0].size();
  int inputWidth = inputMatrix[0][0].size();
  int kernelDepth = kernelMatrix.size();
  int kernelHeight = kernelMatrix[0].size();
  int kernelWidth = kernelMatrix[0][0].size();

  // Calculate the dimensions of the output matrix assuming the input matrix is already padded
  int outMatRow = std::floor((inputHeight - kernelHeight) / params.stride) + 1;
  int outMatCol = std::floor((inputWidth - kernelWidth) / params.stride) + 1;   
  int numOfMatPerRow = std::floor((1.0 * numCols * numOfCore) / (outMatRow * outMatCol)) < inputDepth ? std::floor((1.0 * numCols * numOfCore) / (outMatRow * outMatCol)) : inputDepth;
  int numOfPIMRow = kernelHeight * kernelWidth;

  // Debug prints
  if (params.moreDebugPrints)
  {
    std::cout << "Num of rows in the output matrix: " << outMatRow << "\n";    
    std::cout << "Num of columns in the output matrix: " << outMatCol << "\n";    
    std::cout << "Num of matrices per row: " << numOfMatPerRow << "\n";    
    std::cout << "Num of PIM rows: " << numOfPIMRow << "\n";    
  }
  
  std::chrono::duration<double, std::milli> hostElapsedTime = std::chrono::duration<double, std::milli>::zero();
  std::vector<std::vector<std::vector<int>>> resultMatrix(kernelDepth, std::vector<std::vector<int>>(outMatRow, std::vector<int>(outMatCol)));
  int numOfPIMCol = 0;
  int hopSize = outMatCol * outMatRow;
  std::vector<int> dstVec(outMatRow * outMatCol);
  std::vector<std::vector<int>> decompMat;
  std::vector<std::vector<int>> mergedMat(numOfPIMRow);
  std::vector<int> outVector;
  
  // Loop through input depth in chunks
  for (int j = 0; j < inputDepth; j += numOfMatPerRow) {
    int matChunk = (numOfMatPerRow + j) <= inputDepth ? (numOfMatPerRow + j) : inputDepth;

    // Decompose and merge matrices
    for (int k = j; k < matChunk; k++) {
      decomposeMatrix(inputHeight, inputWidth, kernelHeight, kernelWidth, params.stride, 0, inputMatrix[k], decompMat);
      if (params.moreDebugPrints) { 
        // Debug print decomposed matrix
        std::cout << "Decomposed Matrix:" << std::endl;
        for (const auto &row : decompMat) {
          for (const auto &val : row) {
            std::cout << val << " ";
          }
          std::cout << std::endl;
        }
      }
      for (int idx = 0; idx < mergedMat.size(); idx++) {
        mergedMat[idx].reserve(mergedMat[idx].size() + decompMat[idx].size());
        mergedMat[idx].insert(mergedMat[idx].end(), make_move_iterator(decompMat[idx].begin()), make_move_iterator(decompMat[idx].end()));
        numOfPIMCol = mergedMat[idx].size();
      }
    }

    if (params.moreDebugPrints) {
      // Debug print merged matrix
      std::cout << "Merged Matrix:" << std::endl;                 
      for (const auto &row : mergedMat) {
        for (const auto &val : row) {
          std::cout << val << " ";
        }
        std::cout << std::endl;
      }
    }
  }   
  
  // Allocate PIM Objects for the kernels
  // Allocate PIM Objects for the input matrices and copy the corrresponding data from the host to the device 
  unsigned bitsPerElement = 32;
  std::vector<PimObjId> filterObjects;
  std::vector<PimObjId> matrixObjects;  
  PimObjId obj1 = pimAlloc(PIM_ALLOC_AUTO, numOfPIMCol, bitsPerElement, PIM_INT32);
  if (obj1 == -1)
  {
    std::cout << "Function: " << __func__ << ": " << "Abort: pimAlloc failed for obj1" << std::endl;
    return 1;
  }
  filterObjects.push_back(obj1);
  for (int i = 1; i < numOfPIMRow; i++)
  {
    PimObjId obj = pimAllocAssociated(bitsPerElement, filterObjects[0], PIM_INT32);
    if (obj == -1)
    {
      std::cout << "Function: " << __func__ << ": " << "Abort: pimAllocAssociated failed for filterObjects at iteration, i:" << i << std::endl;
      return 1;
    }
    filterObjects.push_back(obj);
  }

  for (int i = 0; i < numOfPIMRow; i++)
  {
    PimObjId obj = pimAllocAssociated(bitsPerElement, filterObjects[0], PIM_INT32);
    if (obj == -1)
    {
      std::cout << "Function: " << __func__ << ": " << "Abort: pimAllocAssociated failed for matrixObjects at iteration, i:" << i << std::endl;
      return 1;
    }
    matrixObjects.push_back(obj);
  }

  for (int i = 0; i < mergedMat.size(); i++)
  {
    PimStatus status = pimCopyHostToDevice((void *)mergedMat[i].data(), matrixObjects[i]);
    if (status != PIM_OK)
    {
      std::cout << "Function: " << __func__ << ": " << "Abort: pimCopyHostToDevice failed for copying between mergedMat and matrixObjects at iteration, i" << i << std::endl;
      return 1;
    }

  }

  // Perform convolution
  for (int i = 0; i < kernelDepth; i++) {

    performConv(kernelMatrix[i], filterObjects, matrixObjects, outVector, numOfPIMRow, numOfPIMCol, params.moreDebugPrints);

    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for    
    for (int j = 0; j < inputDepth; j += numOfMatPerRow) {
      if (j == 0) {
        std::copy(outVector.begin(), outVector.begin() + hopSize, dstVec.begin());
      }

      // Accumulate results
      for (int m = 0; m < hopSize; ++m) {
        for (int n = m + hopSize; n < outVector.size(); n += hopSize) {
          dstVec[m] += outVector[n];
        }
      }
    
      if (params.moreDebugPrints) {
        // Debug print intermediate dstVec
        std::cout << "Intermediate dstVec (Iteration " << i << ", Chunk " << j << "):" << std::endl;
        for (const auto &val : dstVec) {
          std::cout << val << " ";
        }
        std::cout << std::endl;
      }
    }

    // Store result matrix
    int ddx = 0;
    for (int rdx = 0; rdx < outMatRow; ++rdx) {
      for (int cdx = 0; cdx < outMatCol; ++cdx) {
        resultMatrix[i][rdx][cdx] = dstVec[ddx++];
      }
    }
  
    if (params.moreDebugPrints) {
      // Debug print result matrix
      std::cout << "Result matrix from PIM (Kernel " << i << "):" << std::endl;
      for (const auto &row : resultMatrix[i]) {
        for (const auto &val : row) {
          std::cout << val << " ";
        }
        std::cout << std::endl;
      }
    }      
    auto end = std::chrono::high_resolution_clock::now();
    hostElapsedTime += (end - start);
  }

  for (auto elem : filterObjects)
  {
    pimFree(elem);
  }
  for (auto elem : matrixObjects)
  {
    pimFree(elem);
  }
    
  if (params.shouldVerify) {
    // Perform convolution on CPU and compare results with PIM
    VerifyWithCPU(inputMatrix, kernelMatrix, params.padding, params.stride, params.moreDebugPrints, resultMatrix);
  }

  // Show PIM stats
  pimShowStats();
  std::cout << "Host elapsed time: " << std::fixed << std::setprecision(3) << hostElapsedTime.count() << " ms." << std::endl;

  return 0;
}
