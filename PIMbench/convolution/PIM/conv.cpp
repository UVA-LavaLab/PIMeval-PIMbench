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
#include "../../util.h"
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
  p.dim = 64;
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
  while ((opt = getopt(argc, argv, "r:c:d:s:l:w:v:z:f:i:p:m:")) >= 0)
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

void performConv(std::vector<std::vector<int>> &filterMatrix, std::vector<std::vector<int>> &inputMatrix, std::vector<int> &outputVector, int numRequiredPIMRows, int numRequiredPIMCol, bool moreDebugPrints)
{
  std::vector<PimObjId> filterObjects;
  std::vector<int> temp;
  PimObjId obj1 = pimAlloc(PIM_ALLOC_AUTO, numRequiredPIMCol, PIM_INT32);
  if (obj1 == -1)
  {
    std::cout << "Abort: pimAlloc failed for obj1" << std::endl;
    return;
  }
  filterObjects.push_back(obj1);
  for (int i = 1; i < numRequiredPIMRows; i++)
  {
    PimObjId obj = pimAllocAssociated(filterObjects[0], PIM_INT32);
    if (obj == -1)
    {
      std::cout << "Abort: pimAllocAssociated failed for obj (filterObjects) at iteration: " << i << std::endl;
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
        std::cout << "Abort: pimBroadcastInt failed for filterObjects at i=" << i << " and j=" << j << std::endl;
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
      std::cout << "Abort: pimAllocAssociated failed obj (matrixObjects) at iteration: " << i << std::endl;
      return;
    }
    matrixObjects.push_back(obj);
  }

  for (int i = 0; i < inputMatrix.size(); i++)
  {
    PimStatus status = pimCopyHostToDevice((void *)inputMatrix[i].data(), matrixObjects[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort: pimCopyHostToDevice from inputMatrix to matrixObjects at iteration: " << i << std::endl;
      return;
    }
  
    status = pimMul(matrixObjects[i], filterObjects[i], filterObjects[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort: pimMul failed between matrixObjects and filterObjects at iteration: " << i << std::endl;
      return;
    }
  }

  for (int i = 1; i < numRequiredPIMRows; i++)
  {
    PimStatus status = pimAdd(filterObjects[0], filterObjects[i], filterObjects[0]);
    if (status != PIM_OK)
    {
      std::cout << "Abort: pimAddd failed between filterObjects at itertaion: " << i << std::endl;
      return;
    }
  }

  outputVector.resize(numRequiredPIMCol);

  PimStatus status = pimCopyDeviceToHost(filterObjects[0], outputVector.data());
  if (status != PIM_OK)
  {
    std::cout << "Abort: pimCopyDeviceToHost failed between filterObjects[0] and outputVector" << std::endl;
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
      printMatrix(output[i]);
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

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::vector<std::vector<std::vector<int>>> inputMatrix;
  std::vector<std::vector<std::vector<int>>> kernelMatrix;

  if (params.imageMatrixFile == nullptr)
  {
    inputMatrix.resize(params.dim);
    for (int i = 0; i < params.dim; i++)
    {
      getMatrix(params.row, params.column, params.padding, inputMatrix[i]);
    }
  }
  else
  {
    std::cerr << "Reading from the input file is not implemented yet for the input matrix" << std::endl;
    return 1;
  }

  if (params.kernelMatrixFile == nullptr)
  {
    kernelMatrix.resize(params.kernelDim);
    for (auto &mat : kernelMatrix)
    {
      getMatrix(params.kernelHeight, params.kernelWidth, 0, mat);
    }
  }
  else
  {
    std::cerr << "Reading from the input file is not implemented yet for the kernel matrix" << std::endl;
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
  uint64_t numOfBits = uint64_t(deviceProp.numRanks) * uint64_t(deviceProp.numBankPerRank) * uint64_t(deviceProp.numSubarrayPerBank) * numCols * numRows;  

  int inputDepth = inputMatrix.size();
  int inputHeight = inputMatrix[0].size();
  int inputWidth = inputMatrix[0][0].size();
  int kernelHeight = kernelMatrix[0].size();
  int kernelWidth = kernelMatrix[0][0].size();

  int outMatDim = params.kernelDim;
  int outMatRow = std::floor((inputHeight - kernelHeight) / params.stride) + 1;
  int outMatCol = std::floor((inputWidth - kernelWidth) / params.stride) + 1;   
  int numOfMatPerRow = floor((1.0 * numOfBits) / (outMatRow * outMatCol)) < params.dim ? floor((1.0 * numOfBits) / (outMatRow * outMatCol)) : params.dim;
  int numOfPIMRow = params.kernelHeight * params.kernelWidth;

  std::chrono::duration<double, std::milli> hostElapsedTime = std::chrono::duration<double, std::milli>::zero();
  std::vector<std::vector<std::vector<int>>> resultMatrix;
  resultMatrix.resize(outMatDim, std::vector<std::vector<int>>(outMatRow, std::vector<int>(outMatCol)));
  for (int i = 0; i < params.kernelDim; i++)
  {
    int tempcol = 0;
    std::vector<int> dstVec(outMatRow * outMatCol);
    for (int j = 0; j < params.dim; j += numOfMatPerRow)
    {
      int matChunk = (numOfMatPerRow + j) <= params.dim ? (numOfMatPerRow + j) : params.dim;

      std::vector<std::vector<int>> mergedMat(numOfPIMRow);
      std::vector<std::vector<int>> decompMat;      
      for (int k = j; k < matChunk; k++)
      {
        getDecomposedMatrix(params.row, params.column, kernelHeight, kernelWidth, params.stride, inputMatrix[k], decompMat);
        if (params.moreDebugPrints == true) { 
          // Debug print
          std::cout << "Decomposed Matrix:" << std::endl;
          printMatrix(decompMat);
        }
        // Merge the matrices
        for (int idx = 0; idx < mergedMat.size(); idx++) {
          mergedMat[idx].insert(mergedMat[idx].end(),
                                std::make_move_iterator(decompMat[idx].begin()),
                                std::make_move_iterator(decompMat[idx].end()));
        }
        tempcol = mergedMat[0].size();        
      }

      if (params.moreDebugPrints == true) {
        // Debug print
        std::cout << "Merged Matrix (Iteration " << i << ", Chunk " << j << "):" << std::endl;
        printMatrix(mergedMat);
      }      

      std::vector<int> outVector (outMatRow * outMatCol *  inputDepth);
      performConv(kernelMatrix[i], mergedMat, outVector, numOfPIMRow, tempcol, params.moreDebugPrints);
      if (params.moreDebugPrints == true) {
        // Debug print
        std::cout << "Output Matrix from performConv():" << std::endl;
        printVector(outVector);
      }

      int hopSize = outMatCol * outMatRow;
      auto start = std::chrono::high_resolution_clock::now();
      if (j == 0)
      {
        std::copy(outVector.begin(), outVector.begin() + hopSize, dstVec.begin());
      }      
      for (int m = 0; m < hopSize; ++m)
      {
        for (int n = m + hopSize; n < outVector.size(); n += hopSize)
        {
          dstVec[m] += outVector[n];
        }
      }
      auto end = std::chrono::high_resolution_clock::now();
      hostElapsedTime += (end - start);
    
      if (params.moreDebugPrints == true) {
        // Debug print
        std::cout << "Intermediate dstVec (Iteration " << i << ", Chunk " << j << "):" << std::endl;
        printVector(dstVec);
      }

    }
    int ddx = 0;
    for (int rdx = 0; rdx < outMatRow; ++rdx)
    {
      for (int cdx = 0; cdx < outMatCol; ++cdx)
      {
        resultMatrix[i][rdx][cdx] = dstVec[ddx++];
      }
    }
  
    if (params.moreDebugPrints ==  true) {
      // Debug print
      std::cout << "Result matrix from PIM< (Kernel " << i << "):" << std::endl;
      printMatrix(resultMatrix[i]);
    }
  }
  
  if (params.shouldVerify  == true)
  {
    // Perform convolution on CPU and compare results with PIM
    VerifyWithCPU(inputMatrix, kernelMatrix, params.padding, params.stride, params.moreDebugPrints, resultMatrix);
  }

  pimShowStats();
  std::cout << "Host elapsed time: " << std::fixed << std::setprecision(3) << hostElapsedTime.count() << " ms." << std::endl;

  return 0;
}

