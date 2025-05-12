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
#include "util.h"
#include <iomanip>
#include <chrono>

using namespace std;
typedef vector<vector<vector<int>>> Image3D;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t row, column, dim, stride, kernelSize, kernelDim, padding, batchSize;
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
  for (uint64_t i = 0; i < (inputMatrix.size() - filterRow + 1); i += stride)
  {
    for (uint64_t j = 0; j < (inputMatrix[i].size() - filterColumn + 1); j += stride)
    {
      int rowIDX = 0;
      for (uint64_t k = i; k < i + filterRow; k++)
      {
        for (uint64_t l = j; l < j + filterColumn; l++)
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
  if (filterMatrix.empty() || inputMatrix.empty()) return;

  PimObjId filterObject = pimAlloc(PIM_ALLOC_AUTO, numRequiredPIMCol, PIM_INT32);
  if (filterObject == -1)
  {
    std::cout << "Abort: pimAlloc failed for obj1" << std::endl;
    return;
  }

  PimStatus status = pimCopyHostToDevice((void *)outputMatrix.data(), filterObject);
  if (status != PIM_OK)
  {
    std::cout << "Abort: pimCopyHostToDevice from inputMatrix to matrixObject" << std::endl;
    return;
  }

  PimObjId matrixObject = pimAllocAssociated(filterObject, PIM_INT32);
  if (matrixObject == -1)
  {
    std::cout << "Abort: pimAllocAssociated failed obj (matrixObjects) at iteration: " << matrixObject << std::endl;
    return;
  }

  int col = filterMatrix[0].size();
  for (uint64_t j = 0; j < inputMatrix.size(); j += numRequiredPIMRows)
  {
    for (int i = 0; i < numRequiredPIMRows; i++)
    {
      PimStatus status = pimCopyHostToDevice((void *)inputMatrix[i + j].data(), matrixObject);
      if (status != PIM_OK)
      {
        std::cout << "Abort: pimCopyHostToDevice from inputMatrix to "
                     "matrixObjects at iteration: "
                  << i << std::endl;
        return;
      }

      status = pimScaledAdd(matrixObject, filterObject, filterObject,
                            filterMatrix[i / col][i % col]);
      if (status != PIM_OK)
      {
        std::cout << "Abort" << std::endl;
        return;
      }
    }
  }

  outputMatrix.resize(numRequiredPIMCol);

  status = pimCopyDeviceToHost(filterObject, outputMatrix.data());
  if (status != PIM_OK)
  {
    std::cout << "Abort: pimCopyDeviceToHost failed between filterObjects[0] "
                 "and outputVector"
              << std::endl;
    return;
  }

  pimFree(filterObject);
  pimFree(matrixObject);
  
}

void aggregate(std::vector<int> &inputVector, std::vector<int> &outputVector, unsigned hopSize)
{

  uint64_t numChunks = inputVector.size() / hopSize;
  uint64_t remChunk = numChunks;
  while (remChunk > 1)
  {
    uint64_t reduceChunks = remChunk;
    std::vector<int> tempVector(hopSize, 0);

    // If remChunk is odd, save the last chunk and exclude from current level reduction
    if (remChunk % 2) {
      std::copy(inputVector.end() - hopSize, inputVector.end(), tempVector.begin());
      reduceChunks = remChunk - 1;
    }

    uint64_t length = (reduceChunks / 2) * hopSize;

    PimObjId srcObj = pimAlloc(PIM_ALLOC_AUTO, length, PIM_INT32);
    PimObjId dstObj = pimAllocAssociated(srcObj, PIM_INT32);

    if (srcObj == -1 || dstObj == -1) {
      std::cerr << "Abort: pimAlloc failed\n";
      return;
    }

    PimStatus status = pimCopyHostToDevice((void *)inputVector.data(), srcObj);                // left halves
    if (status != PIM_OK)
    {
      std::cout << "Abort: pimCopyDeviceToHost failed"
                << std::endl;
      return;
    }
    status = pimCopyHostToDevice((void *)(inputVector.data() + length), dstObj);       // right halves

    pimAdd(srcObj, dstObj, dstObj);
    inputVector.resize(length);
    pimCopyDeviceToHost(dstObj, inputVector.data());

    pimFree(srcObj);
    pimFree(dstObj);

    // If we saved a leftover chunk, add it to the result
    if (reduceChunks != remChunk) {
      PimObjId finalSrc = pimAlloc(PIM_ALLOC_AUTO, hopSize, PIM_INT32);
      PimObjId finalDst = pimAllocAssociated(finalSrc, PIM_INT32);

      if (finalSrc == -1 || finalDst == -1) {
        std::cerr << "Abort: final PIM alloc failed\n";
        return;
      }

      pimCopyHostToDevice(inputVector.data(), finalSrc);
      pimCopyHostToDevice(tempVector.data(), finalDst);
      pimAdd(finalSrc, finalDst, finalDst);
      pimCopyDeviceToHost(finalDst, inputVector.data());

      pimFree(finalSrc);
      pimFree(finalDst);
    }

    remChunk = reduceChunks / 2;
  }

  outputVector = inputVector;
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
  for (uint64_t i = 0; i < output.size(); ++i) {
      for (uint64_t j = 0; j < output[0].size(); ++j) {
          for (uint64_t k = 0; k < output[0][0].size(); ++k) {
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
    for (uint64_t b = 0; b < params.batchSize; ++b) {
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
  uint64_t numOfBits = deviceProp.numRanks * deviceProp.numBankPerRank * deviceProp.numSubarrayPerBank;  

  // Calculate the input, kernel and the output dimensions 
  uint64_t inputDepth = inputMatrix[0].size();
  uint64_t inputHeight = inputMatrix[0][0].size();
  uint64_t inputWidth = inputMatrix[0][0][0].size();
  uint64_t kernelHeight = kernelMatrix[0].size();
  uint64_t kernelWidth = kernelMatrix[0][0].size(); 
  uint64_t outMatDim = params.kernelDim;
  uint64_t outputHeight = std::floor((inputHeight - kernelHeight) / params.stride) + 1;
  uint64_t outputWidth = std::floor((inputWidth - kernelWidth) / params.stride) + 1;

  // Calculate the required number of PIM rows and number of matrices per row   
  uint64_t numOfPIMRow = params.kernelSize * params.kernelSize;
  uint64_t numOfMatPerRow = std::min(static_cast<uint64_t>(std::floor((1.0 * numCols * numOfBits) / (outputHeight * outputWidth * params.batchSize))), params.dim);  

  std::chrono::duration<double, std::milli> hostElapsedTime = std::chrono::duration<double, std::milli>::zero();

  std::vector<std::vector<std::vector<std::vector<int>>>> resultMatrix(params.batchSize);
  for (uint64_t b = 0; b < params.batchSize; ++b) {
    resultMatrix[b].resize(outMatDim, std::vector<std::vector<int>>(outputHeight, std::vector<int>(outputWidth)));
  }

  for (uint64_t i = 0; i < params.kernelDim; i++)
  {
    int tempcol = 0;
    std::vector<int> dstVec(outputHeight * outputWidth * params.batchSize);
    std::vector<int> outVector(outputHeight * outputWidth * params.batchSize * inputDepth, 0);
    for (uint64_t j = 0; j < params.dim; j += numOfMatPerRow)
    {
      uint64_t matChunk = (numOfMatPerRow + j) <= params.dim ? (numOfMatPerRow + j) : params.dim;
      std::vector<std::vector<int>> mergedMat(numOfPIMRow);
      for (uint64_t k = j; k < matChunk; k++)
      { 
        for (uint64_t b = 0; b < params.batchSize; ++b) 
        {
          std::vector<std::vector<int>> decompMat;
          getDecomposedMatrix(params.row, params.column, kernelMatrix[i].size(), kernelMatrix[i][0].size(), params.stride, inputMatrix[b][k], decompMat);
          if (params.moreDebugPrints == true) { 
            // Debug print
            std::cout << "[INFO]: Decomposed Matrix:" << std::endl;
            printMatrix(decompMat);
          }
          for (uint64_t idx = 0; idx < mergedMat.size(); idx++)
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

      performConv(kernelMatrix[i], mergedMat, outVector, numOfPIMRow, tempcol, params.moreDebugPrints);

      if (params.moreDebugPrints)
      {
        std::cout << "[INFO]: Output from the PIM at iteration (kernel): " << i << std::endl;
        printVector(outVector); 
      }
    }

    int hopSize = outputWidth * outputHeight * params.batchSize;
    aggregate(outVector, dstVec, hopSize);

    int ddx = 0;
    for (uint64_t b = 0; b < params.batchSize; ++b) 
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
    for (uint64_t b = 0; b < params.batchSize; ++b) 
    {
      VerifyWithCPU(inputMatrix[b], resultMatrix[b], kernelMatrix, params.padding, params.stride, b, params.moreDebugPrints);
    }
  }

  pimShowStats();
  std::cout << "Host elapsed time: " << std::fixed << std::setprecision(3) << hostElapsedTime.count() << " ms." << std::endl;

  return 0;

}

