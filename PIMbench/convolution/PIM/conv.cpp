// Test: C++ version of convolution
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include <cmath>
#include <getopt.h>
#include <iostream>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "util.h"
#include <chrono>
#include <iomanip>

using namespace std;
typedef vector<vector<vector<int>>> Image3D;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t row, column, dim, stride, kernelHeight, kernelWidth, kernelDim,
      padding;
  char *kernelMatrixFile;
  char *imageMatrixFile;
  char *dramConfigFile;
  bool shouldVerify;
  bool moreDebugPrints;
} Params;

void usage()
{
  fprintf(stderr, "\nUsage:  ./conv.out [options]"
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
                  "\n    -f    input file containing kernel matrices "
                  "(default=generates matrix with random numbers)"
                  "\n    -i    input file containing image matrices "
                  "(default=generates matrix with random numbers)"
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
  while ((opt = getopt(argc, argv, "r:c:d:s:l:w:v:z:f:i:o:p:m:")) >= 0)
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

void getDecomposedMatrix(int matrixRow, int matrixColumn, int filterRow,
                         int filterColumn, int stride,
                         std::vector<std::vector<int>> &inputMatrix,
                         std::vector<std::vector<int>> &decompMatrix)
{
  decompMatrix.resize(filterRow * filterColumn,
                      std::vector<int>(matrixRow * matrixColumn, 0));
  int colIdx = 0, total = 0;
  for (uint64_t i = 0; i < (inputMatrix.size() - filterRow + 1); i += stride)
  {
    for (uint64_t j = 0; j < (inputMatrix[i].size() - filterColumn + 1);
         j += stride)
    {
      int rowIDX = 0;
      total += 1;
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

void performConv(std::vector<std::vector<int>> &filterMatrix,
                 std::vector<std::vector<int>> &inputMatrix,
                 std::vector<int> &outputVector, uint64_t numRequiredPIMRows,
                 int numRequiredPIMCol, bool moreDebugPrints)
{

  if (filterMatrix.empty() || inputMatrix.empty()) return;

  PimObjId filterObject = pimAlloc(PIM_ALLOC_AUTO, numRequiredPIMCol, PIM_INT32);
  if (filterObject == -1)
  {
    std::cout << "Abort: pimAlloc failed for obj1" << std::endl;
    return;
  }

  PimStatus status = pimCopyHostToDevice((void *)outputVector.data(), filterObject);
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
    for (uint64_t i = 0; i < numRequiredPIMRows; i++)
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

  outputVector.resize(numRequiredPIMCol);

  status = pimCopyDeviceToHost(filterObject, outputVector.data());
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
    pimCopyHostToDevice((void *)(inputVector.data() + length), dstObj);       // right halves

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

// Function to perform 3D convolution on CPU and compare the results with PIM
// results
void VerifyWithCPU(std::vector<std::vector<std::vector<int>>> &input,
                   std::vector<std::vector<std::vector<int>>> &kernel,
                   int padding, int stride, bool moreDebugPrints,
                   std::vector<std::vector<std::vector<int>>> &PIMResult)
{

  // Compute input, kernel and output dimensions
  int inputDepth = input.size();
  int inputHeight = input[0].size();
  int inputWidth = input[0][0].size();
  int kernelDepth = kernel.size();
  int kernelHeight = kernel[0].size();
  int kernelWidth = kernel[0][0].size();
  int outputHeight = (inputHeight - kernelHeight) / stride + 1;
  int outputWidth = (inputWidth - kernelWidth) / stride + 1;
  int outputDepth =
      kernel.size(); // Output depth matches the number of filters in the kernel

  // Check if output dimensions are within reasonable limits
  if (outputHeight <= 0 || outputWidth <= 0 || outputDepth <= 0)
  {
    std::cerr << "Invalid output dimensions." << std::endl;
    exit(0);
  }

  // Properly initialize the output vector
  Image3D output(outputDepth, vector<vector<int>>(outputHeight,
                                                  vector<int>(outputWidth, 0)));

  // Perform convolution
  std::cout << "Performing convolution on CPU " << std::endl;
#pragma omp parallel for collapse(3)
  for (int k = 0; k < kernelDepth; ++k)
  {
    for (int i = 0; i < outputHeight; ++i)
    {
      for (int j = 0; j < outputWidth; ++j)
      {
        int convSum = 0;
        for (int d = 0; d < inputDepth; ++d)
        {
          for (int h = 0; h < kernelHeight; ++h)
          {
            for (int w = 0; w < kernelWidth; ++w)
            {
              convSum +=
                  kernel[k][h][w] * input[d][i * stride + h][j * stride + w];
            }
          }
        }
        output[k][i][j] = convSum;
      }
    }
  }

  int mismatch_counter = 0;
  std::cout << "Comparing PIM convolution results with CPU results "
            << std::endl;
  for (uint64_t i = 0; i < output.size(); ++i)
  {
    for (uint64_t j = 0; j < output[0].size(); ++j)
    {
      for (uint64_t k = 0; k < output[0][0].size(); ++k)
      {
        if (output[i][j][k] != PIMResult[i][j][k])
        {
          if (moreDebugPrints == true)
          {
            std::cout << "Mismatch between PIM and CPU results at index: " << i
                      << ", " << j << ", " << k
                      << "; PIM result: " << PIMResult[i][j][k]
                      << ", CPU result:" << output[i][j][k] << std::endl;
          }
          mismatch_counter += 1;
        }
      }
    }
  }

  if (moreDebugPrints == true)
  {
    std::cout << "Ouput matrix from CPU:" << std::endl;
    for (uint64_t i = 0; i < output.size(); ++i)
    {
      std::cout << "Layer " << i << ":\n";
      printMatrix(output[i]);
      std::cout << "\n";
    }
  }

  if (mismatch_counter == 0)
  {
    std::cout << "Success: PIM results match with CPU" << std::endl;
  }
  else
  {
    std::cout << "Failure: PIM results do not match with CPU, mismatch at "
              << mismatch_counter << " indices" << std::endl;
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
    for (uint64_t i = 0; i < params.dim; i++)
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
    std::cerr << "Reading from the input file is not implemented yet for the "
                 "kernel matrix"
              << std::endl;
    return 1;
  }

  if (!createDevice(params.dramConfigFile))
    return 1;

  PimDeviceProperties deviceProp;
  PimStatus status = pimGetDeviceProperties(&deviceProp);
  if (status != PIM_OK)
  {
    std::cout << "Abort: pimGetDeviceProperties failed" << std::endl;
    return 1;
  }
  // Get the device parameters
  uint64_t numCols = deviceProp.numColPerSubarray;
  uint64_t numRows = deviceProp.numRowPerSubarray;
  uint64_t numOfBits =
      uint64_t(deviceProp.numRanks) * uint64_t(deviceProp.numBankPerRank) *
      uint64_t(deviceProp.numSubarrayPerBank) * numCols * numRows;

  int inputDepth = inputMatrix.size();
  int inputHeight = inputMatrix[0].size();
  int inputWidth = inputMatrix[0][0].size();
  int kernelHeight = kernelMatrix[0].size();
  int kernelWidth = kernelMatrix[0][0].size();

  int outMatDim = params.kernelDim;
  int outMatRow = std::floor((inputHeight - kernelHeight) / params.stride) + 1;
  int outMatCol = std::floor((inputWidth - kernelWidth) / params.stride) + 1;
  int numOfMatPerRow =
      std::floor((1.0 * numOfBits) / (outMatRow * outMatCol)) < params.dim
          ? floor((1.0 * numOfBits) / (outMatRow * outMatCol))
          : params.dim;
  int numOfPIMRow = params.kernelHeight * params.kernelWidth;

  std::chrono::duration<double, std::milli> hostElapsedTime = std::chrono::duration<double, std::milli>::zero();
  
  std::vector<std::vector<std::vector<int>>> resultMatrix;
  resultMatrix.resize(outMatDim, std::vector<std::vector<int>>(outMatRow, std::vector<int>(outMatCol)));


  for (uint64_t i = 0; i < params.kernelDim; i++)
  {
    int tempcol = 0;
    std::vector<int> dstVec(outMatRow * outMatCol);
    std::vector<int> outVector(outMatRow * outMatCol * inputDepth, 0);
    for (uint64_t j = 0; j < params.dim; j += numOfMatPerRow)
    {
      uint64_t matChunk = (numOfMatPerRow + j) <= params.dim
                              ? (numOfMatPerRow + j)
                              : params.dim;

      std::vector<std::vector<int>> mergedMat(numOfPIMRow);
      std::vector<std::vector<int>> decompMat;
      for (uint64_t k = j; k < matChunk; k++)
      {
        getDecomposedMatrix(params.row, params.column, kernelHeight,
                            kernelWidth, params.stride, inputMatrix[k],
                            decompMat);
        if (params.moreDebugPrints == true)
        {
          // Debug print
          std::cout << "Decomposed Matrix:" << std::endl;
          printMatrix(decompMat);
        }
        // Merge the matrices
        for (uint64_t idx = 0; idx < mergedMat.size(); idx++)
        {
          mergedMat[idx].insert(mergedMat[idx].end(),
                                std::make_move_iterator(decompMat[idx].begin()),
                                std::make_move_iterator(decompMat[idx].end()));
        }
        tempcol = mergedMat[0].size();
      }

      if (params.moreDebugPrints == true)
      {
        // Debug print
        std::cout << "Merged Matrix (Iteration " << i << ", Chunk " << j
                  << "):" << std::endl;
        printMatrix(mergedMat);
      }

      performConv(kernelMatrix[i], mergedMat, outVector, numOfPIMRow, tempcol,
                  params.moreDebugPrints);
      if (params.moreDebugPrints == true)
      {
        // Debug print
        std::cout << "Output Matrix from performConv():" << std::endl;
        printVector(outVector);
      }

      if (params.moreDebugPrints == true)
      {
        // Debug print
        std::cout << "Intermediate dstVec (Iteration " << i << ", Chunk " << j
                  << "):" << std::endl;
        printVector(dstVec);
      }
    }
    int hopSize = outMatCol * outMatRow;
    aggregate(outVector, dstVec, hopSize);
    int ddx = 0;
    for (int rdx = 0; rdx < outMatRow; ++rdx)
    {
      for (int cdx = 0; cdx < outMatCol; ++cdx)
      {
        resultMatrix[i][rdx][cdx] = dstVec[ddx++];
      }
    }

    if (params.moreDebugPrints == true)
    {
      // Debug print
      std::cout << "Result matrix from PIM< (Kernel " << i << "):" << std::endl;
      printMatrix(resultMatrix[i]);
    }
  }

  if (params.shouldVerify == true)
  {
    // Perform convolution on CPU and compare results with PIM
    VerifyWithCPU(inputMatrix, kernelMatrix, params.padding, params.stride,
                  params.moreDebugPrints, resultMatrix);
  }

  pimShowStats();
  std::cout << "Host elapsed time: " << std::fixed << std::setprecision(3)
            << hostElapsedTime.count() << " ms." << std::endl;

  return 0;
}
