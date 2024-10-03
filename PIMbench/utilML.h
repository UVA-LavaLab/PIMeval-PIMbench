// PIMeval Simulator - Application Utilities for Machine Learning Code.
// The utilies for Conv, MaxPooling, RELU, GEMV are got from their respective cpp files. So, any change in the individual cpp file should also be updated here manually.
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef PIM_FUNC_SIM_APPS_UTIL_ML_H
#define PIM_FUNC_SIM_APPS_UTIL_ML_H

#include <iostream>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <vector>
#include <iomanip>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include "libpimeval.h"
#include <map>
#include <fstream>
#include <sstream>
#if defined(COMPILE_WITH_JPEG)
#include <jpeglib.h>
#endif  
using namespace std;

std::chrono::duration<double, std::milli> hostElapsedTime = std::chrono::duration<double, std::milli>::zero();

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
// Function to perform softmax operation on Host.
//  -> Find the max value in the input vector
//  -> Compute the exponentials of each (element - max_value) in the vector
//  -> Sum these exponentials
//  -> Normalize each exponential by dividing it by the sum of all exponentials
// To handle the case where the input can be either int or float in the softmaxOnHost function, C++ templates are used.
// Templates allow the function to accept a vector of any numeric type.
template <typename T>
void softmaxOnHost(const std::vector<T> &input, std::vector<double> &output)
{
    // Find the maximum value in the input vector for numerical stability
    T max_input = *std::max_element(input.begin(), input.end());

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

// Perform the convolution operation in PIM between the filter matrix and the input matrix.
// The function allocates the necessary PIM objects, broadcasts filter values to PIM objects, and performs
// element-wise multiplication followed by summation to produce the output.
// The summed results are then copied from the PIM (device) to Host.
void performConv(std::vector<std::vector<int>> &filterMatrix, std::vector<std::vector<int>> &inputMatrix, std::vector<int> &outputMatrix, int numRequiredPIMRows, int numRequiredPIMCol)
{
  std::vector<PimObjId> filterObjects;
  std::vector<int> temp;
  PimObjId obj1 = pimAlloc(PIM_ALLOC_AUTO, numRequiredPIMCol, PIM_INT32);
  if (obj1 == -1)
  {
    std::cout << "Function: " << __func__ << "Abort: pimAlloc failed for obj1" << std::endl;
    return;
  }
  filterObjects.push_back(obj1);
  for (int i = 1; i < numRequiredPIMRows; i++)
  {
    PimObjId obj = pimAllocAssociated(filterObjects[0], PIM_INT32);
    if (obj == -1)
    {
      std::cout << "Function: " << __func__ << "Abort: pimAllocAssociated failed for obj at i=" << i << std::endl;
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
        std::cout << "Function: " << __func__ << "Abort: pimBroadcastInt failed between filterObjects and filterMatrix at i=" << i << ", j=" << j << std::endl;
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
      std::cout << "Function: " << __func__ << "Abort: pimAllocAssociated failed for obj at i=" << i << std::endl;
      return;
    }
    matrixObjects.push_back(obj);
  }

  for (int i = 0; i < inputMatrix.size(); i++)
  {
    PimStatus status = pimCopyHostToDevice((void *)inputMatrix[i].data(), matrixObjects[i]);
    if (status != PIM_OK)
    {
      std::cout << "Function: " << __func__ << "Abort: pimCopyHostToDevice failed between inputMatrix and matrixObjects at i=" << i << std::endl;
      return;
    }
  
    status = pimMul(matrixObjects[i], filterObjects[i], filterObjects[i]);
    if (status != PIM_OK)
    {
      std::cout << "Function: " << __func__ << "Abort: pimMul failed for matrixObjects and filterObjects at i=" << i << std::endl;
      return;
    }
  }

  for (int i = 1; i < numRequiredPIMRows; i++)
  {
    PimStatus status = pimAdd(filterObjects[0], filterObjects[i], filterObjects[0]);
    if (status != PIM_OK)
    {
      std::cout << "Function: " << __func__ << "Abort: pimAdd failed between filterObjects[0] and filterObjects[i] at i=" << i << std::endl;
      return;
    }
  }

  outputMatrix.resize(numRequiredPIMCol);

  PimStatus status = pimCopyDeviceToHost(filterObjects[0], outputMatrix.data());
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << "Abort: pimCopyDeviceToHost failed between filterObjects[0] and outputMatrix" << std::endl;
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

// Function executing the convolution between a 3D input matrix and a 3D kernel based on a given stride and padding.
// For each kernel filter, the input matrix is decomposed based on the kernel dimensions and stride, merged together.
// The merged matrix is then passed to performConv() function along with the kernel to peform the convolution on PIM.
// The result (1D vector) from the PIM is then  reconstructed back to a 2D final result matrix for each filter.
void conv2(std::vector<std::vector<std::vector<int>>> &inputMatrix, std::vector<std::vector<std::vector<int>>> &kernelMatrix, std::vector<std::vector<std::vector<int>>> &resultMatrix, int stride, int padding)
{
  PimDeviceProperties deviceProp;
  PimStatus status = pimGetDeviceProperties(&deviceProp);
  if (status != PIM_OK) {
    std::cout << "Abort: pimGetDeviceProperties failed" << std::endl;
    exit(1);
  }
  // Get the device parameters
  uint64_t numCols = deviceProp.numColPerSubarray;
  uint64_t numRows = deviceProp.numRowPerSubarray;
  uint64_t numOfBits = uint64_t(deviceProp.numRanks) * uint64_t(deviceProp.numBankPerRank) * uint64_t(deviceProp.numSubarrayPerBank) * numCols * numRows;  

  int inputDepth = inputMatrix.size();
  int inputHeight = inputMatrix[0].size();
  int inputWidth = inputMatrix[0][0].size();
  int kernelDepth = kernelMatrix.size();
  int kernelHeight = kernelMatrix[0].size();
  int kernelWidth = kernelMatrix[0][0].size();

  int outMatRow = std::floor((inputHeight - kernelHeight) / stride) + 1;
  int outMatCol = std::floor((inputWidth - kernelWidth) / stride) + 1;   
  int numOfMatPerRow = floor((1.0 * numOfBits) / (outMatRow * outMatCol)) <  inputDepth ? floor((1.0 * numOfBits) / (outMatRow * outMatCol)) : inputDepth;
  int numOfPIMRow = kernelHeight * kernelWidth;

  resultMatrix.resize(kernelDepth, std::vector<std::vector<int>>(outMatRow, std::vector<int>(outMatCol)));

  for (int i = 0; i < kernelDepth; i++)
  {
    int tempcol = 0;
    std::vector<int> dstVec(outMatRow * outMatCol);
    for (int j = 0; j < inputDepth; j += numOfMatPerRow)
    {
      int matChunk = (numOfMatPerRow + j) <= inputDepth ? (numOfMatPerRow + j) : inputDepth;

      std::vector<std::vector<int>> mergedMat(numOfPIMRow);
      for (int k = j; k < matChunk; k++)
      {
        std::vector<std::vector<int>> decompMat;
	      decomposeMatrix(inputHeight, inputWidth, kernelMatrix[i].size(), kernelMatrix[i][0].size(), stride, 0, inputMatrix[k], decompMat);
        // Merge the matrices
        for (int idx = 0; idx < mergedMat.size(); idx++) {
          mergedMat[idx].insert(mergedMat[idx].end(),
                                std::make_move_iterator(decompMat[idx].begin()),
                                std::make_move_iterator(decompMat[idx].end()));
        }
        tempcol = mergedMat[0].size();     
      }

      std::vector<int> outVector;
      performConv(kernelMatrix[i], mergedMat, outVector, numOfPIMRow, tempcol);

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
    
    }
    int ddx = 0;
    for (int rdx = 0; rdx < outMatRow; ++rdx)
    {
      for (int cdx = 0; cdx < outMatCol; ++cdx)
      {
        resultMatrix[i][rdx][cdx] = dstVec[ddx++];
      }
    }
  }  
}

// This should work for bitSIMD or any PIM that requires vertical data layout.
// Perform the Max Pool operation in PIM for the given input matrix.
// The function allocates the necessary PIM objects, copies the data from host to PIM, and performs Max operation.
// The max results from pimObjectList[0] are then copied from the PIM (device) to Host.
void maxPool(const std::vector<std::vector<int>> &inputMatrix, std::vector<int> &outputMatrix)
{
  if (inputMatrix.empty())
  {
    return;
  }
  int numRows = inputMatrix.size();
  int numCols = inputMatrix[0].size();

  std::vector<PimObjId> pimObjectList(numRows);
  PimObjId obj1 = pimAlloc(PIM_ALLOC_AUTO, numCols, PIM_INT32);
  if (obj1 == -1)
  {
    std::cout << "Function: " << __func__ << "Abort: pimAlloc failed for obj1" << std::endl;
    return;
  }
  pimObjectList[0] = obj1;
  for (int i = 1; i < numRows; i++)
  {
    PimObjId obj = pimAllocAssociated(pimObjectList[0], PIM_INT32);
    if (obj == -1)
    {
      std::cout << "Function: " << __func__ << "Abort: pimAllocAssociated failed for obj at i=" << i << std::endl;
      return;
    }
    pimObjectList[i] = obj;
  }

  for (int i = 0; i < pimObjectList.size(); i++)
  {
    PimStatus status = pimCopyHostToDevice((void *)inputMatrix[i].data(), pimObjectList[i]);
    if (status != PIM_OK)
    {
      std::cout << "Function: " << __func__ << "Abort: pimCopyHostToDevice failed between inputMatrix and pimObjectList at i=" << i << std::endl;
      return;
    }
  }

  for (int i = 1; i < pimObjectList.size(); i++)
  {
    PimStatus status = pimMax(pimObjectList[0], pimObjectList[i], pimObjectList[0]);
    if (status != PIM_OK)
    {
      std::cout << "Function: " << __func__ << "Abort: pimMax failed between pimObjectList[0] and pimObjectList[i] at i=" << i << std::endl;
      return;
    }
  }
  outputMatrix.resize(numCols);
  PimStatus status = pimCopyDeviceToHost(pimObjectList[0], outputMatrix.data());
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << "Abort: pimCopyDeviceYoHost failed between pimObjectList[0] and outputMatrix" << std::endl;
    return;
  }
  for (auto elem : pimObjectList)
  {
    pimFree(elem);
  }
}

// Function executing the Max Pool between a 3D input matrix based on a kernel dimensions and a given stride.
// For each input depth, the input matrix is decomposed based on the kernel dimensions and stride, merged together.
// The merged matrix is then passed to maxPool() function to peform max pooling on PIM.
// The result (1D vector) from the PIM is then reconstructed back to a 2D final result matrix for each input depth.
void pool(std::vector<std::vector<std::vector<int>>> &inputMatrix, int kernelHeight, int kernelWidth, int stride, std::vector<std::vector<std::vector<int>>> &resultMatrix)
{
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

  uint64_t inputDepth = inputMatrix.size();
  uint64_t inputHeight = inputMatrix[0].size();
  uint64_t inputWidth = inputMatrix[0][0].size();
  uint64_t outputHeight = (inputHeight - kernelHeight) / stride + 1;
  uint64_t outputWidth = (inputWidth - kernelWidth) / stride + 1;

  uint64_t numOfPIMRow = kernelHeight * kernelWidth;
  uint64_t numOfPIMColumn = outputHeight * outputWidth;
  uint64_t numOfMatPerRow = floor((1.0 * numOfBits) / numOfPIMColumn) < inputDepth ? floor((1.0 * numOfBits) / (numOfPIMColumn)) : inputDepth;

  resultMatrix.resize(inputDepth, std::vector<std::vector<int>>(outputHeight, std::vector<int>(outputWidth)));

  for (uint64_t i = 0; i < inputDepth; i += 1)
  {
    // This vector packs all the matrices that can be fit into one PIM iteration
    std::vector<std::vector<int>> mergedMat(numOfPIMRow);
    uint64_t matChunk = (numOfMatPerRow + i) <= inputDepth ? (numOfMatPerRow + i) : inputDepth;
    for (uint64_t j = i; j < matChunk; j++)
    {
      std::vector<std::vector<int>> decompMat;
      decomposeMatrix(inputHeight, inputWidth, kernelHeight, kernelWidth, stride, 0, inputMatrix[j], decompMat);
      // Merge the matrices
      for (uint64_t idx = 0; idx < mergedMat.size(); idx++) {
        mergedMat[idx].insert(mergedMat[idx].end(),
                             std::make_move_iterator(decompMat[idx].begin()),
                             std::make_move_iterator(decompMat[idx].end()));
      }
    }

    std::vector<int> outVector;
    maxPool(mergedMat, outVector);

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
}

// Performs General Matrix-Vector Multiplication (GEMV) in PIM
// The function computes the matrix-vector product of the source matrix and the source vector, and stores the result in the destination vector.
// The function allocates PIM objects, performs element-wise multiplication, and accumulates the results.
// The accumulated results are then copied from the PIM (device) to the host. 
void gemv(uint64_t row, uint64_t col, std::vector<int> &srcVector, std::vector<std::vector<int>> &srcMatrix, std::vector<int> &dst)
{
  PimObjId srcObj = pimAlloc(PIM_ALLOC_AUTO, row, PIM_INT32);
  if (srcObj == -1)
  {
    std::cout << "Function: " << __func__ << ", Abort: pimAlloc failed for srcObj" << std::endl;
    return;
  }

  PimObjId dstObj = pimAllocAssociated(srcObj, PIM_INT32);
  if (dstObj == -1)
  {
    std::cout << "Function: " << __func__ << ", Abort: pimAllocAssociated failed for dstObj" << std::endl;
    return;
  }

  PimStatus status = pimBroadcastInt(dstObj, 0);
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << ", Abort: pimBroadcastInt failed for dstObj" << std::endl;
    return;
  }

  for (int i = 0; i < col; ++i)
  {
    status = pimCopyHostToDevice((void *)srcMatrix[i].data(), srcObj);
    if (status != PIM_OK)
    {
      std::cout << "Function: " << __func__ << ", Abort: pimCopyHostToDevice failed between srcMatrix and srcObj at i=" << i << std::endl;
      return;
    }

    status = pimScaledAdd(srcObj, dstObj, dstObj, srcVector[i]);
    if (status != PIM_OK)
    {
      std::cout << "Function: " << __func__ << ", Abort: pimScaledAdd failed between srcObj1 and srcObj2 at i=" << i << std::endl;
      return;
    }
  }

  dst.resize(row);
  status = pimCopyDeviceToHost(dstObj, (void *)dst.data());
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << ", Abort: pimCopyDeviceToHost failed between dstObj and dst" << std::endl;
  }
  pimFree(srcObj);
  pimFree(dstObj);
}

// Perform the RELU (REctified Linear Unit) operation, max(0, x), a non-linear activation function in PIM for the given 1D input vector; called after GEMV operation.
// The function allocates the necessary PIM objects, copies the data from host to PIM, and performs Max operation.
// The max results from pimObject are then copied from the PIM (device) to Host.
void performRelu(std::vector<int> &inputVector)
{
  if (inputVector.empty()) {
    std::cout << "Function: " << __func__ << ", Abort: Input matrix is empty" << std::endl;    
    return;
  }
  int numCols = inputVector.size();

  // Initialize reluConst vector with zero for max(0, x) operation.
  std::vector<int> reluConst(numCols, 0);  

  PimObjId pimObject = pimAlloc(PIM_ALLOC_AUTO, numCols, PIM_INT32);
  if (pimObject == -1) {
    std::cout << "Function: " << __func__ << ", Abort: pimAlloc for PimObj pimObject failed" << std::endl;
    return;
  }

  PimObjId RELUConstObj = pimAllocAssociated(pimObject, PIM_INT32);
  if (RELUConstObj == -1) {
    std::cout << "Function: " << __func__ << ", Abort: pimAllocAssociated for PimObj RELUConstObj failed" << std::endl;
    return;
  }

  PimStatus status = pimCopyHostToDevice((void *)inputVector.data(), pimObject);
  if (status != PIM_OK) {
    std::cout << "Function: " << __func__ << ", Abort: pimCopyHostToDevice from inputMatrix to pimObject failed" << std::endl;
    return;
  }

  status = pimCopyHostToDevice((void *)reluConst.data(), RELUConstObj);
  if (status != PIM_OK) {
    std::cout << "Function: " << __func__ << ", Abort: pimCopyHostToDevice from reluConst to RELUConstObj failed" << std::endl;
    return;
  }  

  status = pimMax(RELUConstObj, pimObject, pimObject);
  if (status != PIM_OK) {
    std::cout << "Function: " << __func__ << ", Abort: pimMax failed between RELUConstObj and pimObject" << std::endl;
    return;
  }

  inputVector.resize(numCols);
  status = pimCopyDeviceToHost(pimObject, inputVector.data());
  if (status != PIM_OK) {
    std::cout << "Function: " << __func__ << ", Abort: pimCopyDeviceToHost from pimObject to outputMatrix" << std::endl;
    return;
  }

  pimFree(pimObject);
  pimFree(RELUConstObj);

}

// Perform the RELU (REctified Linear Unit) operation, max(0, x), a non-linear activation function in PIM for the given 2D input matrix.
// The function allocates the necessary PIM objects, copies the data from host to PIM, and performs Max operation.
// The max results from pimObjectList[0] are then copied from the PIM (device) to Host.
void performRelu(const std::vector<std::vector<int>> &inputMatrix, std::vector<int> &outputMatrix)
{
  if (inputMatrix.empty())
  {    
    std::cout << "Function: " << __func__ << ", Abort: Input matrix is empty" << std::endl;
    return;
  }
  int numRows = inputMatrix.size();
  int numCols = inputMatrix[0].size();
  // Initialize reluConst vector with zero for max(0, x) operation. Initialize with a different value 'y' for max(y, x) operation.
  std::vector<int> reluConst(numCols, 0);  

  std::vector<PimObjId> pimObjectList(numRows);
  PimObjId obj1 = pimAlloc(PIM_ALLOC_AUTO, numCols, PIM_INT32);
  if (obj1 == -1)
  {
    std::cout << "Function: " << __func__ << ", Abort: pimAlloc for PimObj obj1 failed" << std::endl;
    return;
  }
  pimObjectList[0] = obj1;
  for (int i = 1; i < numRows; i++)
  {
    PimObjId obj = pimAllocAssociated(pimObjectList[0], PIM_INT32);
    if (obj == -1)
    {
      std::cout << "Function: " << __func__ << ", Abort: pimAllocAssociated for PimObj obj failed" << std::endl;
      return;
    }
    pimObjectList[i] = obj;
  }
  PimObjId RELUConstObj = pimAllocAssociated(pimObjectList[0], PIM_INT32);
  if (RELUConstObj == -1)
  {
      std::cout << "Function: " << __func__ << ", Abort: pimAllocAssociated for PimObj RELUConstObj failed" << std::endl;
      return;
  }

  for (int i = 0; i < pimObjectList.size(); i++)
  {
    PimStatus status = pimCopyHostToDevice((void *)inputMatrix[i].data(), pimObjectList[i]);
    if (status != PIM_OK)
    {
      std::cout << "Function: " << __func__ << ", Abort: pimCopyHostToDevice from inputMatrix[" << i << "] to pimObjectList[" << i << "] failed" << std::endl;
      return;
    }
  }
  PimStatus status = pimCopyHostToDevice((void *) reluConst.data(), RELUConstObj);
  if (status != PIM_OK)
  {
      std::cout << "Function: " << __func__ << ", Abort: pimCopyHostToDevice from reluConst to RELUConstObj failed" << std::endl;
      return;
  }  

  for (int i = 0; i < pimObjectList.size(); i++)
  {
    PimStatus status = pimMax(RELUConstObj, pimObjectList[i], pimObjectList[0]);
    if (status != PIM_OK)
    {
      std::cout << "Function: " << __func__ << ", Abort: pimMax failed between RELUConstObj and pimObjectList[" << i << "]" << std::endl;
      return;
    }
  }
  outputMatrix.resize(numCols);
  status = pimCopyDeviceToHost(pimObjectList[0], outputMatrix.data());
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << ", Abort: pimCopyDeviceToHost from pimObjectList[0] to outputMatrix" << std::endl;
    return;
  }
  for (auto elem : pimObjectList)
  {
    pimFree(elem);
  }
  pimFree(RELUConstObj);

}

// Function executing RELU for a given 3D input matrix.
// For each input depth, the input matrix is decomposed based on the kernel dimensions (1x1) and stride (1), merged together.
// The merged matrix is then passed to performRelu() function to peform RELU on PIM.
// The result (1D vector) from the PIM is then reconstructed back to a 2D final result matrix for each input depth.
void relu (std::vector<std::vector<std::vector<int>>> &inputMatrix) {
  
  // Define parameters for processing
  PimDeviceProperties deviceProp;
  PimStatus status = pimGetDeviceProperties(&deviceProp);
  if (status != PIM_OK) {
    std::cout << "Abort: pimGetDeviceProperties failed" << std::endl;
    exit(1);
  }
  // Get the device parameters
  uint64_t numCols = deviceProp.numColPerSubarray;
  uint64_t numRows = deviceProp.numRowPerSubarray;
  uint64_t numOfBits = uint64_t(deviceProp.numRanks) * uint64_t(deviceProp.numBankPerRank) * uint64_t(deviceProp.numSubarrayPerBank) * numCols * numRows;

  // Calculate matrix dimensions
  uint64_t inputDepth = inputMatrix.size();
  uint64_t inputHeight = inputMatrix[0].size();
  uint64_t inputWidth = inputMatrix[0][0].size();

  uint64_t numOfPIMRow = 1;
  unsigned bitsPerElement = 32;
  uint64_t numOfMatPerRow = std::min(static_cast<uint64_t>(std::floor((1.0 * numOfBits) / (inputHeight * inputWidth * bitsPerElement))), static_cast<uint64_t>(inputWidth));

  std::vector<std::vector<int>> decompMat;
  std::vector<std::vector<int>> mergedMat(numOfPIMRow);
  std::vector<int> outVector;
  outVector.resize(inputDepth * inputHeight * inputWidth);
  
  // Loop through input depth in chunks
  for (uint64_t j = 0; j < inputDepth; j += numOfMatPerRow) {
    uint64_t matChunk = (numOfMatPerRow + j) <= inputDepth ? (numOfMatPerRow + j) : inputDepth;
    // Decompose and merge matrices
    for (uint64_t k = j; k < matChunk; k++) {
      // 1, 1, 1, 0 in the function call below indicates that the kernel dimensions are 1x1, stride is 1 and padding is 0 while decomposing the matrix for ReLU operation.
      // The kernel dimensions, stride and padding are specified in the functional call to make the function reusable for other operations like max pooling.
      decomposeMatrix(inputHeight, inputWidth, 1, 1, 1, 0, inputMatrix[k], decompMat);
      for (uint64_t idx = 0; idx < mergedMat.size(); idx++) {
        mergedMat[idx].reserve(mergedMat[idx].size() + decompMat[idx].size());
        mergedMat[idx].insert(mergedMat[idx].end(), make_move_iterator(decompMat[idx].begin()), make_move_iterator(decompMat[idx].end()));
      }
    }
  }
  
  performRelu(mergedMat, outVector);

  uint64_t idx = 0;
  for (int i = 0; i < inputDepth; i += 1)
  {  
    for (int r = 0; r < inputHeight; ++r)
    {
      for (int c = 0; c < inputWidth; ++c)
      {
        inputMatrix[i][r][c] = outVector[idx++];
      }
    }
  } 

}

// Function to read weights of a specific layer from CSV (for convolutional layers).
vector<vector<vector<float>>> read_conv_layer_weights_from_csv(const string& filename, const string& layer_name) {
    ifstream file(filename); // Open the CSV file
    string line;
    vector<vector<vector<float>>> kernelMatrix; // Initialize a 3D vector to hold the kernel matrix

    while (getline(file, line)) { // Read the file line by line
        stringstream ss(line);
        string item;
        vector<string> items;

        // Split the line by commas
        while (getline(ss, item, ',')) {
            items.push_back(item);
        }

        // Check if the layer name matches the current line
        if (items[0] == layer_name) {
            int num_kernels = stoi(items[1]); // Depth of the kernel matrix (number of kernels)
            int rows = stoi(items[3]);  // Number of rows in each kernel
            int cols = stoi(items[4]);  // Number of columns in each kernel
            kernelMatrix.resize(num_kernels); // Resize the 3D vector to accommodate the kernels

            int idx = 5; // Start index of the actual weights in the CSV line
            for (int d = 0; d < num_kernels; ++d) { // Loop through each kernel
                kernelMatrix[d].resize(rows, vector<float>(cols, 0)); // Resize each kernel matrix
                for (int r = 0; r < rows; ++r) { // Loop through each row
                    for (int c = 0; c < cols; ++c) { // Loop through each column
                        kernelMatrix[d][r][c] = std::stof(items[idx++]); // Assign the weight value
                    }
                }
            }

            return kernelMatrix; // Return the kernel matrix for the specified layer
        }
    }

    throw runtime_error("Layer not found in the CSV file"); // Throw an error if the layer is not found
}

// Function to read weights of a specific layer from CSV (for dense layers).
std::vector<std::vector<float>> read_dense_layer_weights_from_csv(const std::string& filename, const std::string& layer_name) {
    std::ifstream file(filename); // Open the CSV file
    std::string line; // Variable to hold each line of the CSV
    std::vector<std::vector<float>> denseMatrix; // Matrix to store the weights of the dense layer

    while (std::getline(file, line)) {
        std::stringstream ss(line); // Create a string stream from the line
        std::string item; // Variable to hold each item in the line
        std::vector<std::string> items; // Vector to store all items in the line

        // Split the line by commas
        while (std::getline(ss, item, ',')) {
            items.push_back(item);
        }

        // Check if the layer name matches the current line
        if (items[0] == layer_name) {
            int rows = std::stoi(items[2]); // Number of rows in the dense matrix
            int cols = std::stoi(items[1]); // Number of columns in the dense matrix
            denseMatrix.resize(rows, std::vector<float>(cols, 0)); // Resize the matrix to the appropriate dimensions

            int idx = 3; // Start index of the actual weights in the CSV line
            // Populate the dense matrix with weights
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    denseMatrix[r][c] = std::stof(items[idx++]); // Convert the string weight to float and store it in the matrix
                }
            }

            return denseMatrix;
        }
    }

    throw std::runtime_error("Layer not found in the CSV file"); // Throw an error if the layer name is not found
}

// Function to binarize a 3D matrix of floats.
// Added to binarize the values of kernel matrices of the convolutional layers. May not be needed after adding float support.
std::vector<std::vector<std::vector<int>>> binarizeMatrix(const std::vector<std::vector<std::vector<float>>>& weights) {
  std::vector<std::vector<std::vector<int>>> binarizedMatrix(weights.size());
    for (size_t d = 0; d < weights.size(); ++d) {
      binarizedMatrix[d].resize(weights[d].size());
      for (size_t r = 0; r < weights[d].size(); ++r) {
        binarizedMatrix[d][r].resize(weights[d][r].size());
        for (size_t c = 0; c < weights[d][r].size(); ++c) {
          binarizedMatrix[d][r][c] = (weights[d][r][c] > 0) ? 1 : 0;
        }
      }
    }
  return binarizedMatrix;
}

// Function to binarize a 2D matrix of floats.
// Added to binarize the values of kernel matrices of the dense layers. May not be needed after adding float support.
std::vector<std::vector<int>> binarizeMatrix(const std::vector<std::vector<float>>& weights) {
  std::vector<std::vector<int>> binarizedMatrix(weights.size());
    for (size_t r = 0; r < weights.size(); ++r) {
      binarizedMatrix[r].resize(weights[r].size());
      for (size_t c = 0; c < weights[r].size(); ++c) {
        binarizedMatrix[r][c] = (weights[r][c] > 0) ? 1 : 0;
      }
    }
  return binarizedMatrix;
}

#ifdef COMPILE_WITH_JPEG
// Function to read a JPEG image and store its pixel data into inputMatrix.
void readJPEG(const std::string& filename, std::vector<std::vector<std::vector<int>>>& inputMatrix, int &targetHeight, int &targetWidth) {
  // Open the JPEG file
  FILE* file = fopen(filename.c_str(), "rb");
  if (!file) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }

  // Initialize the JPEG decompression object
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  // Specify the input file
  jpeg_stdio_src(&cinfo, file);

  // Read JPEG header
  jpeg_read_header(&cinfo, TRUE);

  // Set parameters for decompression
  cinfo.out_color_space = JCS_RGB; // RGB color space

  // Start decompression
  jpeg_start_decompress(&cinfo);

  // Allocate memory for storing scanline of decompressed image
  int row_stride = cinfo.output_width * cinfo.output_components;
  JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

  // Calculate scaling factors
  double scaleX = (double)cinfo.output_width / targetWidth;
  double scaleY = (double)cinfo.output_height / targetHeight;

  // Resize the image
  inputMatrix.resize(3, std::vector<std::vector<int>>(targetHeight, std::vector<int>(targetWidth)));

  int row = 0;
  while (cinfo.output_scanline < cinfo.output_height) {
    // Read scanline
    jpeg_read_scanlines(&cinfo, buffer, 1);
    if (row < targetHeight) {
      // Process each pixel in the scanline
      for (int col = 0; col < targetWidth; ++col) {
        int origX = (int)(col * scaleX);
        if (origX >= cinfo.output_width) origX = cinfo.output_width - 1;
        for (int color = 0; color < 3; ++color) {
          inputMatrix[color][row][col] = buffer[0][origX * cinfo.output_components + color];
        }
      }
    }
    ++row;
  }

  // Finish decompression
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  // Close the file
  fclose(file);
}

// Function to write the resized image data to a JPEG file.
// Added to verify the correctness of the resized image.
void writeResizedImage(const std::string& outputFilename, const std::vector<std::vector<std::vector<int>>>& inputMatrix) {
  // Get dimensions of the resized image
  int depth = inputMatrix.size(); // Should be 3 (R, G, B)
  int height = inputMatrix[0].size(); // Should be 224
  int width = inputMatrix[0][0].size(); // Should be 224

  // Create a JPEG file pointer
  FILE* outfile = fopen(outputFilename.c_str(), "wb");
  if (!outfile) {
    std::cerr << "Error opening output JPEG file: " << outputFilename << std::endl;
    return;
  }

  // Initialize the JPEG compression object
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  // Specify the output file
  jpeg_stdio_dest(&cinfo, outfile);

  // Set image parameters
  cinfo.image_width = width;
  cinfo.image_height = height;
  cinfo.input_components = depth; // Number of color components (RGB)
  cinfo.in_color_space = JCS_RGB; // Colorspace of input image

  // Set default compression parameters
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, 90, TRUE); // Adjust quality settings here (optional)

  // Start compression
  jpeg_start_compress(&cinfo, TRUE);

  // Write scanlines of image data
  JSAMPROW row_pointer = new JSAMPLE[width * depth]; // Allocate memory for one row

  while (cinfo.next_scanline < cinfo.image_height) {
    // Fill row_pointer with RGB pixel values from inputMatrix
    for (int w = 0; w < width; ++w) {
      for (int d = 0; d < depth; ++d) {
        row_pointer[w * depth + d] = (JSAMPLE)inputMatrix[d][cinfo.next_scanline][w];
      }
    }
    // Write scanline to the JPEG file
    jpeg_write_scanlines(&cinfo, &row_pointer, 1);
  }

  // Finish compression
  jpeg_finish_compress(&cinfo);

  // Clean up
  jpeg_destroy_compress(&cinfo);
  fclose(outfile);

  // Deallocate the row pointer
  delete[] row_pointer;

  std::cout << "Resized image saved as JPEG: " << outputFilename << std::endl;
}
#endif // ifdef COMPILE_WITH_JPEG

#endif
