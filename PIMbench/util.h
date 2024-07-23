// PIMeval Simulator - Application Utilities
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef PIM_FUNC_SIM_APPS_UTIL_H
#define PIM_FUNC_SIM_APPS_UTIL_H

#include <iostream>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <vector>
#include <iomanip>
#include <chrono>
#include <random>

#include "libpimeval.h"
#include <map>
#include <fstream>

using namespace std;

void getVector(uint64_t vectorLength, std::vector<int> &srcVector)
{
  srand((unsigned)time(NULL));
  srcVector.resize(vectorLength);
  #pragma omp parallel for
  for (uint64_t i = 0; i < vectorLength; ++i)
  {
    srcVector[i] = rand() % (2 * (i + 1)) - (i + 1);
  }
}

void getVectorFP32(uint64_t vectorLength, std::vector<float> &srcVector, bool nonZero)
{
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(-1e10, 1e10);

  srcVector.resize(vectorLength);
  #pragma omp parallel for
  for (uint64_t i = 0; i < vectorLength; ++i) {
    float val = 0.0;
    do {
      val = dist(gen);
      srcVector[i] = dist(gen);
    } while (nonZero && val == 0.0);
  }
}

void getMatrix(int row, int column, int padding, std::vector<std::vector<int>> &inputMatrix)
{
  srand((unsigned)time(NULL));
  inputMatrix.resize(row + 2 * padding, std::vector<int>(column + 2 * padding, 0));
  #pragma omp parallel for
  for (int i = padding; i < row + padding; ++i)
  {
    for (int j = padding; j < column + padding; ++j)
    {
      inputMatrix[i][j] = rand() % (2 * (i + 1)) - (i + 1);
    }
  }
}

void addPadding(int row, int column, int padding, std::vector<std::vector<int>> &inputMatrix, std::vector<std::vector<int>> &resultMatrix)
{
  resultMatrix.resize(row + 2 * padding, std::vector<int>(column + 2 * padding, 0));
#pragma omp parallel for
  for (int i = 0; i < row; ++i)
  {
    for (int j = 0; j < column; ++j)
    {
      resultMatrix[i + 1][j + 1] = inputMatrix[i][j];
    }
  }
}

void flatten3DMat(std::vector<std::vector<std::vector<int>>>& inputMatrix, std::vector<int>& flattenedVector)
{
  for (const auto &matrix : inputMatrix)
  {
    for (const auto &row : matrix)
    {
      for (int element : row)
      {
        flattenedVector.push_back(element);
      }
    }
  }
}

bool createDevice(char *configFile)
{
  if (configFile == nullptr)
  {
    // Each rank has 8 chips; Total Bank = 16; Each Bank contains 32 subarrays;
    unsigned numRanks = 2;
    unsigned numBankPerRank = 128; // 8 chips * 16 banks
    unsigned numSubarrayPerBank = 32;
    unsigned numRows = 1024;
    unsigned numCols = 8192;

    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return false;
    }
  }
  else
  {
    std::cout << "Creating device from config file is not supported yet.\n";
    return 0;
    PimStatus status = pimCreateDeviceFromConfig(PIM_FUNCTIONAL, configFile);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return false;
    }
  }
  return true;
}

// Function to print the dimensions of a 2D matrix
inline void printMatrixDimensions (std::vector<std::vector<int>> &inputMatrix) {     
  std::cout << inputMatrix.size() << " x " 
            << inputMatrix[0].size() 
            << std::endl;
}

// Function to print the dimensions of a 3D matrix
inline void printMatrixDimensions (std::vector<std::vector<std::vector<int>>> &inputMatrix) {
  std::cout << inputMatrix.size() << " x " 
            << inputMatrix[0].size() << " x " 
            << inputMatrix[0][0].size() 
            << std::endl;   
}

// Function to print a vector
void printVector(std::vector<int>& vec) {
  for (auto val : vec) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
}

// Function to print a 2D matrix
void printMatrix(std::vector<std::vector<int>>& matrix) {
  for (const auto& row : matrix) {
    for (int val : row) {
      std::cout << val << " ";
    }
    std::cout << std::endl;
  }
}

// Function to print a 3D matrix
void printMatrix(std::vector<std::vector<std::vector<int>>>& matrix) {
  for (const auto& mat2D : matrix) {
    for (const auto& row : mat2D) {
      for (int val : row) {
        std::cout << val << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "---" << std::endl; // Separator between 2D matrices
  }
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

#endif
