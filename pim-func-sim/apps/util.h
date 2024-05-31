// PIM Functional Simulator - Application Utilities
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef PIM_FUNC_SIM_APPS_UTIL_H
#define PIM_FUNC_SIM_APPS_UTIL_H

#include <iostream>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <vector>
#include <iomanip>
#include <chrono>

#include "libpimsim.h"
using namespace std;

void getVector(uint64_t vectorLength, std::vector<int> &srcVector)
{
  srand((unsigned)time(NULL));
  srcVector.resize(vectorLength);
#pragma omp parallel for
  for (int i = 0; i < vectorLength; ++i)
  {
    srcVector[i] = (rand() % (i + 1) + 1);
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
      inputMatrix[i][j] = (rand() % ((i * j) + 1)) + 1;
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
    unsigned numBanks = 128; // 8 chips * 16 banks
    unsigned numSubarrayPerBank = 32;
    unsigned numRows = 8192;
    unsigned numCols = 8192;
    
    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numBanks, numSubarrayPerBank, numRows, numCols);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return false;
    }
  }
  else
  {
    PimStatus status = pimCreateDeviceFromConfig(PIM_FUNCTIONAL, configFile);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return false;
    }
  }
  return true;
}

//Bit serial PIM performs reduction operation on host
void reductionSumBitSerial(const std::vector<int> &reductionVector, int &reductionValue, std::chrono::duration<double, std::milli> &reductionTime)
{
  auto start = std::chrono::high_resolution_clock::now();
  reductionValue = 0;
#pragma omp parallel for reduction(+ : sum)
  for (size_t i = 0; i < reductionVector.size(); ++i)
  {
    reductionValue += reductionVector[i];
  }
  auto end = std::chrono::high_resolution_clock::now();
  reductionTime += (end - start);
}

#endif
