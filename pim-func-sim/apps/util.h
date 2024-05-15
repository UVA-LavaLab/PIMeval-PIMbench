// PIM Functional Simulator - Application Utilities
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef PIM_FUNC_SIM_APPS_UTIL_H
#define PIM_FUNC_SIM_APPS_UTIL_H

#include <iostream>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <vector>

#include "libpimsim.h"
using namespace std;

#ifndef DATA_TYPE
typedef int32_t data_t;
#else
typedef DATA_TYPE data_t;
#endif

void getVector(uint64_t vectorLength, std::vector<int>& srcVector) {
  srand((unsigned)time(NULL));
  srcVector.resize(vectorLength);
  #pragma omp parallel for
  for (int i = 0; i < vectorLength; ++i)
  {
    srcVector[i] = rand() % (i+1);
  }
}

void getMatrix(int row, int column, int padding, std::vector<std::vector<int>>& inputMatrix) {
    srand((unsigned)time(NULL));
    inputMatrix.resize(row + 2 * padding, std::vector<int>(column + 2 * padding, 0));
    #pragma omp parallel for
    for (int i = padding; i < row + padding; ++i) {
        for (int j = padding; j < column + padding; ++j) {
            inputMatrix[i][j] = rand() % ((i * j) + 1);
        }
    }
}

bool createDevice(char *configFile)
{
  if (configFile == nullptr)
  {
    unsigned numCores = 16;
    unsigned numRows = 8192;
    unsigned numCols = 65536;
    PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numCores, numRows, numCols);
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

#endif
