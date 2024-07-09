// PIM Functional Simulator - Application Utilities for Floating <-> Fixed-Point Conversions. Currently used for CNNs only.
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef PIM_FUNC_SIM_APPS_UTIL_FIXEDPOINT_H
#define PIM_FUNC_SIM_APPS_UTIL_FIXEDPOINT_H

#include <iostream>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <vector>
#include <iomanip>
#include <chrono>
#include <random>

#include "libpimsim.h"
#include <map>
#include <fstream>

using namespace std;

// Define the number of fractional bits for fixed-point representation
const int FRACTIONAL_BITS = 5;
const int FIXED_POINT_SCALING_FACTOR = 1 << FRACTIONAL_BITS; // 2^FRACTIONAL_BITS

// Function to convert a 3D matrix of floats to a 3D matrix of fixed-point integers.
std::vector<std::vector<std::vector<int>>> floatToFixed(std::vector<std::vector<std::vector<float>>>& inputMatrix) {
  std::vector<std::vector<std::vector<int>>> fixedMatrix(inputMatrix.size(),
    std::vector<std::vector<int>>(inputMatrix[0].size(),
      std::vector<int>(inputMatrix[0][0].size())));

  size_t depth = inputMatrix.size();
  size_t height = inputMatrix[0].size();
  size_t width = inputMatrix[0][0].size();

  #pragma omp parallel for collapse(3)
  for (size_t i = 0; i < depth; ++i) {
    for (size_t j = 0; j < height; ++j) {
      for (size_t k = 0; k < width; ++k) {
        fixedMatrix[i][j][k] = static_cast<int>(inputMatrix[i][j][k] * FIXED_POINT_SCALING_FACTOR);
      }
    }
  }

  return fixedMatrix;
}

// Function to convert a 3D matrix of fixed-point integers to a 3D matrix of floats.
std::vector<std::vector<std::vector<float>>> fixedToFloat(std::vector<std::vector<std::vector<int>>>& fixedMatrix) {
  std::vector<std::vector<std::vector<float>>> floatMatrix(fixedMatrix.size(),
    std::vector<std::vector<float>>(fixedMatrix[0].size(),
      std::vector<float>(fixedMatrix[0][0].size())));

  size_t depth = fixedMatrix.size();
  size_t height = fixedMatrix[0].size();
  size_t width = fixedMatrix[0][0].size();

  #pragma omp parallel for collapse(3)
  for (size_t i = 0; i < depth; ++i) {
    for (size_t j = 0; j < height; ++j) {
      for (size_t k = 0; k < width; ++k) {
        floatMatrix[i][j][k] = static_cast<float>(fixedMatrix[i][j][k]) / FIXED_POINT_SCALING_FACTOR;
      }
    }
  }

  return floatMatrix;
}

// Function to convert a 2D matrix of floats to a 2D matrix of fixed-point integers.
std::vector<std::vector<int>> floatToFixed(std::vector<std::vector<float>>& inputMatrix) {
  std::vector<std::vector<int>> fixedMatrix(inputMatrix.size(),
    std::vector<int>(inputMatrix[0].size()));

  size_t height = inputMatrix.size();
  size_t width = inputMatrix[0].size();

  #pragma omp parallel for collapse(2)
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      fixedMatrix[i][j] = static_cast<int>(inputMatrix[i][j] * FIXED_POINT_SCALING_FACTOR);
    }
  }

  return fixedMatrix;
}

// Function to convert a 2D matrix of fixed-point integers to a 2D matrix of floats.
std::vector<std::vector<float>> fixedToFloat(std::vector<std::vector<int>>& fixedMatrix) {
  std::vector<std::vector<float>> floatMatrix(fixedMatrix.size(),
    std::vector<float>(fixedMatrix[0].size()));

  size_t height = fixedMatrix.size();
  size_t width = fixedMatrix[0].size();

  #pragma omp parallel for collapse(2)
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      floatMatrix[i][j] = static_cast<float>(fixedMatrix[i][j]) / FIXED_POINT_SCALING_FACTOR;
    }
  }

  return floatMatrix;
}

// Function to convert a vector of floats to a vector of fixed-point integers.
std::vector<int> floatToFixed(std::vector<float>& inputVector) {
  std::vector<int> fixedVector(inputVector.size());

  #pragma omp parallel for
  for (size_t i = 0; i < inputVector.size(); ++i) {
    fixedVector[i] = static_cast<int>(inputVector[i] * FIXED_POINT_SCALING_FACTOR);
  }

  return fixedVector;
}

// Function to convert a vector of fixed-point integers to a vector of floats.
std::vector<float> fixedToFloat(std::vector<int>& fixedVector) {
  std::vector<float> floatVector(fixedVector.size());

  #pragma omp parallel for
  for (size_t i = 0; i < fixedVector.size(); ++i) {
    floatVector[i] = static_cast<float>(fixedVector[i]) / FIXED_POINT_SCALING_FACTOR;
  }

  return floatVector;
}

#endif