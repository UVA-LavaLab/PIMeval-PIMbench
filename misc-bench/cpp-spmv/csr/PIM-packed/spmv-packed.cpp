// Test: C++ version of vector multiplication
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "../spmv-util.h"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <stdint.h>
#include <string>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "../../../../libpimeval/src/libpimeval.h"
#include "../../../../util/util.h"

using namespace std;

void pimCSRSpmv(const std::vector<float> &values, size_t valuesLength,
                const std::vector<float> &x, size_t xLength,
                std::vector<u64> rowPointers, size_t rowPointersLength,
                std::vector<u64> colPointers, size_t colPointersLength,
                std::vector<float> *dst) {
  /*u64 blocks_per_row =*/
  /*    static_cast<u64>(std::ceil((1.0 * x.size()) / (BLOCK_LEN)));*/

  PimObjId valuesDevice = pimAlloc(PIM_ALLOC_AUTO, valuesLength, PIM_FP32);

  PimObjId xDevice = pimAllocAssociated(valuesDevice, PIM_FP32);

  PimStatus status;

  status = pimCopyHostToDevice((void *)values.data(), valuesDevice);
  assert(status == PIM_OK);

  std::vector<float> xRearranged(valuesLength);

  for (u64 i = 0; i < colPointersLength; i++)
    xRearranged[i] = x[colPointers[i]];

  status = pimCopyHostToDevice((void *)xRearranged.data(), xDevice);
  assert(status == PIM_OK);

  status = pimMul(xDevice, valuesDevice, valuesDevice);
  assert(status == PIM_OK);

  dst->resize(rowPointersLength - 1);
  float temp;
  for (u64 i = 0; i < rowPointersLength - 1; i++) {
    temp = 0;
    if (rowPointers[i + 1] - rowPointers[i] > 0) {
      status = pimRedSum(valuesDevice, (void *)&temp, rowPointers[i],
                         rowPointers[i + 1]);
      (*dst)[i] = (float)temp;
      assert(status == PIM_OK);
    }
  }
  pimFree(valuesDevice);
  pimFree(xDevice);
}

int main(int argc, char *argv[]) {
  struct Params params = getInputParams(argc, argv);
  // std::cout << "Vector length: " << params.vectorLength << "\n";
  std::vector<float> x, values, Y, dst;
  std::vector<u64> matDim, rowPointers, colPointers;
  // std::vector<int> src1, src2, dst;
  if (params.inputFile == nullptr) {
    // based on input generation type, generate a matrix
    fprintf(stderr, "Generating matrix and vector\n");
    if (params.generationType == GEN_RANDOM_BIASED) {
      generateSpMV(&params, &x, &values, &rowPointers, &colPointers);
    } else if (params.generationType == GEN_UNIFORM) {
      generateUniformSpMV(&params, &x, &values, &rowPointers, &colPointers);
    } else if (params.generationType == GEN_TRIDIAGONAL) {
      generateTriDiagonal(&params, &x, &values, &rowPointers, &colPointers);
    } else if (params.generationType == GEN_LEFT_ALIGNED) {
      generateRandomNonzero(&params, &x, &values, &rowPointers, &colPointers);
    }
    fprintf(stderr, "Matrix and vector generated\n");
    std::cout << "Nonzeros: " << params.nnz << std::endl;
    if (params.shouldVerify) {
      fprintf(stderr, "Calculating result on the CPU\n");
      cpuCSRSpmv(values, values.size(), x, x.size(), rowPointers,
                 rowPointers.size(), colPointers, colPointers.size(), Y);
      fprintf(stderr, "Result calculated\n");
    }
  } else {
    readMatrixFromFile(params.inputFile, &rowPointers, &colPointers, &values, &x, params.seed);
  }
  if (params.configFile == nullptr) {
    if (!pimCreateDevice(params.deviceType, 32, 128, 32, 1024, 8192)) {
      std::cout << "Failed to create device from default values." << std::endl;
      return 1;
    }
  } else if (!pimCreateDeviceFromConfig(params.deviceType, params.configFile)) {
    std::cout << "Failed to create device from config." << std::endl;
    return 1;
  }

  // get device properties
  PimDeviceProperties deviceProp;
  PimStatus status = pimGetDeviceProperties(&deviceProp);
  if (status != PIM_OK) {
    std::cout << "Abort: pimGetDeviceProperties failed" << std::endl;
    return 1;
  }

  std::cout << "Starting PIM calculation" << std::endl;
  dst.resize(rowPointers.size() - 1);
  pimCSRSpmv(values, values.size(), x, x.size(), rowPointers,
             rowPointers.size(), colPointers, colPointers.size(), &dst);

  if (params.shouldVerify) {
    std::cout << "Verifying results of computation" << std::endl;
    bool ret_false = false;
    for (u64 i = 0; i < dst.size(); i++) {
      if (abs(dst[i] - Y[i]) > (dst[i] / 10000.0f)) {
        ret_false = true;
        std::cout << "Index " << i
                  << " of returned vector does not match test case's output."
                  << std::endl;
        std::cout << "Returned output: " << dst[i] << std::endl;
        std::cout << "Intended output: " << Y[i] << std::endl;
      }
    }
    if (ret_false) {
      return 1;
    }
  }

  pimShowStats();

  return 0;
}
