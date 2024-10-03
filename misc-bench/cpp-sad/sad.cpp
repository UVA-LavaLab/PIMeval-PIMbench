// Test: C++ version of SAD
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include <iostream>
#include <vector>

#define INT_MAX __INT_MAX__

int main()
{
  std::cout << "PIM test: SAD" << std::endl;

  unsigned numCores = 4;
  unsigned numRows = 1024;
  unsigned numCols = 8192;

  PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, 1, 1, numCores, numRows, numCols);
  if (status != PIM_OK) {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  int vectorLength = 512;
  int subvectorLength = 64;

  PimObjId obj1 = pimAlloc(PIM_ALLOC_AUTO, vectorLength, PIM_INT32);
  if (obj1 == -1) {
    std::cout << "Abort" << std::endl;
    return 1;
  }
  PimObjId obj2 = pimAllocAssociated(obj1, PIM_INT32);
  if (obj2 == -1) {
    std::cout << "Abort" << std::endl;
    return 1;
  }
  PimObjId obj3 = pimAllocAssociated(obj1, PIM_INT32);
  if (obj3 == -1) {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  std::vector<unsigned> src1(vectorLength);
  std::vector<unsigned> src2(subvectorLength);
  std::vector<unsigned> replicateSrc2(vectorLength);
  int correct_idx = rand() % (vectorLength - subvectorLength);

  // assign some initial values
  for (int i = 0; i < vectorLength; ++i) {
    src1[i] = i;
  }

  for (int i = 0; i < subvectorLength; ++i) {
    src2[i] = src1[i + correct_idx];
  }

  for (int i = 0; i < vectorLength; i += subvectorLength) {
    for (int j = 0; j < subvectorLength; j++) {
      replicateSrc2[i+j] = src2[j];
    }
  }

  status = pimCopyHostToDevice((void*)src1.data(), obj1);
  if (status != PIM_OK) {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  status = pimCopyHostToDevice((void*)replicateSrc2.data(), obj2);
  if (status != PIM_OK) {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  int min_diff = INT_MAX;
  int min_idx = -1;
  int64_t sum_abs_diff;

  for (int idx = 0; idx < subvectorLength; idx++) {
    status = pimSub(obj1, obj2, obj3);
    if (status != PIM_OK) {
      std::cout << "Abort" << std::endl;
      return 1;
    }

    status = pimAbs(obj3, obj3);
    if (status != PIM_OK) {
      std::cout << "Abort" << std::endl;
      return 1;
    }

    for (int i = idx; i + subvectorLength - 1 < vectorLength; i += subvectorLength) {
      status = pimRedSumRangedInt(obj3, i, i + subvectorLength, &sum_abs_diff);
      if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        return 1;
      }
      // Update minimum
      // TODO: put the reduction sum ranged value in a vector object and calculate the minimum in PIM. Currently it executes the comparison on CPU
      if (sum_abs_diff < min_diff) {
        min_idx = i;
        min_diff = sum_abs_diff;
      }
    }

    status = pimRotateElementsRight(obj2);
    if (status != PIM_OK) {
      std::cout << "Abort" << std::endl;
      return 1;
    }
  }

  if (min_idx == correct_idx) {
    printf("Best match found!\n");
  } else {
    std::cout << "Failed to find best match" << std::endl;
    std::cout << "Result index is " << min_idx << ", expected to be " << correct_idx << std::endl;
  }

  pimShowStats();

  return 0;
}

