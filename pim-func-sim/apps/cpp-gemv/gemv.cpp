// Test: C++ version of vector add
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "libpimsim.h"
#include <iostream>
#include <vector>

int main()
{
  std::cout << "PIM test: Vector multiplication" << std::endl;

  unsigned numCores = 4;
  unsigned numRows = 128;
  unsigned numCols = 256;

  PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numCores, numRows, numCols);
  if (status != PIM_OK) {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  unsigned numMatRows = 1;
  unsigned numMatCols = 1024;
  unsigned bitsPerElement = 32;
  unsigned totalElementCount = numMatCols*numMatRows;

  PimObjId obj1 = pimAlloc(PIM_ALLOC_V1, totalElementCount, bitsPerElement);
  if (obj1 == -1) {
    std::cout << "Abort" << std::endl;
    return 1;
  }
  PimObjId obj2 = pimAllocAssociated(PIM_ALLOC_V1, totalElementCount, bitsPerElement, obj1);
  if (obj2 == -1) {
    std::cout << "Abort" << std::endl;
    return 1;
  }
  PimObjId obj3 = pimAllocAssociated(PIM_ALLOC_V1, totalElementCount, bitsPerElement, obj1);
  if (obj3 == -1) {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  std::vector<int> src1(totalElementCount);
  std::vector<int> src2(totalElementCount);
  int dest = 0;

  // assign some initial values
  for (unsigned i = 0; i < totalElementCount; ++i) {
    src1[i] = i;
  }


  for (unsigned i = 0; i < numMatCols; i++) {
    for (unsigned j = 0; j < numMatRows; j++) {
      src2[numMatCols*j + i] = i*2 - i;
    }
  }

  status = pimCopyHostToDevice(PIM_COPY_V, (void*)src1.data(), obj1);
  if (status != PIM_OK) {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  status = pimCopyHostToDevice(PIM_COPY_V, (void*)src2.data(), obj2);
  if (status != PIM_OK) {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  status = pimInt32Mul(obj1, obj2, obj3);
  if (status != PIM_OK) {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  dest = pimInt32RedSum(obj3);
  
  // check results
  bool ok = true;
  int result = 0;
  for (unsigned i = 0; i < numMatRows; i++) {
    for (unsigned j = 0; j < numMatCols; j++) {
      result += src1[i*numMatCols + j] * src2[j];
    }
    if (dest != result) {
      std::cout << "Wrong answer: " << dest << " (expected " << result << ")" << std::endl;
      ok = false;
    }
  }

  if (ok) {
    std::cout << "All correct!" << std::endl;
  }

  return 0;
}

