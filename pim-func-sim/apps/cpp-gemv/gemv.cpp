// Test: C++ version of matrix vector multiplication
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "libpimsim.h"
#include <iostream>
#include <vector>

int main()
{
  std::cout << "PIM test: Matrix vector multiplication" << std::endl;

  unsigned numCores = 4;
  unsigned numRows = 512;
  unsigned numCols = 512;

  PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numCores, numRows, numCols);
  if (status != PIM_OK) {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  unsigned numMatRows = 64;
  unsigned numMatCols = 8;
  unsigned bitsPerElement = 32;
  unsigned totalElementCount = numMatCols*numMatRows;

  PimObjId obj1 = pimAlloc(PIM_ALLOC_V1, totalElementCount, bitsPerElement, PIM_INT32);
  if (obj1 == -1) {
    std::cout << "Abort" << std::endl;
    return 1;
  }
  PimObjId obj2 = pimAllocAssociated(PIM_ALLOC_V1, totalElementCount, bitsPerElement, obj1, PIM_INT32);
  if (obj2 == -1) {
    std::cout << "Abort" << std::endl;
    return 1;
  }
  PimObjId obj3 = pimAllocAssociated(PIM_ALLOC_V1, totalElementCount, bitsPerElement, obj1, PIM_INT32);
  if (obj3 == -1) {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  std::vector<int> src1(totalElementCount);
  std::vector<int> src2(totalElementCount);
  std::vector<int> dest(numMatRows);

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

  status = pimMul(obj1, obj2, obj3);
  if (status != PIM_OK) {
    std::cout << "Abort" << std::endl;
    return 1;
  }

  for (unsigned i = 0; i < numMatRows; i++) {
    int stIDX = numMatCols*i;
    dest[i] = pimRedSumRanged(obj3, stIDX, stIDX+numMatCols );
  }
  
  // check results
  bool ok = true;
  int result = 0;
  for (unsigned i = 0; i < numMatRows; i++) {
    for (unsigned j = 0; j < numMatCols; j++) {
      result += src1[i*numMatCols + j] * src2[j];
    }
    if (dest[i] != result) {
      std::cout << "IDX: " << i << " Wrong answer: " << dest[i] << " (expected " << result << ")" << std::endl;
      ok = false;
      break;
    }
    result = 0;
  }

  pimShowStats();
  if (ok) {
    std::cout << "All correct!" << std::endl;
  }

  return 0;
}

