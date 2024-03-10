// Test: C++ version of vector add
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "libpimsim.h"
#include <cstdio>
#include <vector>

int main()
{
  std::printf("PIM test: Vector add\n");

  unsigned numCores = 4;
  unsigned numRows = 128;
  unsigned numCols = 256;

  PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, numCores, numRows, numCols);
  if (status != PIM_OK) {
    std::printf("Abort\n");
    return 1;
  }

  unsigned numElements = 512;
  unsigned bitsPerElement = 32;

  PimObjId obj1 = pimAlloc(PIM_ALLOC_V1, numElements, bitsPerElement);
  if (obj1 == -1) {
    std::printf("Abort\n");
    return 1;
  }
  PimObjId obj2 = pimAllocAssociated(PIM_ALLOC_V1, numElements, bitsPerElement, obj1);
  if (obj2 == -1) {
    std::printf("Abort\n");
    return 1;
  }
  
  std::vector<int> src1(numElements);
  std::vector<int> src2(numElements);
  std::vector<int> dest(numElements);

  // assign some initial values
  for (int i = 0; i < numElements; ++i) {
    src1[i] = i;
    src2[i] = i * 2 - 10;
  }

  status = pimCopyHostToDevice(PIM_COPY_V, (void*)src1.data(), obj1);
  if (status != PIM_OK) {
    std::printf("Abort\n");
    return 1;
  }

  status = pimCopyHostToDevice(PIM_COPY_V, (void*)src2.data(), obj2);
  if (status != PIM_OK) {
    std::printf("Abort\n");
    return 1;
  }

  return 0;
}
