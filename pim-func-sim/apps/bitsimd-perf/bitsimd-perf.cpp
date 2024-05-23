// Test: BitSIMD-V basic tests
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "libpimsim.h"
#include <iostream>
#include <vector>
#include <cassert>


bool createPimDevice()
{
  unsigned numCores = 4;
  unsigned numRows = 1024;
  unsigned numCols = 256;

  PimStatus status = pimCreateDevice(PIM_DEVICE_BITSIMD_V, numCores, numRows, numCols);
  assert(status == PIM_OK);
  return true;
}

bool testMicroOps()
{
  unsigned numElements = 3000;
  unsigned bitsPerElement = 32;

  std::vector<int> src1(numElements);
  std::vector<int> src2(numElements);
  std::vector<int> dest(numElements);

  PimObjId obj1 = pimAlloc(PIM_ALLOC_V1, numElements, bitsPerElement, PIM_INT32);
  assert(obj1 != -1);
  PimObjId obj2 = pimAllocAssociated(PIM_ALLOC_V1, numElements, bitsPerElement, obj1, PIM_INT32);
  assert(obj2 != -1);
  PimObjId obj3 = pimAllocAssociated(PIM_ALLOC_V1, numElements, bitsPerElement, obj1, PIM_INT32);
  assert(obj3 != -1);

  // assign some initial values
  for (unsigned i = 0; i < numElements; ++i) {
    src1[i] = i;
    src2[i] = i * 2 - 10;
  }

  PimStatus status = PIM_OK;
  status = pimCopyHostToDevice(PIM_COPY_V, (void*)src1.data(), obj1);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice(PIM_COPY_V, (void*)src2.data(), obj2);
  assert(status == PIM_OK);

  status = pimAdd(obj1, obj2, obj3);
  assert(status == PIM_OK);

  status = pimCopyDeviceToHost(PIM_COPY_V, obj3, (void*)dest.data());
  assert(status == PIM_OK);

  for (unsigned i = 0; i < numElements; ++i) {
    if (dest[i] != src1[i] + src2[i]) {
      return false;
    }
  }
  return true;
}

int main()
{
  std::cout << "PIM test: BitSIMD-V basic" << std::endl;

  bool ok = createPimDevice();
  if (!ok) {
    std::cout << "PIM device creation failed!" << std::endl;
    return 1;
  }

  ok = testMicroOps();
  if (!ok) {
    std::cout << "Test failed!" << std::endl;
    return 1;
  }
  std::cout << "All correct!" << std::endl;

  pimShowStats();
  return 0;
}

