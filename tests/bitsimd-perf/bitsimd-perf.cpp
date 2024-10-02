// Test: BitSIMD-V basic tests
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include <iostream>
#include <vector>
#include <cassert>


bool createPimDevice()
{
  unsigned numCores = 4;
  unsigned numRows = 1024;
  unsigned numCols = 256;

  PimStatus status = pimCreateDevice(PIM_DEVICE_BITSIMD_V, 1, 1, numCores, numRows, numCols);
  assert(status == PIM_OK);
  return true;
}

bool testMicroOps()
{
  unsigned numElements = 3000;

  std::vector<int> src1(numElements);
  std::vector<int> src2(numElements);
  std::vector<int> dest(numElements);

  PimObjId obj1 = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_INT32);
  assert(obj1 != -1);
  PimObjId obj2 = pimAllocAssociated(obj1, PIM_INT32);
  assert(obj2 != -1);
  PimObjId obj3 = pimAllocAssociated(obj1, PIM_INT32);
  assert(obj3 != -1);

  // assign some initial values
  for (unsigned i = 0; i < numElements; ++i) {
    src1[i] = i;
    src2[i] = i * 2 - 10;
  }

  PimStatus status = PIM_OK;
  status = pimCopyHostToDevice((void*)src1.data(), obj1);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void*)src2.data(), obj2);
  assert(status == PIM_OK);

  status = pimAdd(obj1, obj2, obj3);
  assert(status == PIM_OK);

  status = pimCopyDeviceToHost(obj3, (void*)dest.data());
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

