// Test: SIMDRAM micro-ops
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include <iostream>
#include <vector>
#include <cassert>


bool createPimDevice()
{
  unsigned numSubarrayPerBank = 8;
  unsigned numRows = 128;
  unsigned numCols = 256;

  PimStatus status = pimCreateDevice(PIM_DEVICE_SIMDRAM, 1, 1, numSubarrayPerBank, numRows, numCols);
  assert(status == PIM_OK);
  return true;
}

bool testMicroOps()
{
  unsigned numElements = 1000;

  std::vector<int> src1(numElements);
  std::vector<int> src2(numElements);
  std::vector<int> dest1(numElements);
  std::vector<int> dest2(numElements);
  for (unsigned i = 0; i < numElements; ++i) {
    src1[i] = static_cast<int>(i);
    src2[i] = static_cast<int>(i * 3 + 1);
  }

  PimObjId obj1 = pimAlloc(PIM_ALLOC_V1, numElements, PIM_INT32);
  assert(obj1 != -1);
  PimObjId obj2 = pimAllocAssociated(obj1, PIM_INT32);
  assert(obj2 != -1);
  PimObjId obj3 = pimCreateDualContactRef(obj2);
  assert(obj3 != -1);

  // Test AP
  PimStatus status = pimCopyHostToDevice((void*)src1.data(), obj1);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void*)src2.data(), obj2);
  assert(status == PIM_OK);
  status = pimOpAP(3, obj1, 0, obj1, 1, obj2, 1);
  assert(status == PIM_OK);
  status = pimCopyDeviceToHost(obj1, (void*)dest1.data());
  assert(status == PIM_OK);
  status = pimCopyDeviceToHost(obj2, (void*)dest2.data());
  assert(status == PIM_OK);
  for (unsigned i = 0; i < numElements; ++i) {
    bool val1 = (src1[i] & 1);
    bool val2 = (src1[i] & 2);
    bool val3 = (src2[i] & 2);
    bool maj = ((int)val1 + val2 + val3) > 1;
    bool val1a = (dest1[i] & 1);
    bool val2a = (dest1[i] & 2);
    bool val3a = (dest2[i] & 2);
    if (val1a != maj || val2a != maj || val3a != maj) {
      std::cout << "Row AP failed" << std::endl;
      return false;
    }
    // check remaining bits
    int val1i = src1[i];
    val1i = (val1i & ~3) | (maj) | ((int)maj << 1);
    int val2i = src2[i];
    val2i = (val2i & ~2) | ((int)maj << 1);
    if (val1i != dest1[i] || val2i != dest2[i]) {
      std::cout << "Row AP failed - incorrect mem contents" << std::endl;
      return false;
    }
  }
  std::cout << "Row AP OK." << std::endl;

  // Test AAP
  status = pimCopyHostToDevice((void*)src1.data(), obj1);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void*)src2.data(), obj2);
  assert(status == PIM_OK);
  status = pimOpAAP(3, 3, obj1, 0, obj1, 1, obj2, 1, obj1, 4, obj2, 4, obj2, 5);
  assert(status == PIM_OK);
  status = pimCopyDeviceToHost(obj1, (void*)dest1.data());
  assert(status == PIM_OK);
  status = pimCopyDeviceToHost(obj2, (void*)dest2.data());
  assert(status == PIM_OK);
  for (unsigned i = 0; i < numElements; ++i) {
    bool val1 = (src1[i] & 1);
    bool val2 = (src1[i] & 2);
    bool val3 = (src2[i] & 2);
    bool maj = ((int)val1 + val2 + val3) > 1;
    bool val1a = (dest1[i] & 1);
    bool val2a = (dest1[i] & 2);
    bool val3a = (dest2[i] & 2);
    bool val1b = (dest1[i] & 16);
    bool val2b = (dest2[i] & 16);
    bool val3b = (dest2[i] & 32);
    if (val1a != maj || val2a != maj || val3a != maj || val1b != maj || val2b != maj || val3b != maj) {
      std::cout << "Row AAP failed" << std::endl;
      return false;
    }
    // check remaining bits
    int val1i = src1[i];
    for (int ofst : {0, 1, 4}) {
      val1i = ((val1i & ~(1 << ofst)) | ((int)maj << ofst));
    }
    int val2i = src2[i];
    for (int ofst : {1, 4, 5}) {
      val2i = ((val2i & ~(1 << ofst)) | ((int)maj << ofst));
    }
    if (val1i != dest1[i] || val2i != dest2[i]) {
      std::cout << "Row AAP failed - incorrect mem contents\n" << std::endl;
      return false;
    }
  }
  std::cout << "Row AAP OK." << std::endl;

  // Test Dual-Contact Cells
  status = pimCopyHostToDevice((void*)src1.data(), obj1);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void*)src2.data(), obj3);
  assert(status == PIM_OK);
  status = pimCopyDeviceToHost(obj2, (void*)dest1.data());
  assert(status == PIM_OK);
  status = pimCopyDeviceToHost(obj3, (void*)dest2.data());
  assert(status == PIM_OK);
  for (unsigned i = 0; i < numElements; ++i) {
    if (src2[i] != ~dest1[i] || src2[i] != dest2[i]) {
      std::cout << "Row DCC failed - expected negated copy\n" << std::endl;
      return false;
    }
  }
  status = pimOpAAP(1, 2, obj1, 0, obj2, 0, obj3, 1);
  assert(status == PIM_OK);
  status = pimCopyDeviceToHost(obj2, (void*)dest1.data());
  assert(status == PIM_OK);
  status = pimCopyDeviceToHost(obj3, (void*)dest2.data());
  assert(status == PIM_OK);
  for (unsigned i = 0; i < numElements; ++i) {
    bool val1 = src1[i] & 1;
    bool val2 = dest1[i] & 1;
    bool val3 = dest1[i] & 2;
    bool val2n = dest2[i] & 1;
    bool val3n = dest2[i] & 2;
    if (val2 != !val2n || val3 != !val3n) {
      std::cout << "Row DCC failed - expected negated copy 2\n" << std::endl;
      return false;
    }
    if (val1 != val2 || val1 != !val3) {
      std::cout << "Row DCC failed - incorrect AAP results with DCCN\n" << std::endl;
      return false;
    }
  }

  return true;
}

int main()
{
  std::cout << "PIM test: SIMDRAM micro-ops" << std::endl;

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

  pimShowStats();
  std::cout << "All correct!" << std::endl;
  return 0;
}

