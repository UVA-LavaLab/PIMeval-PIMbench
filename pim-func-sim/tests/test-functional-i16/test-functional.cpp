// Test: Test functional behavior
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "libpimsim.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstdlib>


void createPimDevice()
{
  unsigned numCores = 4;
  unsigned numRows = 1024;
  unsigned numCols = 256;

  PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, 1, 1, numCores, numRows, numCols);
  assert(status == PIM_OK);
}

void testFunctional()
{
  unsigned numElements = 3000;
  unsigned bitsPerElement = 16;

  std::vector<int16_t> src1(numElements);
  std::vector<int16_t> src2(numElements);
  std::vector<int16_t> dest(numElements);

  // Assign some initial values
  for (unsigned i = 0; i < numElements; ++i) {
    src1[i] = static_cast<int16_t>(i);
    src2[i] = static_cast<int16_t>(i*2 + 9);
  }

  PimObjId obj1 = pimAlloc(PIM_ALLOC_AUTO, numElements, bitsPerElement, PIM_INT16);
  assert(obj1 != -1);
  PimObjId obj2 = pimAllocAssociated(bitsPerElement, obj1, PIM_INT16);
  assert(obj2 != -1);
  PimObjId obj3 = pimAllocAssociated(bitsPerElement, obj1, PIM_INT16);
  assert(obj3 != -1);

  PimStatus status = PIM_OK;
  status = pimCopyHostToDevice((void*)src1.data(), obj1);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void*)src2.data(), obj2);
  assert(status == PIM_OK);

  // Test add
  {
    status = pimAdd(obj1, obj2, obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == static_cast<int16_t>(src1[i] + src2[i]));
    }
    std::cout << "[PASSED] pimAdd" << std::endl;
  }

  // Test sub
  {
    status = pimSub(obj1, obj2, obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == static_cast<int16_t>(src1[i] - src2[i]));
    }
    std::cout << "[PASSED] pimSub" << std::endl;
  }

  // Test mul
  {
    status = pimMul(obj1, obj2, obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == static_cast<int16_t>(src1[i] * src2[i]));
    }
    std::cout << "[PASSED] pimMul" << std::endl;
  }

  // Test div
  {
    status = pimDiv(obj1, obj2, obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == static_cast<int16_t>(src1[i] / src2[i]));
    }
    std::cout << "[PASSED] pimDiv" << std::endl;
  }

  // Test abs
  {
    status = pimAbs(obj2, obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == static_cast<int16_t>(std::abs(src2[i])));
    }
    std::cout << "[PASSED] pimAbs" << std::endl;
  }

  // Test and
  {
    status = pimAnd(obj1, obj2, obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == static_cast<int16_t>((src1[i] & src2[i])));
    }
    std::cout << "[PASSED] pimAnd" << std::endl;
  }

  // Test or
  {
    status = pimOr(obj1, obj2, obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == static_cast<int16_t>(src1[i] | src2[i]));
    }
    std::cout << "[PASSED] pimOr" << std::endl;
  }

  // Test xor
  {
    status = pimXor(obj1, obj2, obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == static_cast<int16_t>(src1[i] ^ src2[i]));
    }
    std::cout << "[PASSED] pimXor" << std::endl;
  }

  // Test xnor
  {
    status = pimXnor(obj1, obj2, obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == static_cast<int16_t>(~(src1[i] ^ src2[i])));
    }
    std::cout << "[PASSED] pimXnor" << std::endl;
  }

  // Test GT
  {
    status = pimGT(obj1, obj2, obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == (src1[i] > src2[i] ? 1 : 0));
    }
    std::cout << "[PASSED] pimGT" << std::endl;
  }

  // Test LT
  {
    status = pimLT(obj1, obj2, obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == (src1[i] < src2[i] ? 1 : 0));
    }
    std::cout << "[PASSED] pimLT" << std::endl;
  }

  // Test EQ
  {
    status = pimEQ(obj1, obj2, obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == (src1[i] == src2[i] ? 1 : 0));
    }
    std::cout << "[PASSED] pimEQ" << std::endl;
  }

  // Test min
  {
    status = pimMin(obj1, obj2, obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == std::min(src1[i], src2[i]));
    }
    std::cout << "[PASSED] pimMin" << std::endl;
  }

  // Test max
  {
    status = pimMax(obj1, obj2, obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == std::max(src1[i], src2[i]));
    }
    std::cout << "[PASSED] pimMax" << std::endl;
  }

  // Test popcount
  {
    status = pimPopCount(obj1, obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      int16_t cnt = std::bitset<16>(src1[i]).count();
      assert(dest[i] == cnt);
    }
    std::cout << "[PASSED] pimPopCount" << std::endl;
  }

  // Test redsum
  {
    int64_t sumDevice = 0;
    status = pimRedSumSignedInt(obj1, &sumDevice);
    assert(status == PIM_OK);
    int64_t sumHost = 0;
    for (unsigned i = 0; i < numElements; ++i) {
      sumHost += src1[i];
    }
    assert(sumDevice == sumHost);
    std::cout << "[PASSED] pimRedSum" << std::endl;
  }

  // Test redsum ranged
  {
    unsigned idxBegin = 100;
    unsigned idxEnd = 500;
    int64_t sumDevice = 0;
    status = pimRedSumRanged(obj1, idxBegin, idxEnd, &sumDevice);
    assert(status == PIM_OK);
    int64_t sumHost = 0;
    for (unsigned i = idxBegin; i < idxEnd; ++i) {
      sumHost += src1[i];
    }
    assert(sumDevice == sumHost);
    std::cout << "[PASSED] pimRedSumRanged" << std::endl;
  }

  // Test broadcast
  {
    status = pimBroadcast(obj3, 12);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == 12);
    }
    std::cout << "[PASSED] pimBroadcast" << std::endl;
  }

  // Test rotate R
  {
    status = pimCopyHostToDevice((void *)src1.data(), obj3);
    assert(status == PIM_OK);
    status = pimRotateElementsRight(obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements - 1; ++i) {
      assert(dest[i + 1] == src1[i]);
    }
    assert(dest.front() == src1.back());
    std::cout << "[PASSED] pimRotateElementsRight" << std::endl;
  }

  // Test rotate L
  {
    status = pimCopyHostToDevice((void *)src1.data(), obj3);
    assert(status == PIM_OK);
    status = pimRotateElementsLeft(obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements - 1; ++i) {
      assert(dest[i] == src1[i + 1]);
    }
    assert(dest.back() == src1.front());
    std::cout << "[PASSED] pimRotateElementsLeft" << std::endl;
  }

  // Test shift elements R
  {
    status = pimCopyHostToDevice((void *)src1.data(), obj3);
    assert(status == PIM_OK);
    status = pimShiftElementsRight(obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements - 1; ++i) {
      assert(dest[i + 1] == src1[i]);
    }
    assert(dest.front() == 0);
    std::cout << "[PASSED] pimShiftElementsRight" << std::endl;
  }

  // Test shift elements L
  {
    status = pimCopyHostToDevice((void *)src1.data(), obj3);
    assert(status == PIM_OK);
    status = pimShiftElementsLeft(obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements - 1; ++i) {
      assert(dest[i] == src1[i + 1]);
    }
    assert(dest.back() == 0);
    std::cout << "[PASSED] pimShiftElementsLeft" << std::endl;
  }

  // Test shift bits R
  {
    status = pimCopyHostToDevice((void *)src1.data(), obj1);
    assert(status == PIM_OK);
    status = pimShiftBitsRight(obj1, obj3, 4);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == static_cast<int16_t>(src1[i] >> 4));
    }
    std::cout << "[PASSED] pimShiftBitsRight" << std::endl;
  }

  // Test shift bits L
  {
    status = pimCopyHostToDevice((void *)src1.data(), obj1);
    assert(status == PIM_OK);
    status = pimShiftBitsLeft(obj1, obj3, 2);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == static_cast<int16_t>(src1[i] << 2));
    }
    std::cout << "[PASSED] pimShiftBitsLeft" << std::endl;
  }
}

int main()
{
  std::cout << "PIM Regression Test: Functional" << std::endl;

  createPimDevice();

  testFunctional();

  pimShowStats();

  return 0;
}

