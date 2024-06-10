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
  unsigned bitsPerElement = 32;

  std::vector<int> src1(numElements);
  std::vector<int> src2(numElements);
  std::vector<int> dest(numElements);

  // Assign some initial values
  for (unsigned i = 0; i < numElements; ++i) {
    src1[i] = i;
    src2[i] = i * 2 - 9;
  }

  PimObjId obj1 = pimAlloc(PIM_ALLOC_AUTO, numElements, bitsPerElement, PIM_INT32);
  assert(obj1 != -1);
  PimObjId obj2 = pimAllocAssociated(bitsPerElement, obj1, PIM_INT32);
  assert(obj2 != -1);
  PimObjId obj3 = pimAllocAssociated(bitsPerElement, obj1, PIM_INT32);
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
      assert(dest[i] == src1[i] + src2[i]);
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
      assert(dest[i] == src1[i] - src2[i]);
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
      assert(dest[i] == src1[i] * src2[i]);
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
      assert(dest[i] == src1[i] / src2[i]);
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
      assert(dest[i] == std::abs(src2[i]));
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
      assert(dest[i] == (src1[i] & src2[i]));
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
      assert(dest[i] == (src1[i] | src2[i]));
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
      assert(dest[i] == (src1[i] ^ src2[i]));
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
      assert(dest[i] == ~(src1[i] ^ src2[i]));
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
      int cnt = std::bitset<32>(src1[i]).count();
      assert(dest[i] == cnt);
    }
    std::cout << "[PASSED] pimPopCount" << std::endl;
  }

  // Test redsum
  {
    int sum = 0;
    status = pimRedSum(obj1, &sum);
    assert(status == PIM_OK);
    int sum2 = 0;
    for (unsigned i = 0; i < numElements; ++i) {
      sum2 += src1[i];
    }
    assert(sum == sum2);
    std::cout << "[PASSED] pimRedSum" << std::endl;
  }

  // Test redsum ranged
  {
    unsigned idxBegin = 100;
    unsigned idxEnd = 500;
    int sum = 0;
    status = pimRedSumRanged(obj1, idxBegin, idxEnd, &sum);
    assert(status == PIM_OK);
    int sum2 = 0;
    for (unsigned i = idxBegin; i < idxEnd; ++i) {
      sum2 += src1[i];
    }
    assert(sum == sum2);
    std::cout << "[PASSED] pimRedSumRanged" << std::endl;
  }

  // Test broadcast
  {
    status = pimBroadcast(obj3, 123456);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == 123456);
    }
    std::cout << "[PASSED] pimBroadcast" << std::endl;
  }

  // Test rotate R
  {
    status = pimCopyHostToDevice((void *)src1.data(), obj3);
    assert(status == PIM_OK);
    status = pimRotateR(obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements - 1; ++i) {
      assert(dest[i + 1] == src1[i]);
    }
    assert(dest.front() == src1.back());
    std::cout << "[PASSED] pimRotateR" << std::endl;
  }

  // Test rotate L
  {
    status = pimCopyHostToDevice((void *)src1.data(), obj3);
    assert(status == PIM_OK);
    status = pimRotateL(obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements - 1; ++i) {
      assert(dest[i] == src1[i + 1]);
    }
    assert(dest.back() == src1.front());
    std::cout << "[PASSED] pimRotateL" << std::endl;
  }

  // Test shift R
  {
    status = pimCopyHostToDevice((void *)src1.data(), obj3);
    assert(status == PIM_OK);
    status = pimShiftR(obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements - 1; ++i) {
      assert(dest[i + 1] == src1[i]);
    }
    assert(dest.front() == 0);
    std::cout << "[PASSED] pimShiftR" << std::endl;
  }

  // Test shift L
  {
    status = pimCopyHostToDevice((void *)src1.data(), obj3);
    assert(status == PIM_OK);
    status = pimShiftL(obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements - 1; ++i) {
      assert(dest[i] == src1[i + 1]);
    }
    assert(dest.back() == 0);
    std::cout << "[PASSED] pimShiftL" << std::endl;
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

