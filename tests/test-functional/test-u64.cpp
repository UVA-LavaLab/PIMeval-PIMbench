// Test: Test functional behavior
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "test-functional.h"
#include "libpimeval.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstdlib>


void
testFunctional::testU64()
{
  pimResetStats();

  unsigned numElements = 3000;
  unsigned bitsPerElement = 64;

  std::vector<uint64_t> src1(numElements);
  std::vector<uint64_t> src2(numElements);
  std::vector<uint64_t> dest(numElements);

  // Assign some initial values
  for (unsigned i = 0; i < numElements; ++i) {
    src1[i] = i;
    src2[i] = i * 2 + 2;
  }

  PimObjId obj1 = pimAlloc(PIM_ALLOC_AUTO, numElements, bitsPerElement, PIM_UINT64);
  assert(obj1 != -1);
  PimObjId obj2 = pimAllocAssociated(bitsPerElement, obj1, PIM_UINT64);
  assert(obj2 != -1);
  PimObjId obj3 = pimAllocAssociated(bitsPerElement, obj1, PIM_UINT64);
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

  // Test mul-aggregate
  {
    status = pimScaledAdd(obj1, obj2, obj3, 2);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == ((src1[i] * 2) + src2[i]));
    }
    std::cout << "[PASSED] pimMulAggregate" << std::endl;
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

    // Test add scaler
  {
    status = pimAddScalar(obj1, obj3, 9);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == static_cast<uint64_t>(src1[i] + 9));
    }
    std::cout << "[PASSED] pimAddScaler" << std::endl;
  }

  // Test sub scaler
  {
    status = pimSubScalar(obj1, obj3, 126);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == static_cast<uint64_t>(src1[i] - 126));
    }
    std::cout << "[PASSED] pimSubScaler" << std::endl;
  }

  // Test mul scaler
  {
    status = pimMulScalar(obj1, obj3, 99);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == static_cast<uint64_t>(src1[i] * 99));
    }
    std::cout << "[PASSED] pimMulScaler" << std::endl;
  }

  // Test div scaler
  {
    status = pimDivScalar(obj1, obj3, 23);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == static_cast<uint64_t>(src1[i] / 23));
    }
    std::cout << "[PASSED] pimDivScaler" << std::endl;
  }

  // Test and scaler
  {
    status = pimAndScalar(obj1, obj3, 35);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == static_cast<uint64_t>((src1[i] & 35)));
    }
    std::cout << "[PASSED] pimAndScaler" << std::endl;
  }

  // Test or scaler
  {
    status = pimOrScalar(obj1, obj3, 100);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == static_cast<uint64_t>(src1[i] | 100));
    }
    std::cout << "[PASSED] pimOrScaler" << std::endl;
  }

  // Test xor scaler
  {
    status = pimXorScalar(obj1, obj3, 60);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == static_cast<uint64_t>(src1[i] ^ 60));
    }
    std::cout << "[PASSED] pimXorScaler" << std::endl;
  }

  // Test xnor scaler
  {
    status = pimXnorScalar(obj1, obj3, 55);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == static_cast<uint64_t>(~(src1[i] ^ 55)));
    }
    std::cout << "[PASSED] pimXnorScaler" << std::endl;
  }

  // Test GT scaler
  {
    status = pimGTScalar(obj1, obj3, 19);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == (src1[i] > 19 ? 1 : 0));
    }
    std::cout << "[PASSED] pimGTScaler" << std::endl;
  }

  // Test LT scaler
  {
    status = pimLTScalar(obj1, obj3, 23);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == (src1[i] < 23 ? 1 : 0));
    }
    std::cout << "[PASSED] pimLTScaler" << std::endl;
  }

  // Test EQ scaler
  {
    status = pimEQScalar(obj1, obj3, 123);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == (src1[i] == 123 ? 1 : 0));
    }
    std::cout << "[PASSED] pimEQScaler" << std::endl;
  }

  // Test min scaler
  {
    status = pimMinScalar(obj1, obj3, -1);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == std::min(src1[i], (uint64_t)-1));
    }
    std::cout << "[PASSED] pimMinScaler" << std::endl;
  }

  // Test max scaler
  {
    status = pimMaxScalar(obj1, obj3, 127);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == std::max(src1[i], (uint64_t)127));
    }
    std::cout << "[PASSED] pimMaxScaler" << std::endl;
  }

  // Test popcount
  {
    status = pimPopCount(obj1, obj3);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      uint64_t cnt = std::bitset<32>(src1[i]).count();
      assert(dest[i] == cnt);
    }
    std::cout << "[PASSED] pimPopCount" << std::endl;
  }

  // Test redsum
  {
    uint64_t sumDevice = 0;
    status = pimRedSumUInt(obj1, &sumDevice);
    assert(status == PIM_OK);
    uint64_t sumHost = 0;
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
    uint64_t sumDevice = 0;
    status = pimRedSumRangedUInt(obj1, idxBegin, idxEnd, &sumDevice);
    assert(status == PIM_OK);
    uint64_t sumHost = 0;
    for (unsigned i = idxBegin; i < idxEnd; ++i) {
      sumHost += src1[i];
    }
    assert(sumDevice == sumHost);
    std::cout << "[PASSED] pimRedSumRangedInt" << std::endl;
  }

  // Test broadcast
  {
    status = pimBroadcastInt(obj3, 429496729990);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj3, (void *)dest.data());
    assert(status == PIM_OK);
    for (unsigned i = 0; i < numElements; ++i) {
      assert(dest[i] == 429496729990);
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
      assert(dest[i] == (src1[i] >> 4));
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
      assert(dest[i] == (src1[i] << 2));
    }
    std::cout << "[PASSED] pimShiftBitsLeft" << std::endl;
  }

  pimShowStats();

  status = pimFree(obj1);
  assert(status == PIM_OK);
  status = pimFree(obj2);
  assert(status == PIM_OK);
  status = pimFree(obj3);
  assert(status == PIM_OK);
}

