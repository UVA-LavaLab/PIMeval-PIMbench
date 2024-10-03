// Test: Test reduction sum
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstdlib>
#include <cstdio>


void testRedSum(PimDeviceEnum deviceType)
{
  // 8GB capacity
  unsigned numRanks = 1;
  unsigned numBankPerRank = 2;
  unsigned numSubarrayPerBank = 4;
  unsigned numRows = 1024;
  unsigned numCols = 1024;

  uint64_t numElements = 65536;
  std::vector<unsigned> src(numElements);
  std::vector<unsigned> dest(numElements);
  unsigned sum32 = 0;
  uint64_t sum64 = 0;
  unsigned sumRanged32 = 0;
  uint64_t sumRanged64 = 0;
  unsigned idxBegin = 12345;
  unsigned idxEnd = 22222;
  for (uint64_t i = 0; i < numElements; ++i) {
    src[i] = i;
    sum32 += src[i];
    sum64 += src[i];
    if (i >= idxBegin && i < idxEnd) {
      sumRanged32 += src[i];
      sumRanged64 += src[i];
    }
  }

  PimStatus status = pimCreateDevice(deviceType, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
  assert(status == PIM_OK);

  // test a few iterations
  for (int iter = 0; iter < 2; ++iter) {
    PimObjId obj = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_UINT32);
    assert(obj != -1);

    status = pimCopyHostToDevice((void*)src.data(), obj);
    assert(status == PIM_OK);
    uint64_t sum = 0;
    status = pimRedSumUInt(obj, &sum);
    assert(status == PIM_OK);

    uint64_t sumRanged = 0;
    status = pimRedSumRangedUInt(obj, idxBegin, idxEnd, &sumRanged);
    assert(status == PIM_OK);

    std::cout << "Result: RedSum: PIM " << sum << " expected 32-bit " << sum32 << " 64-bit " << sum64 << std::endl;
    std::cout << "Result: RedSumRanged: PIM " << sumRanged << " expected 32-bit " << sumRanged32 << " 64-bit " << sumRanged64 << std::endl;

    if (sum == sum32 && sumRanged == sumRanged32) {
      std::cout << "Passed!" << std::endl;
    } else {
      std::cout << "Failed!" << std::endl;
    }

    pimFree(obj);
  }

  pimShowStats();
  pimResetStats();
  pimDeleteDevice();
}

int main()
{
  std::cout << "PIM Regression Test: Reduction Sum" << std::endl;

  testRedSum(PIM_DEVICE_BITSIMD_V);

  testRedSum(PIM_DEVICE_FULCRUM);

  return 0;
}

