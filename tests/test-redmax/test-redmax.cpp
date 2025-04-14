// Test: Test reduction max
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <limits> // Include this for numeric_limits

bool testRedMax(PimDeviceEnum deviceType)
{
  unsigned numRanks = 2;
  unsigned numBankPerRank = 2;
  unsigned numSubarrayPerBank = 8;
  unsigned numRows = 1024;
  unsigned numCols = 8192;

  uint64_t numElements = 65536;
  std::vector<unsigned> src(numElements);
  unsigned expectedMax = std::numeric_limits<unsigned>::min(); // Start with the smallest possible value
  unsigned idxBegin = 12345;
  unsigned idxEnd = 22222;
  unsigned expectedMaxRanged = std::numeric_limits<unsigned>::min();

  for (uint64_t i = 0; i < numElements; ++i) {
    src[i] = i;
    if (src[i] > expectedMax) {
      expectedMax = src[i];
    }
    if (i >= idxBegin && i < idxEnd && src[i] > expectedMaxRanged) {
      expectedMaxRanged = src[i];
    }
  }

  PimStatus status = pimCreateDevice(deviceType, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
  assert(status == PIM_OK);

  bool ok = true;
  for (int iter = 0; iter < 2; ++iter) {
    PimObjId obj = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_UINT32);
    assert(obj != -1);

    status = pimCopyHostToDevice((void*)src.data(), obj);
    assert(status == PIM_OK);

    uint64_t max = std::numeric_limits<uint32_t>::lowest();
    status = pimRedMax(obj, &max);
    assert(status == PIM_OK);

    uint64_t maxRanged = std::numeric_limits<uint32_t>::lowest();;
    status = pimRedMax(obj, &maxRanged, idxBegin, idxEnd);
    assert(status == PIM_OK);

    std::cout << "Result: RedMax: PIM " << max << " expected " << expectedMax << std::endl;
    std::cout << "Result: RedMaxRanged: PIM " << maxRanged << " expected " << expectedMaxRanged << std::endl;

    if (max == expectedMax && maxRanged == expectedMaxRanged) {
      std::cout << "Passed!" << std::endl;
    } else {
      std::cout << "Failed!" << std::endl;
      ok = false;
    }

    pimFree(obj);
  }

  pimShowStats();
  pimResetStats();
  pimDeleteDevice();
  return ok;
}

int main()
{
  std::cout << "PIM Regression Test: Reduction Max" << std::endl;

  bool ok = true;
  ok &= testRedMax(PIM_DEVICE_BITSIMD_V);
  ok &= testRedMax(PIM_DEVICE_FULCRUM);
  ok &= testRedMax(PIM_DEVICE_BANK_LEVEL);
  std::cout << (ok ? "ALL PASSED!" : "FAILED!") << std::endl;

  return 0;
}
