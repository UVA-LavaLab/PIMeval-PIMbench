// Test: Test reduction min
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <limits> // Include this for UINT_MAX

bool testRedMin(PimDeviceEnum deviceType)
{
  unsigned numRanks = 2;
  unsigned numBankPerRank = 2;
  unsigned numSubarrayPerBank = 8;
  unsigned numRows = 1024;
  unsigned numCols = 8192;

  uint64_t numElements = 65536;
  std::vector<unsigned> src(numElements);
  unsigned expectedMin = std::numeric_limits<unsigned>::max(); // Use limits header for UINT_MAX
  unsigned idxBegin = 12345;
  unsigned idxEnd = 22222;
  unsigned expectedMinRanged = std::numeric_limits<unsigned>::max();

  for (uint64_t i = 0; i < numElements; ++i) {
    src[i] = i;
    if (src[i] < expectedMin) {
      expectedMin = src[i];
    }
    if (i >= idxBegin && i < idxEnd && src[i] < expectedMinRanged) {
      expectedMinRanged = src[i];
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

    uint32_t min = std::numeric_limits<uint32_t>::max();
    status = pimRedMin(obj, static_cast<void*>(&min));
    assert(status == PIM_OK);

    uint32_t minRanged = std::numeric_limits<uint32_t>::max();;
    status = pimRedMin(obj, static_cast<void*>(&minRanged), idxBegin, idxEnd);
    assert(status == PIM_OK);

    std::cout << "Result: RedMin: PIM " << min << " expected " << expectedMin << std::endl;
    std::cout << "Result: RedMinRanged: PIM " << minRanged << " expected " << expectedMinRanged << std::endl;

    if (min == expectedMin && minRanged == expectedMinRanged) {
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
  std::cout << "PIM Regression Test: Reduction Min" << std::endl;

  bool ok = true;
  ok &= testRedMin(PIM_DEVICE_BITSIMD_V);
  ok &= testRedMin(PIM_DEVICE_FULCRUM);
  ok &= testRedMin(PIM_DEVICE_BANK_LEVEL);
  std::cout << (ok ? "ALL PASSED!" : "FAILED!") << std::endl;

  return 0;
}
