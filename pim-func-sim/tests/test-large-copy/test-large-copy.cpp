// Test: Test large data copy
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "libpimsim.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstdlib>
#include <cstdio>


void testLargeCopy(PimDeviceEnum deviceType, const std::vector<int>& src, unsigned numElements)
{
  // 1GB capacity
  unsigned numRanks = 1;
  unsigned numBankPerRank = 128; // 8 chips * 16 banks
  unsigned numSubarrayPerBank = 32;
  unsigned numRows = 256;
  unsigned numCols = 8192;

  PimStatus status = pimCreateDevice(deviceType, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
  assert(status == PIM_OK);

  unsigned bitsPerElement = 32;
  std::vector<int> dest(numElements);

  // test a few iterations
  for (int iter = 0; iter < 2; ++iter) {
    PimObjId obj = pimAlloc(PIM_ALLOC_AUTO, numElements, bitsPerElement, PIM_INT32);
    assert(obj != -1);

    status = pimCopyHostToDevice((void*)src.data(), obj);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj, (void*)dest.data());

    int numError = 0;
    for (unsigned i = 0; i < numElements; ++i) {
      if (src[i] != dest[i]) {
        numError++;
        if (numError < 100) {
          std::printf("ERROR: found mismatch at idx %d: src 0x%x dest 0x%x\n", i, src[i], dest[i]);
        }
      }
    }

    pimFree(obj);
    std::printf("Total mismatch: %d\n", numError);
  }

  pimShowStats();
  pimResetStats();
  pimDeleteDevice();
}

int main()
{
  std::cout << "PIM Regression Test: Large data copy" << std::endl;

  unsigned numElements = 128 * 1024 * 1024 + 2; // 128M + 2 elements, 512MB + 8
  std::vector<int> src(numElements);
  for (unsigned i = 0; i < numElements; ++i) {
    src[i] = i;
  }

  testLargeCopy(PIM_DEVICE_BITSIMD_V, src, numElements);

  testLargeCopy(PIM_DEVICE_FULCRUM, src, numElements);

  return 0;
}

