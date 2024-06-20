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
  unsigned numRanks = 1;
  unsigned numBankPerRank = 128; // 8 chips * 16 banks
  unsigned numSubarrayPerBank = 32;
  unsigned numRows = 8192;
  unsigned numCols = 8192;

  PimStatus status = pimCreateDevice(deviceType, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
  assert(status == PIM_OK);

  unsigned bitsPerElement = 32;
  std::vector<int> dest(numElements);

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
  std::printf("Total mismatch: %d\n", numError);

  pimDeleteDevice();
}

int main()
{
  std::cout << "PIM Regression Test: Functional" << std::endl;

  unsigned numElements = 134217730;
  std::vector<int> src(numElements);
  for (unsigned i = 0; i < numElements; ++i) {
    src[i] = i;
  }

  testLargeCopy(PIM_DEVICE_BITSIMD_V, src, numElements);

  testLargeCopy(PIM_DEVICE_FULCRUM, src, numElements);

  return 0;
}

