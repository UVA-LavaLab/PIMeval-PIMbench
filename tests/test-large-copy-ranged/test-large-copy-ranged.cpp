// Test: Test large data copy within a range
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


void testLargeCopy(PimDeviceEnum deviceType)
{
  // 8GB capacity
  unsigned numRanks = 1;
  unsigned numBankPerRank = 128; // 8 chips * 16 banks
  unsigned numSubarrayPerBank = 32;
  unsigned numRows = 2048;
  unsigned numCols = 8192;

  uint64_t numElements = 2LL * 1024 * 1024 + 2; // 4G + 2 elements
  uint64_t HostNumElements = numElements;  
  std::vector<char> src(HostNumElements);
  std::vector<char> dest(HostNumElements);
  for (uint64_t i = 0; i < HostNumElements; ++i) {
    src[i] = i % 256;
  }

  PimStatus status = pimCreateDevice(deviceType, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
  assert(status == PIM_OK);

  // test a few iterations
  for (int iter = 0; iter < 2; ++iter) {
    PimObjId obj = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_INT8);
    assert(obj != -1);

    status = pimCopyHostToDevice((void*)src.data(), obj, 0, HostNumElements);
    assert(status == PIM_OK);
    status = pimCopyDeviceToHost(obj, (void*)dest.data(), 0, HostNumElements);

    uint64_t numError = 0;
    for (uint64_t i = 0; i < HostNumElements; ++i) {
      if (src[i] != dest[i]) {
        numError++;
        if (numError < 100) {
          std::printf("ERROR: found mismatch at idx %lld: src 0x%x dest 0x%x\n", i, src[i], dest[i]);
        }
      }
    }

    pimFree(obj);
    std::printf("Total mismatch: %lld\n", numError);
  }

  pimShowStats();
  pimResetStats();
  pimDeleteDevice();
}

int main()
{
  std::cout << "PIM Regression Test: Large data copy" << std::endl;

  testLargeCopy(PIM_DEVICE_BITSIMD_V);

  testLargeCopy(PIM_DEVICE_FULCRUM);

  return 0;
}

