// Test: Test buffer functionalities
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
#include <limits>


// test UINT32 reduction sum
bool testBuffer(PimDeviceEnum deviceType)
{
  unsigned numRanks = 2;
  unsigned numBankPerRank = 2;
  unsigned numSubarrayPerBank = 8;
  unsigned numRows = 1024;
  unsigned numCols = 8192;

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
    src[i] = std::numeric_limits<unsigned>::max() - i;  // test when sum is greater than unsigned max
    sum32 += src[i];
    sum64 += src[i];
    if (i >= idxBegin && i < idxEnd) {
      sumRanged32 += src[i];
      sumRanged64 += src[i];
    }
  }

  PimStatus status = pimCreateDevice(deviceType, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols, numCols / 8);
  assert(status == PIM_OK);

  // test a few iterations
  bool ok = true;
  PimObjId buffObj = pimAllocBuffer(numCols/32, PIM_INT32);
  assert(buffObj != -1);

  PimObjId obj = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_INT32);  // non-associated
  assert(obj != -1);

  PimObjId objAssociated = pimAllocAssociated(obj, PIM_INT32);  // associated
  assert(objAssociated != -1);

  pimShowStats();
  pimResetStats();
  pimDeleteDevice();
  return ok;
}

int main()
{
  std::cout << "PIM Test: Buffer Functionalities" << std::endl;

  bool ok = true;

  ok &= testBuffer(PIM_DEVICE_AIM);

  std::cout << (ok ? "ALL PASSED!" : "FAILED!") << std::endl;

  return 0;
}

