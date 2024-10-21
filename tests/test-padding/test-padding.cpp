// Test: Test PIM alloc
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


void testAlloc(PimDeviceEnum deviceType)
{
  // 1GB capacity
  unsigned numRanks = 1;
  unsigned numBankPerRank = 1;
  unsigned numSubarrayPerBank = 4;
  unsigned numRows = 1024;
  unsigned numCols = 8192;

  uint64_t numElements = 32 * 1024;

  std::vector<int> srcInt(numElements);
  std::vector<int> srcIntNarrow(numElements);
  std::vector<char> srcChar(numElements);
  std::vector<char> srcCharNarrow(numElements);

  for (uint64_t i = 0; i < numElements; ++i) {
    srcInt[i] = static_cast<int>(i);
    srcIntNarrow[i] = (srcInt[i] & 0xffffff);
    srcChar[i] = i % 256;
    srcCharNarrow[i] = (srcChar[i] & 0xf);
  }

  PimStatus status = pimCreateDevice(deviceType, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
  assert(status == PIM_OK);

  // test PIM allocation
  PimObjId objInt = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_INT32);
  PimObjId objIntNarrow = pimAllocCustomized(PIM_ALLOC_AUTO, numElements, PIM_INT32, 24/*bitsPerElement*/);
  PimObjId objChar = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_INT8);
  PimObjId objCharNarrow = pimAllocCustomized(PIM_ALLOC_AUTO, numElements, PIM_INT8, 4/*bitsPerElement*/);
  assert(objInt != -1);
  assert(objIntNarrow != -1);
  assert(objChar != -1);
  assert(objCharNarrow != -1);

  // copy host to device
  status = pimCopyHostToDevice((void*)srcInt.data(), objInt);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void*)srcInt.data(), objIntNarrow); // auto truncate
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void*)srcChar.data(), objChar);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void*)srcChar.data(), objCharNarrow); // auto truncate
  assert(status == PIM_OK);

  std::vector<int> destInt(numElements);
  std::vector<int> destIntNarrow(numElements);
  std::vector<char> destChar(numElements);
  std::vector<char> destCharNarrow(numElements);

  status = pimCopyDeviceToHost(objInt, (void*)destInt.data());
  assert(status == PIM_OK);
  status = pimCopyDeviceToHost(objIntNarrow, (void*)destIntNarrow.data());
  assert(status == PIM_OK);
  status = pimCopyDeviceToHost(objChar, (void*)destChar.data());
  assert(status == PIM_OK);
  status = pimCopyDeviceToHost(objCharNarrow, (void*)destCharNarrow.data());
  assert(status == PIM_OK);

  if (srcInt != destInt) {
    std::printf("ERROR: Incorrect int32 values\n");
  }
  if (srcIntNarrow != destIntNarrow) {
    std::printf("ERROR: Incorrect int32 truncated values\n");
  }
  if (srcChar != destChar) {
    std::printf("ERROR: Incorrect int8 values\n");
  }
  if (srcCharNarrow != destCharNarrow) {
    std::printf("ERROR: Incorrect int8 truncated values\n");
  }

  pimFree(objInt);
  pimFree(objIntNarrow);
  pimFree(objChar);
  pimFree(objCharNarrow);

  pimShowStats();
  pimResetStats();
  pimDeleteDevice();
}

int main()
{
  std::cout << "PIM Regression Test: PIM alloc and padding" << std::endl;

  testAlloc(PIM_DEVICE_BITSIMD_V);

  testAlloc(PIM_DEVICE_FULCRUM);

  return 0;
}

