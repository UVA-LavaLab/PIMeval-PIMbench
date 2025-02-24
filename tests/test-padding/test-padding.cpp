// Test: Test PIM alloc and padding
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


bool testAlloc(PimDeviceEnum deviceType)
{
  // 1GB capacity
  unsigned numRanks = 1;
  unsigned numBankPerRank = 1;
  unsigned numSubarrayPerBank = 4;
  unsigned numRows = 1024;
  unsigned numCols = 8192;

  uint64_t numElements = 32 * 1024;

  std::vector<int> srcInt(numElements);
  std::vector<uint8_t> srcChar(numElements);
  std::vector<uint8_t> srcBool(numElements);

  for (uint64_t i = 0; i < numElements; ++i) {
    srcInt[i] = static_cast<int>(i);
    srcChar[i] = i % 256;
    srcBool[i] = i % 2;
  }

  PimStatus status = pimCreateDevice(deviceType, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
  assert(status == PIM_OK);

  // test PIM allocation
  PimObjId objInt = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_INT32);
  PimObjId objChar = pimAllocAssociated(objInt, PIM_INT8);  // padding
  PimObjId objBool = pimAllocAssociated(objInt, PIM_BOOL);  // padding
  assert(objInt != -1);
  assert(objChar != -1);
  assert(objBool != -1);

  // copy host to device
  status = pimCopyHostToDevice((void*)srcInt.data(), objInt);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void*)srcChar.data(), objChar);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void*)srcBool.data(), objBool);
  assert(status == PIM_OK);

  status = pimNot(objInt, objInt);
  assert(status == PIM_OK);
  status = pimNot(objChar, objChar);
  assert(status == PIM_OK);
  status = pimNot(objBool, objBool);
  assert(status == PIM_OK);

  std::vector<int> destInt(numElements);
  std::vector<uint8_t> destChar(numElements);
  std::vector<uint8_t> destBool(numElements);

  status = pimCopyDeviceToHost(objInt, (void*)destInt.data());
  assert(status == PIM_OK);
  status = pimCopyDeviceToHost(objChar, (void*)destChar.data());
  assert(status == PIM_OK);
  status = pimCopyDeviceToHost(objBool, (void*)destBool.data());
  assert(status == PIM_OK);

  bool okInt = true;
  bool okChar = true;
  bool okBool = true;
  for (uint64_t i = 0; i < numElements; ++i) {
    if (srcInt[i] != ~destInt[i]) {
      okInt = false;
      std::printf("Error: PIM_INT32 src 0x%x dest 0x%x\n", srcInt[i], destInt[i]);
    }
    if (srcChar[i] != (uint8_t)~destChar[i]) {  // Note: Cast back after ~
      okChar = false;
      std::printf("Error: PIM_INT8 src 0x%x dest 0x%x\n", (unsigned int)srcChar[i], (unsigned int)destChar[i]);
    }
    if (srcBool[i] != !destBool[i]) {  // Note: Use logical but not bitwise
      okBool = false;
      std::printf("Error: PIM_BOOL src 0x%x dest 0x%x\n", (unsigned int)srcBool[i], (unsigned int)destBool[i]);
    }
  }

  pimFree(objInt);
  pimFree(objChar);
  pimFree(objBool);

  pimShowStats();
  pimResetStats();
  pimDeleteDevice();

  std::cout << "PIM_INT32 " << (okInt ? "PASSED" : "FAILED") << std::endl;
  std::cout << "PIM_INT8 " << (okChar ? "PASSED" : "FAILED") << std::endl;
  std::cout << "PIM_BOOL " << (okBool ? "PASSED" : "FAILED") << std::endl;
  return okInt && okChar && okBool;
}

int main()
{
  std::cout << "PIM Regression Test: PIM alloc and padding" << std::endl;

  bool ok = true;
  ok &= testAlloc(PIM_DEVICE_BITSIMD_V);
  ok &= testAlloc(PIM_DEVICE_FULCRUM);

  std::cout << (ok ? "PASSED" : "FAILED") << std::endl;
  return 0;
}

