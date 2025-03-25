// Test: Test PIM APIs with mixed data types
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


bool testMixedTypes(PimDeviceEnum deviceType)
{
  unsigned numRanks = 1;
  unsigned numBankPerRank = 2;
  unsigned numSubarrayPerBank = 4;
  unsigned numRows = 1024;
  unsigned numCols = 8192;

  uint64_t numElements = 65536;

  std::vector<int> src1(numElements);
  std::vector<int> src2(numElements);
  std::vector<int> dest(numElements);

  for (uint64_t i = 0; i < numElements; ++i) {
    src1[i] = static_cast<int>(i % 1000);
    src2[i] = static_cast<int>((i + 1) % 777);
  }

  PimStatus status = pimCreateDevice(deviceType, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
  assert(status == PIM_OK);

  // allocate src1/2, dest, and bit slice objects
  PimObjId objSrc1 = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_INT32);
  PimObjId objSrc2 = pimAllocAssociated(objSrc1, PIM_INT32);
  PimObjId objSrc3 = pimAllocAssociated(objSrc1, PIM_INT8);
  PimObjId objSrc4 = pimAllocAssociated(objSrc1, PIM_INT8);
  PimObjId objDest = pimAllocAssociated(objSrc1, PIM_INT32);
  PimObjId objBool1 = pimAllocAssociated(objSrc1, PIM_BOOL);  // padding
  PimObjId objBool2 = pimAllocAssociated(objSrc1, PIM_BOOL);  // padding
  PimObjId objBool3 = pimAllocAssociated(objSrc1, PIM_BOOL);  // padding
  PimObjId objBool4 = pimAllocAssociated(objSrc1, PIM_BOOL);  // padding
  assert(objSrc1 != -1 && objSrc2 != -1 && objDest != -1);
  assert(objBool1 != -1 && objBool2 != -1 && objBool3 != -1 && objBool4 != -1);

  // copy host to device
  status = pimCopyHostToDevice((void*)src1.data(), objSrc1);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void*)src2.data(), objSrc2);
  assert(status == PIM_OK);

  // Performe mixed data type operations
  bool ok = true;

  // broadcast
  status = pimBroadcastInt(objDest, 1234); assert(status == PIM_OK);
  status = pimCopyDeviceToHost(objDest, (void*)dest.data()); assert(status == PIM_OK);
  for (uint64_t i = 0; i < numElements; ++i) {
    if (dest[i] != 1234) {
      ok = false;
      std::printf("Error: Broadcast: dest %d, expected 1234\n", dest[i]);
    }
  }

  // type conversion: int32 -> int8 -> int32
  status = pimConvertType(objSrc1, objSrc3); assert(status == PIM_OK);
  status = pimConvertType(objSrc3, objDest); assert(status == PIM_OK);
  status = pimCopyDeviceToHost(objDest, (void*)dest.data()); assert(status == PIM_OK);
  for (uint64_t i = 0; i < numElements; ++i) {
    int expected = static_cast<int>(static_cast<int8_t>(src1[i]));
    if (dest[i] != expected) {
      ok = false;
      std::printf("Error: Convert type: src1 0x%x, dest 0x%x, expected 0x%x\n", src1[i], dest[i], expected);
    }
  }

  // comparison, bit slice
  status = pimBroadcastInt(objDest, 0); assert(status == PIM_OK);
  status = pimLT(objSrc1, objSrc2, objBool1); assert(status == PIM_OK);
  status = pimBitSliceInsert(objBool1, objDest, 0); assert(status == PIM_OK);
  status = pimBitSliceExtract(objSrc1, objBool2, 0); assert(status == PIM_OK);
  status = pimBitSliceInsert(objBool2, objDest, 1); assert(status == PIM_OK);
  status = pimCopyDeviceToHost(objDest, (void*)dest.data()); assert(status == PIM_OK);
  for (uint64_t i = 0; i < numElements; ++i) {
    int expected = ((src1[i] < src2[i] ? 1 : 0) | (src1[i] % 2 == 1 ? 2 : 0));
    if (dest[i] != expected) {
      ok = false;
      std::printf("Error: LT + bit-slice: src1 0x%x, src2 0x%x, dest 0x%x, expected 0x%x\n", src1[i], src2[i], dest[i], expected);
    }
  }

  // add mixed type - accumulate bool for pop count
  status = pimBroadcastInt(objDest, 0); assert(status == PIM_OK);
  for (int i = 0; i < 32; i++) {
    status = pimBitSliceExtract(objSrc1, objBool1, i); assert(status == PIM_OK);
    status = pimAdd(objDest, objBool1, objDest); assert(status == PIM_OK);
  }
  status = pimCopyDeviceToHost(objDest, (void*)dest.data()); assert(status == PIM_OK);
  for (uint64_t i = 0; i < numElements; ++i) {
    int expected = std::bitset<32>(src1[i]).count();
    if (dest[i] != expected) {
      ok = false;
      std::printf("Error: int + bool: src1 0x%x, dest 0x%x, expected 0x%x\n", src1[i], dest[i], expected);
    }
  }

  pimFree(objSrc1);
  pimFree(objSrc2);
  pimFree(objSrc3);
  pimFree(objSrc4);
  pimFree(objDest);
  pimFree(objBool1);
  pimFree(objBool2);
  pimFree(objBool3);
  pimFree(objBool4);

  pimShowStats();
  pimResetStats();
  pimDeleteDevice();

  std::cout << "Mixed data type test " << (ok ? "PASSED" : "FAILED") << std::endl;
  return ok;
}

int main()
{
  std::cout << "PIM Regression Test: PIM mixed data type tests" << std::endl;

  bool ok = true;
  ok &= testMixedTypes(PIM_DEVICE_BITSIMD_V);
  ok &= testMixedTypes(PIM_DEVICE_FULCRUM);
  ok &= testMixedTypes(PIM_DEVICE_BANK_LEVEL);

  std::cout << (ok ? "PASSED" : "FAILED") << std::endl;
  return 0;
}

