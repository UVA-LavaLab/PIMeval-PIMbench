// Test: Test PIM bit slice operations
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


bool testBitSlice(PimDeviceEnum deviceType, bool runArithmetic)
{
  // 1GB capacity
  unsigned numRanks = 1;
  unsigned numBankPerRank = 1;
  unsigned numSubarrayPerBank = 4;
  unsigned numRows = 1024;
  unsigned numCols = 8192;

  uint64_t numElements = 16 * 1024;

  std::vector<int> src1(numElements);
  std::vector<int> src2(numElements);
  std::vector<int> dest(numElements);

  for (uint64_t i = 0; i < numElements; ++i) {
    src1[i] = static_cast<int>(i);
    src2[i] = static_cast<int>(i + 1);
  }

  PimStatus status = pimCreateDevice(deviceType, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
  assert(status == PIM_OK);

  // allocate src1/2, dest, and bit slice objects
  PimObjId objSrc1 = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_INT32);
  PimObjId objSrc2 = pimAllocAssociated(objSrc1, PIM_INT32);
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

  // Performe functional bit-serial int32 addition using bit-slice APIs
  if (runArithmetic) {
    status = pimAdd(objSrc1, objSrc2, objDest);
    assert(status == PIM_OK);
  } else {
    status = pimBroadcastUInt(objBool3, 0); // carry
    assert(status == PIM_OK);
    for (unsigned i = 0; i < 32; i++) {
      // SUM = A XOR B XOR CIN
      status = pimBitSliceExtract(objSrc1, objBool1, i);  assert(status == PIM_OK);
      status = pimBitSliceExtract(objSrc2, objBool2, i);  assert(status == PIM_OK);
      status = pimXor(objBool1, objBool2, objBool4);      assert(status == PIM_OK);
      status = pimXor(objBool3, objBool4, objBool4);      assert(status == PIM_OK);
      status = pimBitSliceInsert(objBool4, objDest, i);   assert(status == PIM_OK);
      // COUT = (A AND B) OR (A AND CIN)
      status = pimAnd(objBool1, objBool2, objBool3);      assert(status == PIM_OK);
      status = pimAnd(objBool1, objBool2, objBool4);      assert(status == PIM_OK);
      status = pimOr(objBool3, objBool4, objBool3);       assert(status == PIM_OK);
    }
  }

  status = pimCopyDeviceToHost(objDest, (void*)dest.data());
  assert(status == PIM_OK);

  bool ok = true;
  for (uint64_t i = 0; i < numElements; ++i) {
    if (dest[i] != src1[i] + src2[i]) {
      ok = false;
      std::printf("Bit-slice Test Error: src1 0x%x src2 0x%x dest 0x%x\n", src1[i], src2[i], dest[i]);
    }
  }

  pimFree(objSrc1);
  pimFree(objSrc2);
  pimFree(objDest);
  pimFree(objBool1);
  pimFree(objBool2);
  pimFree(objBool3);
  pimFree(objBool4);

  pimShowStats();
  pimResetStats();
  pimDeleteDevice();

  std::cout << "Bit-slice Test " << (ok ? "PASSED" : "FAILED") << std::endl;
  return ok;
}

int main()
{
  std::cout << "PIM Regression Test: PIM bit slice operations" << std::endl;

  bool ok = true;
  ok &= testBitSlice(PIM_DEVICE_BITSIMD_V, false);
  ok &= testBitSlice(PIM_DEVICE_FULCRUM, false);
  ok &= testBitSlice(PIM_DEVICE_BANK_LEVEL, false);
  ok &= testBitSlice(PIM_DEVICE_BITSIMD_V, true);
  ok &= testBitSlice(PIM_DEVICE_FULCRUM, true);
  ok &= testBitSlice(PIM_DEVICE_BANK_LEVEL, true);

  std::cout << (ok ? "PASSED" : "FAILED") << std::endl;
  return 0;
}

