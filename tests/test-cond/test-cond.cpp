// Test: Test PIM conditional APIs
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


bool testCond(PimDeviceEnum deviceType)
{
  // 1GB capacity
  unsigned numRanks = 1;
  unsigned numBankPerRank = 1;
  unsigned numSubarrayPerBank = 8;
  unsigned numRows = 1024;
  unsigned numCols = 8192;

  uint64_t numElements = 16 * 1024;

  std::vector<uint8_t> cond(numElements);
  std::vector<int> src1(numElements);
  std::vector<int> src2(numElements);
  std::vector<int> dest1(numElements);
  std::vector<int> dest2(numElements);
  std::vector<int> dest3(numElements);
  std::vector<int> dest4(numElements);
  const int scalarVal = -123;

  for (uint64_t i = 0; i < numElements; ++i) {
    cond[i] = (i % 3) & 1;
    src1[i] = static_cast<int>(i);
    src2[i] = static_cast<int>(i + 1);
  }

  PimStatus status = pimCreateDevice(deviceType, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
  assert(status == PIM_OK);

  // allocate src1/2, dest, and bit slice objects
  PimObjId objSrc1 = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_INT32);
  PimObjId objSrc2 = pimAllocAssociated(objSrc1, PIM_INT32);
  PimObjId objDest1 = pimAllocAssociated(objSrc1, PIM_INT32);
  PimObjId objDest2 = pimAllocAssociated(objSrc1, PIM_INT32);
  PimObjId objDest3 = pimAllocAssociated(objSrc1, PIM_INT32);
  PimObjId objDest4 = pimAllocAssociated(objSrc1, PIM_INT32);
  PimObjId objBool = pimAllocAssociated(objSrc1, PIM_BOOL);  // padding
  assert(objSrc1 != -1 && objSrc2 != -1 && objDest1 != -1 && objDest2 != -1 && objDest3 != -1 && objDest4 != -1);
  assert(objBool != -1);

  // copy host to device
  status = pimCopyHostToDevice((void*)cond.data(), objBool);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void*)src1.data(), objSrc1);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void*)src2.data(), objSrc2);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void*)src2.data(), objDest1);  // for cond copy
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void*)src2.data(), objDest2);  // for cond broadcast
  assert(status == PIM_OK);

  // test conditional operations
  status = pimCondCopy(objBool, objSrc1, objDest1);
  assert(status == PIM_OK);
  status = pimCondBroadcast(objBool, static_cast<uint64_t>(scalarVal), objDest2);
  assert(status == PIM_OK);
  status = pimCondSelect(objBool, objSrc1, objSrc2, objDest3);
  assert(status == PIM_OK);
  status = pimCondSelectScalar(objBool, objSrc1, static_cast<uint64_t>(scalarVal), objDest4);
  assert(status == PIM_OK);

  status = pimCopyDeviceToHost(objDest1, (void*)dest1.data());
  assert(status == PIM_OK);
  status = pimCopyDeviceToHost(objDest2, (void*)dest2.data());
  assert(status == PIM_OK);
  status = pimCopyDeviceToHost(objDest3, (void*)dest3.data());
  assert(status == PIM_OK);
  status = pimCopyDeviceToHost(objDest4, (void*)dest4.data());
  assert(status == PIM_OK);

  bool ok = true;
  for (uint64_t i = 0; i < numElements; ++i) {
    if (dest1[i] != (cond[i] ? src1[i] : src2[i])) {
      ok = false;
      std::printf("CondCopy Test Error: cond %d src %d dest %d\n", (int)cond[i], src1[i], dest1[i]);
    }
  }
  for (uint64_t i = 0; i < numElements; ++i) {
    if (dest2[i] != (cond[i] ? scalarVal : src2[i])) {
      ok = false;
      std::printf("CondBroadcast Test Error: cond %d scalar %d dest %d\n", (int)cond[i], scalarVal, dest1[i]);
    }
  }
  for (uint64_t i = 0; i < numElements; ++i) {
    if (dest3[i] != (cond[i] ? src1[i] : src2[i])) {
      ok = false;
      std::printf("Select Test Error: cond %d src1 %d src2 %d dest %d\n", (int)cond[i], src1[i], src2[i], dest3[i]);
    }
  }
  for (uint64_t i = 0; i < numElements; ++i) {
    if (dest4[i] != (cond[i] ? src1[i] : scalarVal)) {
      ok = false;
      std::printf("SelectScalar Test Error: cond %d src1 %d scalar %d dest %d\n", (int)cond[i], src1[i], scalarVal, dest4[i]);
    }
  }

  pimFree(objSrc1);
  pimFree(objSrc2);
  pimFree(objDest1);
  pimFree(objDest2);
  pimFree(objBool);

  pimShowStats();
  pimResetStats();
  pimDeleteDevice();

  std::cout << "Cond Test " << (ok ? "PASSED" : "FAILED") << std::endl;
  return ok;
}

int main()
{
  std::cout << "PIM Regression Test: PIM conditional operations" << std::endl;

  bool ok = true;
  ok &= testCond(PIM_DEVICE_BITSIMD_V);
  ok &= testCond(PIM_DEVICE_FULCRUM);
  ok &= testCond(PIM_DEVICE_BANK_LEVEL);

  std::cout << (ok ? "PASSED" : "FAILED") << std::endl;
  return 0;
}

