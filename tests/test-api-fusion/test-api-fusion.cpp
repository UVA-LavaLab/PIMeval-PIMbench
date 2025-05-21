// Test: Test PIM API Fusion
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


bool testFused(PimDeviceEnum deviceType)
{
  // 1GB capacity
  unsigned numRanks = 1;
  unsigned numBankPerRank = 1;
  unsigned numSubarrayPerBank = 8;
  unsigned numRows = 1024;
  unsigned numCols = 8192;

  uint64_t numElements = 16 * 1024;

  std::vector<int> src1(numElements);
  std::vector<int> src2(numElements);
  std::vector<int> dest1(numElements);
  std::vector<int> dest2(numElements);
  const int scalarVal = -123;

  for (uint64_t i = 0; i < numElements; ++i) {
    src1[i] = static_cast<int>(i);
    src2[i] = static_cast<int>(i + 1);
  }

  PimStatus status = pimCreateDevice(deviceType, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
  assert(status == PIM_OK);

  PimObjId objSrc1 = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_INT32);
  PimObjId objSrc2 = pimAllocAssociated(objSrc1, PIM_INT32);
  PimObjId objDest1 = pimAllocAssociated(objSrc1, PIM_INT32);
  PimObjId objDest2 = pimAllocAssociated(objSrc1, PIM_INT32);
  assert(objSrc1 != -1 && objSrc2 != -1 && objDest1 != -1 && objDest2 != -1);

  // copy host to device
  status = pimCopyHostToDevice((void*)src1.data(), objSrc1);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void*)src2.data(), objSrc2);
  assert(status == PIM_OK);


  // Fused PIM APIs
  PimProg prog;
  prog.add(pimMulScalar, objSrc1, objDest1, static_cast<uint64_t>(scalarVal));
  prog.add(pimAdd, objDest1, objSrc2, objDest1);
  status = pimFuse(prog);
  assert(status == PIM_OK);

  // Direct APIs
  status = pimScaledAdd(objSrc1, objSrc2, objDest2, static_cast<uint64_t>(scalarVal));
  assert(status == PIM_OK);


  status = pimCopyDeviceToHost(objDest1, (void*)dest1.data());
  assert(status == PIM_OK);
  status = pimCopyDeviceToHost(objDest2, (void*)dest2.data());
  assert(status == PIM_OK);

  bool ok = true;
  for (uint64_t i = 0; i < numElements; ++i) {
    if (dest1[i] != (src1[i] * scalarVal + src2[i]) || dest1[i] != dest2[i]) {
      ok = false;
      std::printf("Fused Test Error: src1 %d src2 %d dest1 %d dest2 %d\n", src1[i], src2[i], dest1[i], dest2[i]);
    }
  }

  pimFree(objSrc1);
  pimFree(objSrc2);
  pimFree(objDest1);
  pimFree(objDest2);

  pimShowStats();
  pimResetStats();
  pimDeleteDevice();

  std::cout << "Fused Test " << (ok ? "PASSED" : "FAILED") << std::endl;
  return ok;
}

int main()
{
  std::cout << "PIM Regression Test: PIM fused operations" << std::endl;

  bool ok = true;
  ok &= testFused(PIM_DEVICE_BITSIMD_V);
  ok &= testFused(PIM_DEVICE_FULCRUM);
  ok &= testFused(PIM_DEVICE_BANK_LEVEL);

  std::cout << (ok ? "PASSED" : "FAILED") << std::endl;
  return 0;
}

