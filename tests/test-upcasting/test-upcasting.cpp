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

  uint64_t numElements = 48;

  std::vector<int16_t> src(numElements);
  std::vector<int8_t> dest(numElements);

  for (uint64_t i = 0; i < numElements; ++i) {
    src[i] = static_cast<int16_t>(i);
  }

  PimStatus status = pimCreateDevice(deviceType, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
  assert(status == PIM_OK);

  bool ok = true;
  // Emitting Allocations
  PimObjId fuse_root = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_INT8);
  PimObjId fuse_expr_0 = pimAllocAssociated(fuse_root, PIM_INT16);
  // Emitting Copy Host to Device
  pimCopyHostToDevice((void*)src.data(), fuse_expr_0, 0UL, 0UL);
  // Creating PIM Fused Program
  pimConvertType(fuse_expr_0 , fuse_root);
  // Emitting Copy Device to Host
  pimCopyDeviceToHost(fuse_root,(void*)dest.data(), 0UL, 0UL);
  
  // Emitting Deallocations
  pimFree(fuse_root);
  pimFree(fuse_expr_0);
  pimShowStats();
  pimResetStats();
  pimDeleteDevice();

  // Verifying results
  for (uint64_t i = 0; i < numElements; ++i) {
    if (dest[i] != static_cast<int8_t>(src[i])) {
      std::cout << "Mismatch at index " << i << ": expected " << static_cast<int8_t>(src[i]) << ", got " << static_cast<int>(dest[i]) << std::endl;
      ok = false;
      break;
    }
  }

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

