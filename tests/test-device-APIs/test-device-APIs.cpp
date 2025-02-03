// Test: Test device APIs
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <cstdio>


void testDeviceAPIs(PimDeviceEnum deviceType)
{
  // 8GB capacity
  unsigned numRanks = 4;
  unsigned numBankPerRank = 32; // 8 chips * 16 banks
  unsigned numSubarrayPerBank = 16;
  unsigned numRows = 1024;
  unsigned numCols = 4096;

  PimStatus status = pimCreateDevice(deviceType, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
  assert(status == PIM_OK);

  PimDeviceProperties deviceProp;
  status = pimGetDeviceProperties(&deviceProp);
  assert(status == PIM_OK);
  assert(deviceProp.deviceType == deviceType);
  assert(deviceProp.numRanks == numRanks);
  assert(deviceProp.numBankPerRank == numBankPerRank);
  assert(deviceProp.numSubarrayPerBank == numSubarrayPerBank);
  assert(deviceProp.numColPerSubarray == numCols);
  assert(deviceProp.numRowPerSubarray == numRows);
  switch(deviceType) {
    case PIM_DEVICE_BITSIMD_V:
      assert(deviceProp.hLayoutDevice == false);
      assert(deviceProp.vLayoutDevice == true);
      break;
    case PIM_DEVICE_FULCRUM:
      assert(deviceProp.hLayoutDevice == true);
      assert(deviceProp.vLayoutDevice == false);
      break;
    case PIM_DEVICE_BANK_LEVEL:
      assert(deviceProp.hLayoutDevice == true);
      assert(deviceProp.vLayoutDevice == false);
      break;
    default:
      break;
  }
  
  pimShowStats();
  pimDeleteDevice();
}

int main()
{
  std::cout << "PIM Regression Test: Device Related APIs" << std::endl;

  testDeviceAPIs(PIM_DEVICE_BITSIMD_V);
  testDeviceAPIs(PIM_DEVICE_FULCRUM);
  testDeviceAPIs(PIM_DEVICE_BANK_LEVEL);

  std::cout << "PIM Regression Test: Device Related APIs Passed!" << std::endl;

  return 0;
}

