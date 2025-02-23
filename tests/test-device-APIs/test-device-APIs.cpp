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
  unsigned numRanks = 1;
  unsigned numBankPerRank = 2;
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

  // Special handling for functional simulation target
  if (deviceType == PIM_FUNCTIONAL) {
    assert(deviceProp.simTarget != PIM_FUNCTIONAL);
    assert(deviceProp.simTarget != PIM_DEVICE_NONE);
    deviceType = deviceProp.simTarget;
  } else {
    assert(deviceProp.simTarget == deviceType);
  }

  switch(deviceType) {
    case PIM_DEVICE_BITSIMD_V:
      assert(deviceProp.isHLayoutDevice == false);
      break;
    case PIM_DEVICE_FULCRUM:
      assert(deviceProp.isHLayoutDevice == true);
      break;
    case PIM_DEVICE_BANK_LEVEL:
      assert(deviceProp.isHLayoutDevice == true);
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
  testDeviceAPIs(PIM_FUNCTIONAL);

  std::cout << "PIM Regression Test: Device Related APIs Passed!" << std::endl;

  return 0;
}

