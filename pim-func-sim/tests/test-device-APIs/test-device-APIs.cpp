// Test: Test device APIs
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

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
  assert(deviceProp.numRanks = numRanks);
  assert(deviceProp.numBankPerRank = numBankPerRank);
  assert(deviceProp.numSubarrayPerBank = numSubarrayPerBank);
  assert(deviceProp.numColPerSubarray = numCols);
  assert(deviceProp.numRowPerSubarray = numRows);

  pimShowStats();
  pimResetStats();
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

