// Test: Test buffer functionalities
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
#include "util.h"

// test UINT32 reduction sum
bool testBuffer(PimDeviceEnum deviceType)
{
  unsigned numRanks = 2;
  unsigned numBankPerRank = 2;
  unsigned numSubarrayPerBank = 8;
  unsigned numRows = 1024;
  unsigned numCols = 8192;

  uint64_t numElements = 1024;  // Smaller test size
  std::vector<int> src(numElements);
  std::vector<int> buffer_data(numCols/32);  // Buffer data
  std::vector<int> dest(numElements);
  
  getVector(numElements, src);  // Initialize source data
  getVector(buffer_data.size(), buffer_data);  // Initialize buffer data

  std::cout << "Test setup:" << std::endl;
  std::cout << "Source elements: " << numElements << std::endl;
  std::cout << "Buffer elements: " << buffer_data.size() << std::endl;

  PimStatus status = pimCreateDevice(deviceType, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols, numCols / 8);
  assert(status == PIM_OK);

  // test a few iterations
  bool ok = true;
  PimObjId buffObj = pimAllocBuffer(numCols/32, PIM_INT32);
  assert(buffObj != -1);

  PimObjId obj = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_INT32);  // non-associated
  assert(obj != -1);

  status = pimCopyHostToDevice((void*)buffer_data.data(), buffObj);
  assert(status == PIM_OK);

  status = pimCopyHostToDevice((void*)src.data(), obj);
  assert(status == PIM_OK);

  PimDeviceProperties deviceProperties;
  status = pimGetDeviceProperties(&deviceProperties);
  assert(status == PIM_OK);
  
  std::cout << "Device has " << deviceProperties.numPIMCores << " cores" << std::endl;
  std::cout << "Buffer size: " << buffer_data.size() << " elements" << std::endl;
  std::cout << "Source data size: " << numElements << " elements" << std::endl;
  std::cout << "Elements per core (estimated): " << numElements / deviceProperties.numPIMCores << std::endl;

  std::vector<int64_t> mac_results(deviceProperties.numPIMCores, 0);
  status = pimMAC(obj, buffObj, (void*)mac_results.data());
  assert(status == PIM_OK);
  
  // Calculate expected results using your original logic
  for (unsigned core = 0; core < deviceProperties.numPIMCores; ++core) {
    int64_t expected_result = 0;
    for (size_t i = 0; i < buffer_data.size(); ++i) {
      expected_result += src[buffer_data.size()*core + i] * buffer_data[i];
    }
    if (mac_results[core] != expected_result) {
      std::cout << "Core " << core << " result mismatch: expected " << expected_result << ", got " << mac_results[core] << std::endl;
      ok = false;
    } else {
      std::cout << "Core " << core << " result matches: " << mac_results[core] << std::endl;
    }
  }

  pimShowStats();
  pimResetStats();
  pimDeleteDevice();
  return ok;
}

int main()
{
  std::cout << "PIM Test: Buffer Functionalities" << std::endl;

  bool ok = true;

  ok &= testBuffer(PIM_DEVICE_AIM);

  std::cout << (ok ? "ALL PASSED!" : "FAILED!") << std::endl;

  return 0;
}

