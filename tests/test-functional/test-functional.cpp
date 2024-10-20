// Test: Test functional behavior
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "test-functional.h"
#include "test-functional-int.h"
#include "test-functional-fp.h"
#include "libpimeval.h"
#include <iostream>
#include <cassert>
#include <cstdint>

//! @brief  Create PIM device
void
testFunctional::createPimDevice()
{
  unsigned numCores = 4;
  unsigned numRows = 1024;
  unsigned numCols = 256;

  PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, 1, 1, numCores, numRows, numCols);
  assert(status == PIM_OK);
}

//! @brief  Delete PIM device
void
testFunctional::deletePimDevice()
{
  PimStatus status = pimDeleteDevice();
  assert(status == PIM_OK);
}

//! @brief  Run all tests
bool
testFunctional::runAllTests()
{
  bool ok = true;

  createPimDevice();

  ok &= testInt<int8_t>("INT8", PIM_INT8);
  ok &= testInt<uint8_t>("UINT8", PIM_UINT8);
  ok &= testInt<int16_t>("INT16", PIM_INT16);
  ok &= testInt<uint16_t>("UINT16", PIM_UINT16);
  ok &= testInt<int32_t>("INT32", PIM_INT32);
  ok &= testInt<uint32_t>("UINT32", PIM_UINT32);
  ok &= testInt<int64_t>("INT64", PIM_INT64);
  ok &= testInt<uint64_t>("UINT64", PIM_UINT64);

  ok &= testFp<float>("FP32", PIM_FP32);

  deletePimDevice();

  return ok;
}

//! @brief  Main entry
int main()
{
  std::cout << "PIMeval Functional Testing" << std::endl;

  testFunctional test;
  bool ok = test.runAllTests();

  std::cout << "Result: " << (ok ? "COMPLETED" : "FAILED") << std::endl;
  return 0;
}

