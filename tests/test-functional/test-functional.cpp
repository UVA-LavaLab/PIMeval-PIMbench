// Test: Test functional behavior
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "test-functional.h"
#include "libpimeval.h"
#include <iostream>
#include <cassert>

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
  createPimDevice();
  testI8();
  testI16();
  testI32();
  testI64();
  testU8();
  testU16();
  testU32();
  testU64();
  deletePimDevice();
  return true;
}

//! @brief  Main entry
int main()
{
  std::cout << "PIM Regression Test: Functional" << std::endl;

  testFunctional test;
  bool ok = test.runAllTests();

  std::cout << "Result: " << (ok ? "PASSED" : "FAILED") << std::endl;
  return 0;
}

