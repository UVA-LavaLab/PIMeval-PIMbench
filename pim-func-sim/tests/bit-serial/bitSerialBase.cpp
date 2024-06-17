// Bit-Serial Performance Modeling - Base
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "bitSerialBase.h"
#include <vector>
#include <random>
#include <iostream>
#include <string>

//! @brief  Create device
void
bitSerialBase::createDevice()
{
  // create with any v-layout PIM device
  PimDeviceEnum deviceType = getDeviceType();
  PimStatus status = pimCreateDevice(deviceType, 1, 1, 16, 8192, 1024);
  assert(status == PIM_OK);
}

//! @brief  Delete device
void
bitSerialBase::deleteDevice()
{
  PimStatus status = pimDeleteDevice();
  assert(status == PIM_OK);
}

//! @brief  Generate random int
std::vector<int>
bitSerialBase::getRandInt(int numElements, int min, int max, bool allowZero)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(min, max);
  std::vector<int> vec(numElements);
  for (int i = 0; i < numElements; ++i) {
    int val = 0;
    do {
      val = dis(gen);
    } while (val == 0 && !allowZero);
    vec[i] = val;
  }
  return vec;
}

//! @brief  Run tests
bool
bitSerialBase::runTests()
{
  bool ok = true;

  createDevice();

  ok &= testInt32();

  deleteDevice();

  return ok;
}

//! @brief  Test PIM_INT32
bool
bitSerialBase::testInt32()
{
  bool ok = true;
  std::cout << "[INT32] Run tests ..." << std::endl;

  // create vectors
  int numElements = 4000;
  int maxVal = 100000;
  int minVal = -100000;
  std::vector<int> vec1 = getRandInt(numElements, minVal, maxVal);
  std::vector<int> vec2 = getRandInt(numElements, minVal, maxVal);
  std::vector<int> vec3 = getRandInt(numElements, minVal, maxVal, false);
  std::vector<int> vec4 = getRandInt(numElements, minVal, maxVal);
  std::vector<int> vec5(numElements);
  std::vector<int> vec1a(numElements);
  std::vector<int> vec2a(numElements);
  std::vector<int> vec3a(numElements);
  std::vector<int> vec4a(numElements);
  std::vector<int> vec5a(numElements);
  // create EQ cases
  vec2[100] = vec1[100];
  vec2[3000] = vec1[3000];

  // allocate PIM objects
  PimObjId src1 = pimAlloc(PIM_ALLOC_V1, numElements, 32, PIM_INT32);
  PimObjId src2 = pimAllocAssociated(32, src1, PIM_INT32);
  PimObjId src3 = pimAllocAssociated(32, src1, PIM_INT32);
  PimObjId dest1 = pimAllocAssociated(32, src1, PIM_INT32);
  PimObjId dest2 = pimAllocAssociated(32, src1, PIM_INT32);
  PimObjId src4 = pimAllocAssociated(32, src1, PIM_INT32);  // oob check

  const std::vector<std::string> testNames = {
    "add", "sub", "mul", "div", "abs",
    "and", "or", "xor", "xnor",
    "gt", "lt", "eq",
    "min", "max",
    "popcount"
  };
  for (int i = 0; i < 15; ++i) {
    std::cout << "[INT32] Run test " << i << ": " << testNames[i] << std::endl;

    pimCopyHostToDevice((void *)vec1.data(), src1);
    pimCopyHostToDevice((void *)vec2.data(), src2);
    pimCopyHostToDevice((void *)vec3.data(), src3);
    pimCopyHostToDevice((void *)vec4.data(), src4);

    switch (i) {
    case 0: pimAdd(src1, src2, dest1); break;
    case 1: pimSub(src1, src2, dest1); break;
    case 2: pimMul(src1, src2, dest1); break;
    case 3: pimDiv(src1, src3, dest1); break;
    case 4: pimAbs(src1, dest1); break;
    case 5: pimAnd(src1, src2, dest1); break;
    case 6: pimOr(src1, src2, dest1); break;
    case 7: pimXor(src1, src2, dest1); break;
    case 8: pimXnor(src1, src2, dest1); break;
    case 9: pimGT(src1, src2, dest1); break;
    case 10: pimLT(src1, src2, dest1); break;
    case 11: pimEQ(src1, src2, dest1); break;
    case 12: pimMin(src1, src2, dest1); break;
    case 13: pimMax(src1, src2, dest1); break;
    case 14: pimPopCount(src1, dest1); break;
    default:
      assert(0);
    }

    pimResetStats();

    switch (i) {
    case 0: bitSerialIntAdd(32, src1, src2, dest2); break;
    case 1: bitSerialIntSub(32, src1, src2, dest2); break;
    case 2: bitSerialIntMul(32, src1, src2, dest2); break;
    case 3: bitSerialIntDiv(32, src1, src3, dest2); break;
    case 4: bitSerialIntAbs(32, src1, dest2); break;
    case 5: bitSerialIntAnd(32, src1, src2, dest2); break;
    case 6: bitSerialIntOr(32, src1, src2, dest2); break;
    case 7: bitSerialIntXor(32, src1, src2, dest2); break;
    case 8: bitSerialIntXnor(32, src1, src2, dest2); break;
    case 9: bitSerialIntGT(32, src1, src2, dest2); break;
    case 10: bitSerialIntLT(32, src1, src2, dest2); break;
    case 11: bitSerialIntEQ(32, src1, src2, dest2); break;
    case 12: bitSerialIntMin(32, src1, src2, dest2); break;
    case 13: bitSerialIntMax(32, src1, src2, dest2); break;
    case 14: bitSerialIntPopCount(32, src1, dest2); break;
    default:
      assert(0);
    }

    pimCopyDeviceToHost(dest1, (void *)vec5a.data());
    pimCopyDeviceToHost(dest2, (void *)vec5.data());
    if (vec5 != vec5a) {
      ok = false;
      std::cout << "[INT32] Error: Test " << i << " failed !!!!!" << std::endl;
      if (1) {
        int numFailed = 0;
        for (int i = 0; i < numElements; ++i) {
          if (vec5[i] != vec5a[i]) {
            numFailed++;
            std::cout << " Idx " << i
                      << " Operand1 " << vec1[i]
                      << " Operand2 " << vec3[i]
                      << " Result " << vec5[i]
                      << " Expected " << vec5a[i] << std::endl;
          }
        }
        std::cout << " Total " << numFailed << " out of " << numElements
                  << " failed" << std::endl;
      }
    } else {
      // For capturing PIM command stats. Do not count above two data copy
      pimShowStats();
    }

    pimCopyDeviceToHost(src1, (void *)vec1a.data());
    pimCopyDeviceToHost(src2, (void *)vec2a.data());
    pimCopyDeviceToHost(src3, (void *)vec3a.data());
    pimCopyDeviceToHost(src4, (void *)vec4a.data());
    if (vec1 != vec1a || vec2 != vec2a || vec3 != vec3a || vec4 != vec4a) {
      ok = false;
      std::cout << "[INT32] Error: Test " << i << " input modified !!!!!" << std::endl;
    }
  }

  return ok;
}

