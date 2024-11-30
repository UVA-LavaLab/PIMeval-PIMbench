// Test: PIMeval Functional Tests - Integer
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef PIMEVAL_TEST_FUNCTIONAL_INT_H
#define PIMEVAL_TEST_FUNCTIONAL_INT_H

#include "test-functional.h"
#include "libpimeval.h"
#include <cassert>
#include <vector>
#include <iostream>
#include <string>
#include <map>
#include <memory>
#include <limits>
#include <bitset>
#include <type_traits>
#include <cstdlib>


//! @brief  Functional tests for PIM integer operations
template <typename T> bool
testFunctional::testInt(const std::string& category, PimDataType dataType)
{
  pimResetStats();

  std::cout << "================================================================" << std::endl;
  std::cout << "INFO: PIMeval Functional Tests for " << category << std::endl;
  std::cout << "================================================================" << std::endl;

  // Number of elements to allocate in each vector
  unsigned numElements = 4000;

  // Allocate host vectors
  std::vector<T> vecSrc1 = getRandInt<T>(numElements);
  std::vector<T> vecSrc2 = getRandInt<T>(numElements);
  std::vector<T> vecSrc2nz = getRandInt<T>(numElements, false/*allowZero*/); // non-zero for div
  std::vector<T> vecDest(numElements);

  // Create a few equal cases for pimEQ
  vecSrc2[100] = vecSrc1[100];
  vecSrc2[3000] = vecSrc1[3000];

  // Cover scalar value testing
  // PIMeval uses uint64_t to represent bits of scalarValue
  const uint64_t scalarVal = 123;
  const int64_t scalarValInt = -11; // for int broadcasting
  vecSrc1[500] = static_cast<T>(scalarVal); // cover scalar EQ
  vecSrc1[501] = static_cast<T>(scalarVal - 1); // cover scalar LT

  // Pick a range for testing ranged operations
  const uint64_t idxBegin = 777;
  const uint64_t idxEnd = 3456;
  // Bit shift amount
  const unsigned shiftAmount = 5;

  // Allocate PIM objects
  PimObjId objSrc1 = pimAlloc(PIM_ALLOC_AUTO, numElements, dataType);
  PimObjId objSrc2 = pimAllocAssociated(objSrc1, dataType);
  PimObjId objSrc2nz = pimAllocAssociated(objSrc1, dataType);
  PimObjId objDest = pimAllocAssociated(objSrc1, dataType);

  // Copy src vectors from host to PIM
  PimStatus status = PIM_OK;
  status = pimCopyHostToDevice((void*)vecSrc1.data(), objSrc1);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void*)vecSrc2.data(), objSrc2);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void*)vecSrc2nz.data(), objSrc2nz);
  assert(status == PIM_OK);

  // Define tests : test name -> test id
  std::map<int, std::string> tests = {
      {  0, "pimAdd"                 },
      {  1, "pimSub"                 },
      {  2, "pimMul"                 },
      {  3, "pimDiv"                 },
      {  4, "pimAbs"                 },
      {  5, "pimAnd"                 },
      {  6, "pimOr"                  },
      {  7, "pimXor"                 },
      {  8, "pimXnor"                },
      {  9, "pimGT"                  },
      { 10, "pimLT"                  },
      { 11, "pimEQ"                  },
      { 12, "pimMin"                 },
      { 13, "pimMax"                 },
      { 14, "pimAddScalar"           },
      { 15, "pimSubScalar"           },
      { 16, "pimMulScalar"           },
      { 17, "pimDivScalar"           },
      { 18, "pimAndScalar"           },
      { 19, "pimOrScalar"            },
      { 20, "pimXorScalar"           },
      { 21, "pimXnorScalar"          },
      { 22, "pimGTScalar"            },
      { 23, "pimLTScalar"            },
      { 24, "pimEQScalar"            },
      { 25, "pimMinScalar"           },
      { 26, "pimMaxScalar"           },
      { 27, "pimScaledAdd"           },
      { 28, "pimPopCount"            },
      { 29, "pimRedMin"              },
      { 30, "pimRedMinRanged"        },
      { 31, "pimRedMax"              },
      { 32, "pimRedMaxRanged"        },
      { 33, "pimRedSum"              },
      { 34, "pimRedSumRanged"        },
      { 35, "pimBroadcastInt"        },
      { 36, "pimBroadcastUInt"       },
      { 37, "pimRotateElementsRight" },
      { 38, "pimRotateElementsLeft"  },
      { 39, "pimShiftElementsRight"  },
      { 40, "pimShiftElementsLeft"   },
      { 41, "pimShiftBitsRight"      },
      { 42, "pimShiftBitsLeft"       },
  };

  // Running tests
  const int testOnly = -1; // set to a testId to debug it locally. please keep default as -1
  const int maxErrorToShow = 10; // max number of errors to show when an operation failed
  for (auto& [testId, testName] : tests) {

    // Run only one testcase if specified
    if (testOnly >= 0 && testOnly != testId) {
      continue;
    }

    // For in-place element-wise shift/rotate, copy the data to objDest first
    if (testName == "pimRotateElementsRight" || testName == "pimRotateElementsLeft" ||
        testName == "pimShiftElementsRight" || testName == "pimShiftElementsLeft") {
      status = pimCopyDeviceToDevice(objSrc1, objDest);
      assert(status == PIM_OK);
    }

    int64_t sumInt = 0;
    T min = std::numeric_limits<T>::max(), max = std::numeric_limits<T>::lowest();
    switch (testId) {
      case  0: status = pimAdd                  (objSrc1, objSrc2, objDest);            break;
      case  1: status = pimSub                  (objSrc1, objSrc2, objDest);            break;
      case  2: status = pimMul                  (objSrc1, objSrc2, objDest);            break;
      case  3: status = pimDiv                  (objSrc1, objSrc2nz, objDest);          break;
      case  4: status = pimAbs                  (objSrc1, objDest);                     break;
      case  5: status = pimAnd                  (objSrc1, objSrc2, objDest);            break;
      case  6: status = pimOr                   (objSrc1, objSrc2, objDest);            break;
      case  7: status = pimXor                  (objSrc1, objSrc2, objDest);            break;
      case  8: status = pimXnor                 (objSrc1, objSrc2, objDest);            break;
      case  9: status = pimGT                   (objSrc1, objSrc2, objDest);            break;
      case 10: status = pimLT                   (objSrc1, objSrc2, objDest);            break;
      case 11: status = pimEQ                   (objSrc1, objSrc2, objDest);            break;
      case 12: status = pimMin                  (objSrc1, objSrc2, objDest);            break;
      case 13: status = pimMax                  (objSrc1, objSrc2, objDest);            break;
      case 14: status = pimAddScalar            (objSrc1, objDest, scalarVal);          break;
      case 15: status = pimSubScalar            (objSrc1, objDest, scalarVal);          break;
      case 16: status = pimMulScalar            (objSrc1, objDest, scalarVal);          break;
      case 17: status = pimDivScalar            (objSrc1, objDest, scalarVal);          break;
      case 18: status = pimAndScalar            (objSrc1, objDest, scalarVal);          break;
      case 19: status = pimOrScalar             (objSrc1, objDest, scalarVal);          break;
      case 20: status = pimXorScalar            (objSrc1, objDest, scalarVal);          break;
      case 21: status = pimXnorScalar           (objSrc1, objDest, scalarVal);          break;
      case 22: status = pimGTScalar             (objSrc1, objDest, scalarVal);          break;
      case 23: status = pimLTScalar             (objSrc1, objDest, scalarVal);          break;
      case 24: status = pimEQScalar             (objSrc1, objDest, scalarVal);          break;
      case 25: status = pimMinScalar            (objSrc1, objDest, scalarVal);          break;
      case 26: status = pimMaxScalar            (objSrc1, objDest, scalarVal);          break;
      case 27: status = pimScaledAdd            (objSrc1, objSrc2, objDest, scalarVal); break;
      case 28: status = pimPopCount             (objSrc1, objDest);                     break;
      case 29: status = pimRedMin               (objSrc1, static_cast<void*>(&min));    break;
      case 30: status = pimRedMin               (objSrc1, static_cast<void*>(&min), idxBegin, idxEnd);      break;
      case 31: status = pimRedMax               (objSrc1, static_cast<void*>(&max));                     break;
      case 32: status = pimRedMax               (objSrc1, static_cast<void*>(&max), idxBegin, idxEnd);                    break;   
      case 33: status = pimRedSum               (objSrc1, static_cast<void*>(&sumInt));                     break;
      case 34: status = pimRedSum               (objSrc1, static_cast<void*>(&sumInt), idxBegin, idxEnd);            break;
      case 35: status = pimBroadcastInt         (objDest, scalarValInt);                break;
      case 36: status = pimBroadcastUInt        (objDest, scalarVal);                   break;
      case 37: status = pimRotateElementsRight  (objDest);                              break;
      case 38: status = pimRotateElementsLeft   (objDest);                              break;
      case 39: status = pimShiftElementsRight   (objDest);                              break;
      case 40: status = pimShiftElementsLeft    (objDest);                              break;
      case 41: status = pimShiftBitsRight       (objSrc1, objDest, shiftAmount);        break;
      case 42: status = pimShiftBitsLeft        (objSrc1, objDest, shiftAmount);        break;
      default: assert(0);
    }
    assert(status == PIM_OK);

    // Copy results from PIM to host
    // Always copy although objDest is not used for reduction sum
    status = pimCopyDeviceToHost(objDest, (void *)vecDest.data());
    assert(status == PIM_OK);

    // Skip result verification in analysis mode
    if (pimIsAnalysisMode()) {
      continue;
    }

    // Verify results
    if (testName == "pimRedMin" || testName == "pimRedMinRanged") {
      uint64_t begin = (testName == "pimRedMin" ? 0 : idxBegin);
      uint64_t end = (testName == "pimRedMin" ? numElements : idxEnd);
      T sumIntExpected = vecSrc1[begin];
      for (uint64_t i = begin+1; i < end; ++i) {
        sumIntExpected = sumIntExpected > vecSrc1[i] ? vecSrc1[i] : sumIntExpected;
      }
      assert(min == sumIntExpected);
      std::cout << "[PASS] " << category << " " << testName << std::endl;
    } else if (testName == "pimRedMax" || testName == "pimRedMaxRanged") {
      uint64_t begin = (testName == "pimRedMax" ? 0 : idxBegin);
      uint64_t end = (testName == "pimRedMax" ? numElements : idxEnd);
      T sumIntExpected = vecSrc1[begin];
      for (uint64_t i = begin + 1; i < end; ++i) {
        sumIntExpected = sumIntExpected < vecSrc1[i] ? vecSrc1[i] : sumIntExpected;
      }
      assert(max == sumIntExpected);
      std::cout << "[PASS] " << category << " " << testName << std::endl;
    } else if (testName == "pimRedSum" || testName == "pimRedSumRanged") {
      uint64_t begin = (testName == "pimRedSum" ? 0 : idxBegin);
      uint64_t end = (testName == "pimRedSum" ? numElements : idxEnd);
      int64_t sumIntExpected = 0;
      for (uint64_t i = begin; i < end; ++i) {
        sumIntExpected += vecSrc1[i];
      }
      assert(sumInt == sumIntExpected);
      std::cout << "[PASS] " << category << " " << testName << std::endl;
    } else {
      int numError = 0;
      for (unsigned i = 0; i < numElements; ++i) {
        T expected = 0;
        T val = static_cast<T>(scalarVal);
        T valInt = static_cast<T>(scalarValInt);
        switch (testId) {
          case  0: expected = vecSrc1[i] + vecSrc2[i];              break; // pimAdd
          case  1: expected = vecSrc1[i] - vecSrc2[i];              break; // pimSub
          case  2: expected = vecSrc1[i] * vecSrc2[i];              break; // pimMul
          case  3: expected = vecSrc1[i] / vecSrc2nz[i];            break; // pimDiv
          case  4: expected = safeAbs(vecSrc1[i]);                  break; // pimAbs
          case  5: expected = vecSrc1[i] & vecSrc2[i];              break; // pimAnd
          case  6: expected = vecSrc1[i] | vecSrc2[i];              break; // pimOr
          case  7: expected = vecSrc1[i] ^ vecSrc2[i];              break; // pimXor
          case  8: expected = ~(vecSrc1[i] ^ vecSrc2[i]);           break; // pimXnor
          case  9: expected = (vecSrc1[i] > vecSrc2[i] ? 1 : 0);    break; // pimGT
          case 10: expected = (vecSrc1[i] < vecSrc2[i] ? 1 : 0);    break; // pimLT
          case 11: expected = (vecSrc1[i] == vecSrc2[i] ? 1 : 0);   break; // pimEQ
          case 12: expected = std::min(vecSrc1[i], vecSrc2[i]);     break; // pimMin
          case 13: expected = std::max(vecSrc1[i], vecSrc2[i]);     break; // pimMax
          case 14: expected = vecSrc1[i] + val;                     break; // pimAddScalar
          case 15: expected = vecSrc1[i] - val;                     break; // pimSubScalar
          case 16: expected = vecSrc1[i] * val;                     break; // pimMulScalar
          case 17: expected = vecSrc1[i] / val;                     break; // pimDivScalar
          case 18: expected = vecSrc1[i] & val;                     break; // pimAndScalar
          case 19: expected = vecSrc1[i] | val;                     break; // pimOrScalar
          case 20: expected = vecSrc1[i] ^ val;                     break; // pimXorScalar
          case 21: expected = ~(vecSrc1[i] ^ val);                  break; // pimXnorScalar
          case 22: expected = (vecSrc1[i] > val ? 1 : 0);           break; // pimGTScalar
          case 23: expected = (vecSrc1[i] < val ? 1 : 0);           break; // pimLTScalar
          case 24: expected = (vecSrc1[i] == val ? 1 : 0);          break; // pimEQScalar
          case 25: expected = std::min(vecSrc1[i], val);            break; // pimMinScalar
          case 26: expected = std::max(vecSrc1[i], val);            break; // pimMaxScalar
          case 27: expected = (vecSrc1[i] * val) + vecSrc2[i];      break; // pimScaledAdd
          case 28: expected = std::bitset<sizeof(T) * 8>(vecSrc1[i]).count(); break; // pimPopCount
          case 29: assert(0); break; // pimRedSumInt
          case 30: assert(0); break; // pimRedSumUInt
          case 31: assert(0); break; // pimRedSumRangedInt
          case 32: assert(0); break; // pimRedSumRangedUInt
          case 33: assert(0); break; // pimRedSumInt
          case 34: assert(0); break; // pimRedSumUInt
          case 35: expected = valInt; break; // pimBroadcastInt
          case 36: expected = val;    break; // pimBroadcastUInt
          case 37: expected = (i == 0 ? vecSrc1.back() : vecSrc1[i - 1]);                break; // pimRotateElementsRight
          case 38: expected = (i == numElements - 1 ? vecSrc1.front() : vecSrc1[i + 1]); break; // pimRotateElementsLeft
          case 39: expected = (i == 0 ? 0 : vecSrc1[i - 1]);                             break; // pimShiftElementsRight
          case 40: expected = (i == numElements - 1 ? 0 : vecSrc1[i + 1]);               break; // pimShiftElementsLeft
          case 41: expected = vecSrc1[i] >> shiftAmount; break; // pimShiftBitsRight
          case 42: expected = vecSrc1[i] << shiftAmount; break; // pimShiftBitsLeft
          default: assert(0);
        }
        if (vecDest[i] != expected) {
          if (numError < maxErrorToShow) {
          std::cout << "Error: Index = " << i << " Result = " << vecDest[i] << " Expected = " << expected << std::endl;
          }
          ++numError;
        }
      }
      if (numError > 0) {
        std::cout << "[FAIL] " << category << " " << testName << " -- #error = " << numError << std::endl;
        assert(numError == 0);
      } else {
        std::cout << "[PASS] " << category << " " << testName << std::endl;
      }
    }
  }

  pimShowStats();

  status = pimFree(objSrc1);
  assert(status == PIM_OK);
  status = pimFree(objSrc2);
  assert(status == PIM_OK);
  status = pimFree(objSrc2nz);
  assert(status == PIM_OK);
  status = pimFree(objDest);
  assert(status == PIM_OK);

  return true;
}

#endif

