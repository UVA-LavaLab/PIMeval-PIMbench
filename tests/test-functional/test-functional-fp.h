// Test: PIMeval Functional Tests - Integer
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef PIMEVAL_TEST_FUNCTIONAL_FP_H
#define PIMEVAL_TEST_FUNCTIONAL_FP_H

#include "test-functional.h"
#include "libpimeval.h"
#include <cassert>
#include <vector>
#include <iostream>
#include <string>
#include <map>
#include <cstring>
#include <memory>
#include <limits>
#include <type_traits>
#include <cstdlib>
#include <iomanip> // used fot std::setprecision


//! @brief  Functional tests for PIM float-point operations
template <typename T> bool
testFunctional::testFp(const std::string& category, PimDataType dataType)
{
  pimResetStats();

  std::cout << "================================================================" << std::endl;
  std::cout << "INFO: PIMeval Functional Tests for " << category << std::endl;
  std::cout << "================================================================" << std::endl;

  // Number of elements to allocate in each vector
  unsigned numElements = 4000;

  // Allocate host vectors
  std::vector<T> vecSrc1 = getRandFp<T>(numElements);
  std::vector<T> vecSrc2 = getRandFp<T>(numElements);
  std::vector<T> vecSrc2nz = getRandFp<T>(numElements, false/*allowZero*/); // non-zero for div
  std::vector<T> vecDest(numElements);
  std::vector<uint8_t> vecDestBool(numElements);

  // Create a few equal cases for pimEQ
  vecSrc2[100] = vecSrc1[100];
  vecSrc2[3000] = vecSrc1[3000]; 

  // Cover scalar value testing
  // PIMeval uses uint64_t to represent bits of scalarValue
  const float scalarValFloat = 123.0f;
  uint32_t temp32;
  std::memcpy(&temp32, &scalarValFloat, sizeof(temp32));
  const uint64_t scalarVal = temp32;
  const int64_t scalarValInt = -11; // for int broadcasting
  vecSrc1[500] = static_cast<T>(scalarVal); // cover scalar EQ
  vecSrc1[501] = static_cast<T>(scalarVal - 1); // cover scalar LT

  // Pick a range for testing ranged operations
  const uint64_t idxBegin = 777;
  const uint64_t idxEnd = 3456;
  // Bit shift amount
  //const unsigned shiftAmount = 5;

  // Allocate PIM objects
  PimObjId objSrc1 = pimAlloc(PIM_ALLOC_AUTO, numElements, dataType);
  PimObjId objSrc2 = pimAllocAssociated(objSrc1, dataType);
  PimObjId objSrc2nz = pimAllocAssociated(objSrc1, dataType);
  PimObjId objDest = pimAllocAssociated(objSrc1, dataType);
  PimObjId objDestBool = pimAllocAssociated(objSrc1, PIM_BOOL);

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
      //{  5, "pimAnd"                 }, // not supported
      //{  6, "pimOr"                  }, // not supported
      //{  7, "pimXor"                 }, // not supported
      //{  8, "pimXnor"                }, // not supported
      {  9, "pimGT"                  }, 
      { 10, "pimLT"                  },
      { 11, "pimEQ"                  },
      { 12, "pimMin"                 },
      { 13, "pimMax"                 },
      { 14, "pimAddScalar"           },
      { 15, "pimSubScalar"           },
      { 16, "pimMulScalar"           },
      { 17, "pimDivScalar"           },
      //{ 18, "pimAndScalar"           }, // not supported
      //{ 19, "pimOrScalar"            }, // not supported
      //{ 20, "pimXorScalar"           }, // not supported
      //{ 21, "pimXnorScalar"          }, // not supported
      { 22, "pimGTScalar"            },
      { 23, "pimLTScalar"            },
      { 24, "pimEQScalar"            },
      { 25, "pimMinScalar"           },
      { 26, "pimMaxScalar"           },
      { 27, "pimScaledAdd"           },
      //{ 28, "pimPopCount"            }, // not supported
      //{ 29, "pimRedSumInt"           }, // not supported
      { 29, "pimRedSum"          },
      //{ 31, "pimRedSumRangedInt"     }, // not supported
      { 30, "pimRedSumRanged"    },
      //{ 33, "pimBroadcastInt"        }, // not supported
      { 32, "pimBroadcastFP32"       },
      { 33, "pimRotateElementsRight" },
      { 34, "pimRotateElementsLeft"  },
      { 35, "pimShiftElementsRight"  },
      { 36, "pimShiftElementsLeft"   },
      { 37, "pimRedMin"              },
      { 38, "pimRedMinRanged"        },
      { 39, "pimRedMax"              },
      { 40, "pimRedMaxRanged"        },
      //{ 39, "pimShiftBitsRight"      }, // not supported
      //{ 40, "pimShiftBitsLeft"       }, // not supported
      { 41, "pimCopyObjectToObject"  },
  };

  const std::unordered_set<std::string> cmpAPIs = {
    "pimGT", "pimLT", "pimEQ",
    "pimGTScalar", "pimLTScalar", "pimEQScalar",
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

    T min = std::numeric_limits<T>::max();
    T max = std::numeric_limits<T>::lowest();
    float sumFP32 = 0.0f;
    switch (testId) {
      case  0: status = pimAdd                  (objSrc1, objSrc2, objDest);            break;
      case  1: status = pimSub                  (objSrc1, objSrc2, objDest);            break;
      case  2: status = pimMul                  (objSrc1, objSrc2, objDest);            break;
      case  3: status = pimDiv                  (objSrc1, objSrc2nz, objDest);          break;
      case  4: status = pimAbs                  (objSrc1, objDest);                     break;
      // case  5: status = pimAnd                  (objSrc1, objSrc2, objDest);            break;
      // case  6: status = pimOr                   (objSrc1, objSrc2, objDest);            break;
      // case  7: status = pimXor                  (objSrc1, objSrc2, objDest);            break;
      // case  8: status = pimXnor                 (objSrc1, objSrc2, objDest);            break;
      case  9: status = pimGT                   (objSrc1, objSrc2, objDestBool);        break;
      case 10: status = pimLT                   (objSrc1, objSrc2, objDestBool);        break;
      case 11: status = pimEQ                   (objSrc1, objSrc2, objDestBool);        break;
      case 12: status = pimMin                  (objSrc1, objSrc2, objDest);            break;
      case 13: status = pimMax                  (objSrc1, objSrc2, objDest);            break;
      case 14: status = pimAddScalar            (objSrc1, objDest, scalarVal);          break;
      case 15: status = pimSubScalar            (objSrc1, objDest, scalarVal);          break;
      case 16: status = pimMulScalar            (objSrc1, objDest, scalarVal);          break;
      case 17: status = pimDivScalar            (objSrc1, objDest, scalarVal);          break;
      // case 18: status = pimAndScalar            (objSrc1, objDest, scalarVal);          break;
      // case 19: status = pimOrScalar             (objSrc1, objDest, scalarVal);          break;
      // case 20: status = pimXorScalar            (objSrc1, objDest, scalarVal);          break;
      // case 21: status = pimXnorScalar           (objSrc1, objDest, scalarVal);          break;
      case 22: status = pimGTScalar             (objSrc1, objDestBool, scalarVal);      break;
      case 23: status = pimLTScalar             (objSrc1, objDestBool, scalarVal);      break;
      case 24: status = pimEQScalar             (objSrc1, objDestBool, scalarVal);      break;
      case 25: status = pimMinScalar            (objSrc1, objDest, scalarVal);          break;
      case 26: status = pimMaxScalar            (objSrc1, objDest, scalarVal);          break;
      case 27: status = pimScaledAdd            (objSrc1, objSrc2, objDest, scalarVal); break;
      case 28: status = pimPopCount             (objSrc1, objDest);                     break;
      case 29: status = pimRedSum               (objSrc1, static_cast<void*>(&sumFP32));                    break;
      case 30: status = pimRedSum               (objSrc1, static_cast<void*>(&sumFP32), idxBegin, idxEnd);   break;
      case 31: status = pimBroadcastInt         (objDest, scalarValInt);                break;
      case 32: status = pimBroadcastFP          (objDest, scalarValFloat);            break;
      case 33: status = pimRotateElementsRight  (objDest);                              break;
      case 34: status = pimRotateElementsLeft   (objDest);                              break;
      case 35: status = pimShiftElementsRight   (objDest);                              break;
      case 36: status = pimShiftElementsLeft    (objDest);                              break;
      case 37: status = pimRedMin(objSrc1, static_cast<void*>(&min));                   break;
      case 38: status = pimRedMin(objSrc1, static_cast<void*>(&min), idxBegin, idxEnd); break;
      case 39: status = pimRedMax(objSrc1, static_cast<void*>(&max));                   break;
      case 40: status = pimRedMax(objSrc1, static_cast<void*>(&max), idxBegin, idxEnd); break;
      // case 39: status = pimShiftBitsRight       (objSrc1, objDest, shiftAmount);        break;
      // case 40: status = pimShiftBitsLeft        (objSrc1, objDest, shiftAmount);        break;
      case 41: status = pimCopyObjectToObject   (objSrc1, objDest);                     break;
      default: assert(0);
    }
    assert(status == PIM_OK);

    // Copy results from PIM to host
    // Always copy although objDest is not used for reduction sum
    if (cmpAPIs.find(testName) != cmpAPIs.end()) {
      status = pimCopyDeviceToHost(objDestBool, (void *)vecDestBool.data());
      assert(status == PIM_OK);
    } else {
      status = pimCopyDeviceToHost(objDest, (void *)vecDest.data());
      assert(status == PIM_OK);
    }

    // Skip result verification in analysis mode
    if (pimIsAnalysisMode()) {
      continue;
    }

    // Verify results
    // Validation for redMin and redMax
    if (testName == "pimRedMin" || testName == "pimRedMinRanged") {
      uint64_t begin = (testName == "pimRedMin" ? 0 : idxBegin);
      uint64_t end = (testName == "pimRedMin" ? numElements : idxEnd);
      T minExpected = vecSrc1[begin];
      for (uint64_t i = begin + 1; i < end; ++i) {
        minExpected = std::min(minExpected, vecSrc1[i]);
      }
      if (!fuzzyEqualPercent(min, minExpected)) {
        std::cout << "Large FP reduction min error: Result: " << min 
                  << " Expected: " << minExpected << std::endl;
        assert(0);
      }
      std::cout << "[PASS] " << category << " " << testName << std::endl;
    } else if (testName == "pimRedMax" || testName == "pimRedMaxRanged") {
      uint64_t begin = (testName == "pimRedMax" ? 0 : idxBegin);
      uint64_t end = (testName == "pimRedMax" ? numElements : idxEnd);
      T maxExpected = vecSrc1[begin];
      for (uint64_t i = begin + 1; i < end; ++i) {
        maxExpected = std::max(maxExpected, vecSrc1[i]);
      }
      if (!fuzzyEqualPercent(max, maxExpected)) {
        std::cout << "Large FP reduction max error: Result: " << max 
                  << " Expected: " << maxExpected << std::endl;
        assert(0);
      }
      std::cout << "[PASS] " << category << " " << testName << std::endl;
    } else if (testName == "pimRedSum" || testName == "pimRedSumRanged") {
      uint64_t begin = (testName == "pimRedSum" ? 0 : idxBegin);
      uint64_t end = (testName == "pimRedSum" ? numElements : idxEnd);
      float sumFP32Expected = 0.0f;
      for (uint64_t i = begin; i < end; ++i) {
        sumFP32Expected += vecSrc1[i];
      }
      // Allowing a small tolerance here because of multithreaded reduction in pimCmd
      if (!fuzzyEqualPercent(sumFP32, sumFP32Expected)) {
        std::cout << "Large FP reduction sum error: Result: " << sumFP32 << " Expected: " << sumFP32Expected << std::endl;
        assert(0);
      }
    } else if (cmpAPIs.find(testName) != cmpAPIs.end()) {
      int numError = 0;
      for (unsigned i = 0; i < numElements; ++i) {
        uint8_t expected = 0;
        T val;
        std::memcpy(&val, &scalarVal, sizeof(val));
        switch (testId) {
          case  9: expected = (vecSrc1[i] > vecSrc2[i] ? 1 : 0);    break; // pimGT
          case 10: expected = (vecSrc1[i] < vecSrc2[i] ? 1 : 0);    break; // pimLT
          case 11: expected = (vecSrc1[i] == vecSrc2[i] ? 1 : 0);   break; // pimEQ
          case 22: expected = (vecSrc1[i] > val ? 1 : 0);           break; // pimGTScalar
          case 23: expected = (vecSrc1[i] < val ? 1 : 0);           break; // pimLTScalar
          case 24: expected = (vecSrc1[i] == val ? 1 : 0);          break; // pimEQScalar
          default: assert(0);
        }
        if (vecDestBool[i] != expected) {
          if (numError < maxErrorToShow) {
            std::cout << "Error: Index = " << i << " Result = " << (int)vecDestBool[i] << " Expected = " << (int)expected << std::endl;
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
    } else {
      int numError = 0;
      for (unsigned i = 0; i < numElements; ++i) {
        T expected = 0.0;
        T val;
        std::memcpy(&val, &scalarVal, sizeof(val));
        T valInt = static_cast<T>(scalarValInt);
        switch (testId) {
          case  0: expected = vecSrc1[i] + vecSrc2[i];              break; // pimAdd
          case  1: expected = vecSrc1[i] - vecSrc2[i];              break; // pimSub
          case  2: expected = vecSrc1[i] * vecSrc2[i];              break; // pimMul
          case  3: expected = vecSrc1[i] / vecSrc2nz[i];            break; // pimDiv
          case  4: expected = safeAbs(vecSrc1[i]);                  break; // pimAbs
          //case  5: expected = vecSrc1[i] & vecSrc2[i];              break; // pimAnd
          //case  6: expected = vecSrc1[i] | vecSrc2[i];              break; // pimOr
          //case  7: expected = vecSrc1[i] ^ vecSrc2[i];              break; // pimXor
          //case  8: expected = ~(vecSrc1[i] ^ vecSrc2[i]);           break; // pimXnor
          case 12: expected = std::min(vecSrc1[i], vecSrc2[i]);     break; // pimMin
          case 13: expected = std::max(vecSrc1[i], vecSrc2[i]);     break; // pimMax
          case 14: expected = vecSrc1[i] + val;                     break; // pimAddScalar
          case 15: expected = vecSrc1[i] - val;                     break; // pimSubScalar
          case 16: expected = vecSrc1[i] * val;                     break; // pimMulScalar
          case 17: expected = vecSrc1[i] / val;                     break; // pimDivScalar
          //case 18: expected = vecSrc1[i] & val;                     break; // pimAndScalar
          //case 19: expected = vecSrc1[i] | val;                     break; // pimOrScalar
          //case 20: expected = vecSrc1[i] ^ val;                     break; // pimXorScalar
          //case 21: expected = ~(vecSrc1[i] ^ val);                  break; // pimXnorScalar
          case 25: expected = std::min(vecSrc1[i], val);            break; // pimMinScalar
          case 26: expected = std::max(vecSrc1[i], val);            break; // pimMaxScalar
          case 27: expected = (vecSrc1[i] * val) + vecSrc2[i];      break; // pimScaledAdd
          case 28: expected = std::bitset<sizeof(T) * 8>(vecSrc1[i]).count(); break; // pimPopCount
          case 29: assert(0); break; // pimRedSumInt
          case 30: assert(0); break; // pimRedSumFP
          case 31: expected = valInt; break; // pimBroadcastInt
          case 32: expected = val;    break; // pimBroadcastFP
          case 33: expected = (i == 0 ? vecSrc1.back() : vecSrc1[i - 1]);                break; // pimRotateElementsRight
          case 34: expected = (i == numElements - 1 ? vecSrc1.front() : vecSrc1[i + 1]); break; // pimRotateElementsLeft
          case 35: expected = (i == 0 ? 0 : vecSrc1[i - 1]);                             break; // pimShiftElementsRight
          case 36: expected = (i == numElements - 1 ? 0 : vecSrc1[i + 1]);               break; // pimShiftElementsLeft
          //case 39: expected = vecSrc1[i] >> shiftAmount; break; // pimShiftBitsRight
          //case 40: expected = vecSrc1[i] << shiftAmount; break; // pimShiftBitsLeft
          case 41: expected = vecSrc1[i];                break; // pimCopyObjectToObject 
          default: assert(0);
        }
        if (!fuzzyEqualPercent(vecDest[i], expected)) {
          if (numError < maxErrorToShow) {
          std::cout << "Error: Index = " << i << " Result = " << std::fixed << std::setprecision(12) << vecDest[i] << " Expected = " << std::fixed << std::setprecision(12) << expected << std::endl;
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
  status = pimFree(objDestBool);
  assert(status == PIM_OK);

  return true;
}

#endif

