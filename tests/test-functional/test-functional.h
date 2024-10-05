// Test: PIMeval Functional Tests
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef PIMEVAL_TEST_FUNCTIONAL_H
#define PIMEVAL_TEST_FUNCTIONAL_H

#include "libpimeval.h"
#include <random>
#include <cassert>
#include <type_traits>


//! @class  testFunctional
//! @brief  Functional test for high-level PIM APIs and data types
class testFunctional {
public:
  testFunctional()
    : m_gen(m_rd())
  {}

  bool runAllTests();

private:
  void createPimDevice();
  void deletePimDevice();

  template <typename T> std::vector<T> getRandInt(uint64_t numElements, bool allowZero = true);
  template <typename T> std::vector<T> getRandFp(uint64_t numElements, bool allowZero = true);
  template <typename T> T safeAbs(T value);

  template <typename T> bool testInt(const std::string& category, PimDataType dataType);
  template <typename T> bool testFp(const std::string& category, PimDataType dataType);

  std::random_device m_rd;
  std::mt19937 m_gen;
};

//! @brief  Generate a random vector of integer data type T
template <typename T> std::vector<T>
testFunctional::getRandInt(uint64_t numElements, bool allowZero)
{
  std::vector<T> vec(numElements);
  std::uniform_int_distribution<T> dist;
  for (uint64_t i = 0; i < numElements; ++i) {
    T val = 0;
    do {
      val = dist(m_gen);
    } while (val == 0 && !allowZero);
    vec[i] = val;
  }
  return vec;
}

//! @brief  Generate a random vector of floating-point data type T
template <typename T> std::vector<T>
testFunctional::getRandFp(uint64_t numElements, bool allowZero)
{
  std::vector<T> vec(numElements);
  std::uniform_real_distribution<T> dist;
  for (uint64_t i = 0; i < numElements; ++i) {
    T val = 0.0;
    do {
      val = dist(m_gen);
    } while (val == 0.0 && !allowZero);
    vec[i] = val;
  }
  return vec;
}

//! @brief  Handle compiler errors when calling std::abs with template types
template <typename T> T
testFunctional::safeAbs(T value)
{
  T val;
  if constexpr (std::is_integral<T>::value) {
    if constexpr (std::is_signed<T>::value) {
      val = std::abs(value);
    } else {
      val = value;
    }
  } else if constexpr (std::is_floating_point<T>::value) {
    val = std::abs(value);
  } else {
    assert(0);
  }
  return val;
}

#endif

