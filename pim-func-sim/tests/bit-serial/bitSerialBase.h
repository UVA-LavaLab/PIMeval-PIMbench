// Bit-Serial Performance Modeling - Base
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef BIT_SERIAL_BASE_H
#define BIT_SERIAL_BASE_H

#include "libpimsim.h"
#include <cassert>
#include <vector>
#include <random>
#include <iostream>
#include <string>

//! @class  bitSerialBase
//! @brief  Bit-serial perf base class
class bitSerialBase
{
public:
  bitSerialBase() {}
  virtual ~bitSerialBase() {}

  bool runTests();

protected:

  // virtual: get device type
  virtual PimDeviceEnum getDeviceType() = 0;

  // virtual: high-level APIs to evaluate
  virtual void bitSerialIntAdd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialIntSub(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialIntMul(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialIntDiv(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialIntAbs(int numBits, PimObjId src, PimObjId dest) {}
  virtual void bitSerialIntAnd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialIntOr(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialIntXor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialIntXnor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialIntGT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialIntLT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialIntEQ(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialIntMin(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialIntMax(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialIntPopCount(int numBits, PimObjId src, PimObjId dest) {}
  virtual void bitSerialUIntAdd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialUIntSub(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialUIntMul(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialUIntDiv(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialUIntAbs(int numBits, PimObjId src, PimObjId dest) {}
  virtual void bitSerialUIntAnd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialUIntOr(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialUIntXor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialUIntXnor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialUIntGT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialUIntLT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialUIntEQ(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialUIntMin(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialUIntMax(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialUIntPopCount(int numBits, PimObjId src, PimObjId dest) {}

  // helper functions
  void createDevice();
  void deleteDevice();
  //! @brief  Generate random int
  template <typename T> std::vector<T> getRandInt(int numElements, T min, T max, bool allowZero = true) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<T> dis(min, max);
  std::vector<T> vec(numElements);
  for (int i = 0; i < numElements; ++i) {
    T val = 0;
    do {
      val = dis(gen);
    } while (val == 0 && !allowZero);
    vec[i] = val;
  }
  return vec;
  }
  bool testInt8();
  bool testInt16();
  bool testInt32();
  bool testInt64();
  bool testUInt8();
  bool testUInt16();
  bool testUInt32();
  bool testUInt64();

};

#endif
