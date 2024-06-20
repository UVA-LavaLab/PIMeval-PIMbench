// Bit-Serial Performance Modeling - Base
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef BIT_SERIAL_BASE_H
#define BIT_SERIAL_BASE_H

#include "libpimsim.h"
#include <vector>
#include <cassert>

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

  // helper functions
  void createDevice();
  void deleteDevice();
  std::vector<int> getRandInt(int numElements, int min, int max, bool allowZero = true);
  bool testInt32();

};

#endif
