// Bit-Serial Performance Modeling - Base
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef BIT_SERIAL_BASE_H
#define BIT_SERIAL_BASE_H

#include "libpimsim.h"
#include <vector>

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
  virtual void bitSerialAdd(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialSub(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialMul(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialDiv(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialAbs(PimObjId src, PimObjId dest) {}
  virtual void bitSerialAnd(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialOr(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialXor(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialXnor(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialGT(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialLT(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialEQ(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialMin(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialMax(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bitSerialPopCount(PimObjId src, PimObjId dest) {}

  // helper functions
  void createDevice();
  void deleteDevice();
  std::vector<int> getRandInt(int numElements, int min, int max, bool allowZero = true);
  bool testInt32();

};

#endif
