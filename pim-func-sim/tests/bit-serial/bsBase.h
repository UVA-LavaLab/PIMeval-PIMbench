// Bit-Serial Performance Modeling - Base
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef BIT_SERIAL_BASE_H
#define BIT_SERIAL_BASE_H

#include "libpimsim.h"
#include <vector>

//! @class  bsBase
//! @brief  Bit-serial perf base class
class bsBase
{
public:
  bsBase() {}
  virtual ~bsBase() {}

  bool runTests();

protected:

  // virtual: get device type
  virtual PimDeviceEnum getDeviceType() = 0;

  // virtual: high-level APIs to evaluate
  virtual void bsInt32Add(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bsInt32Sub(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bsInt32Mul(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bsInt32Div(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bsInt32Abs(PimObjId src, PimObjId dest) {}
  virtual void bsInt32And(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bsInt32Or(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bsInt32Xor(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bsInt32Xnor(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bsInt32GT(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bsInt32LT(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bsInt32EQ(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bsInt32Min(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bsInt32Max(PimObjId src1, PimObjId src2, PimObjId dest) {}
  virtual void bsInt32PopCount(PimObjId src, PimObjId dest) {}

  // helper functions
  void bsCreateDevice();
  void bsDeleteDevice();
  std::vector<int> getRandInt(int numElements, int min, int max, bool allowZero = true);
  bool testInt32();

};

#endif
