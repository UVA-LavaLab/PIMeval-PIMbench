// Bit-Serial Performance Modeling - BitSIMD_V
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef BIT_SERIAL_BITSIMD_H
#define BIT_SERIAL_BITSIMD_H

#include "bitSerialBase.h"
#include "libpimsim.h"
#include <vector>

//! @class  bitSerialBitsimd
//! @brief  Bit-serial perf for BitSIMD
//!         Supported micro-ops: R/W/MOVE/SET + NOT/AND/OR/XOR/SEL
class bitSerialBitsimd : public bitSerialBase
{
public:
  bitSerialBitsimd() {}
  ~bitSerialBitsimd() {}

protected:

  // virtual: create device
  virtual PimDeviceEnum getDeviceType() override { return PIM_DEVICE_BITSIMD_V; }

  // virtual: high-level APIs to evaluate
  virtual void bitSerialIntAdd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntSub(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntMul(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntDiv(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntAbs(int numBits, PimObjId src, PimObjId dest) override;
  virtual void bitSerialIntAnd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntOr(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntXor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntXnor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntGT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntLT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntEQ(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntMin(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntMax(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntPopCount(int numBits, PimObjId src, PimObjId dest) override;

private:
  void bitSerialIntMulHelper3Reg(int numBits, PimObjId src1, PimObjId src2, PimObjId dest);
  void bitSerialIntMulHelper4Reg(int numBits, PimObjId src1, PimObjId src2, PimObjId dest);
  void bitSerialIntDivRemHelper(int numBits, PimObjId src1, PimObjId src2, PimObjId dest);
  void bitSerialUintDivRemHelper(int numBits, PimObjId src1, PimObjId src2, PimObjId dest);
};

#endif

