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
  virtual void bitSerialAdd(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialSub(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialMul(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialDiv(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialAbs(PimObjId src, PimObjId dest) override;
  virtual void bitSerialAnd(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialOr(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialXor(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialXnor(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialGT(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialLT(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialEQ(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialMin(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialMax(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialPopCount(PimObjId src, PimObjId dest) override;

private:
  void bitSerialIntMulHelper3Reg(int nBit, PimObjId src1, PimObjId src2, PimObjId dest);
  void bitSerialIntMulHelper4Reg(int nBit, PimObjId src1, PimObjId src2, PimObjId dest);
  void bitSerialIntDivRemHelper(int nBit, PimObjId src1, PimObjId src2, PimObjId dest);
  void bitSerialUintDivRemHelper(int nBit, PimObjId src1, PimObjId src2, PimObjId dest);
};

#endif

