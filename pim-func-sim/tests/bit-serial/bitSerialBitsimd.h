// Bit-Serial Performance Modeling - BitSIMD_V
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef BIT_SERIAL_BITSIMD_H
#define BIT_SERIAL_BITSIMD_H

#include "bitSerialBase.h"
#include "libpimsim.h"
#include <vector>

//! @class  bitSerialBitsimd
//! @brief  Bit-serial perf for BitSIMD-V
//!
//! Instruction set: read/write/move/set + NOT/AND/OR/XOR/SEL
//! Bit registers: SA, R1, R2, R3
//!
class bitSerialBitsimd : public bitSerialBase
{
public:
  bitSerialBitsimd() { m_deviceName = "bitsimd_v"; }
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
  virtual void bitSerialUIntAdd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialUIntSub(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialUIntMul(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialUIntDiv(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialUIntAbs(int numBits, PimObjId src, PimObjId dest) override;
  virtual void bitSerialUIntAnd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialUIntOr(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialUIntXor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialUIntXnor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialUIntGT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialUIntLT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialUIntEQ(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialUIntMin(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialUIntMax(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialUIntPopCount(int numBits, PimObjId src, PimObjId dest) override;

private:
  void bitSerialIntMulHelper3Reg(int numBits, PimObjId src1, PimObjId src2, PimObjId dest);
  void bitSerialIntMulHelper4Reg(int numBits, PimObjId src1, PimObjId src2, PimObjId dest);
  void bitSerialUIntMulHelper3Reg(int numBits, PimObjId src1, PimObjId src2, PimObjId dest);
  void bitSerialUIntMulHelper4Reg(int numBits, PimObjId src1, PimObjId src2, PimObjId dest);
  void bitSerialIntDivRemHelper(int numBits, PimObjId src1, PimObjId src2, PimObjId dest);
  void bitSerialUintDivRemHelper(int numBits, PimObjId src1, PimObjId src2, PimObjId dest);
};

#endif

