// Bit-Serial Performance Modeling - BitSIMD_V
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef BIT_SERIAL_BITSIMD_H
#define BIT_SERIAL_BITSIMD_H

#include "bsBase.h"
#include "libpimsim.h"
#include <vector>

//! @class  bsBitsimd
//! @brief  Bit-serial perf for BitSIMD
//!         Supported micro-ops: R/W/MOVE/SET + NOT/AND/OR/XOR/SEL
class bsBitsimd : public bsBase
{
public:
  bsBitsimd() {}
  ~bsBitsimd() {}

protected:

  // virtual: create device
  virtual PimDeviceEnum getDeviceType() override { return PIM_DEVICE_BITSIMD_V; }

  // virtual: high-level APIs to evaluate
  virtual void bsInt32Add(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bsInt32Sub(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bsInt32Mul(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bsInt32Div(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bsInt32Abs(PimObjId src, PimObjId dest) override;
  virtual void bsInt32And(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bsInt32Or(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bsInt32Xor(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bsInt32Xnor(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bsInt32GT(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bsInt32LT(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bsInt32EQ(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bsInt32Min(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bsInt32Max(PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bsInt32PopCount(PimObjId src, PimObjId dest) override;

private:
  void bsInt32MulHelper3Reg(int nBit, PimObjId src1, PimObjId src2, PimObjId dest);
  void bsInt32MulHelper4Reg(int nBit, PimObjId src1, PimObjId src2, PimObjId dest);
  void bsInt32DivRemHelper(int nBit, PimObjId src1, PimObjId src2, PimObjId dest);
  void bsUint32DivRemHelper(int nBit, PimObjId src1, PimObjId src2, PimObjId dest);
};

#endif

