// Bit-Serial Performance Modeling - BitSIMD_V
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef BIT_SERIAL_BITSIMD_H
#define BIT_SERIAL_BITSIMD_H

#include "bitSerialBase.h"
#include "libpimeval.h"
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
  virtual void bitSerialIntAbs(int numBits, PimObjId src, PimObjId dest) override;
  virtual void bitSerialIntPopCount(int numBits, PimObjId src, PimObjId dest) override;
  virtual void bitSerialIntAdd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntSub(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntMul(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntDiv(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntAnd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntOr(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntXor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntXnor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntGT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntLT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntEQ(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntMin(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntMax(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialIntAddScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal) override;
  virtual void bitSerialIntSubScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal) override;
  virtual void bitSerialIntMulScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal) override;
  virtual void bitSerialIntDivScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal) override;
  virtual void bitSerialIntAndScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal) override;
  virtual void bitSerialIntOrScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal) override;
  virtual void bitSerialIntXorScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal) override;
  virtual void bitSerialIntXnorScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal) override;
  virtual void bitSerialIntGTScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal) override;
  virtual void bitSerialIntLTScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal) override;
  virtual void bitSerialIntEQScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal) override;
  virtual void bitSerialIntMinScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal) override;
  virtual void bitSerialIntMaxScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal) override;

  virtual void bitSerialUIntAbs(int numBits, PimObjId src, PimObjId dest) override;
  virtual void bitSerialUIntDiv(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialUIntGT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialUIntLT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialUIntMin(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialUIntMax(int numBits, PimObjId src1, PimObjId src2, PimObjId dest) override;
  virtual void bitSerialUIntDivScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal) override;
  virtual void bitSerialUIntGTScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal) override;
  virtual void bitSerialUIntLTScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal) override;
  virtual void bitSerialUIntMinScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal) override;
  virtual void bitSerialUIntMaxScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal) override;

private:
  void implIntAdd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal);
  void implIntSub(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal);
  void implIntMul3Reg(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal);
  void implIntMul4Reg(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal);
  void implIntDivRem(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal);
  void implUintDivRem(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal);
  void implIntAnd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal);
  void implIntOr(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal);
  void implIntXor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal);
  void implIntXnor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal);
  void implIntGT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal);
  void implUIntGT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal);
  void implIntLT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal);
  void implUIntLT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal);
  void implIntEQ(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal);
  void implIntMin(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal);
  void implUIntMin(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal);
  void implIntMax(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal);
  void implUIntMax(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal);

  void implReadRowOrScalar(PimObjId src, unsigned bitIdx, bool useScalar = false, uint64_t scalarVal = 0);
};

#endif

