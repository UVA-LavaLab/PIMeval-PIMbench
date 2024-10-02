// Bit-Serial Performance Modeling - BitSIMD_V
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "bitSerialBitsimd.h"
#include <iostream>
#include <cassert>

////////////////////////////////////////////////////////////////////////////////
// INTEGER ABS
////////////////////////////////////////////////////////////////////////////////
void
bitSerialBitsimd::bitSerialIntAbs(int numBits, PimObjId src, PimObjId dest)
{
  pimOpReadRowToSa(src, numBits - 1);
  pimOpMove(src, PIM_RREG_SA, PIM_RREG_R1);
  pimOpMove(src, PIM_RREG_SA, PIM_RREG_R2);
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src, i);
    pimOpXor(src, PIM_RREG_SA, PIM_RREG_R2, PIM_RREG_SA);
    pimOpXor(src, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
    pimOpNot(src, PIM_RREG_SA, PIM_RREG_SA);
    pimOpAnd(src, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
}

void
bitSerialBitsimd::bitSerialUIntAbs(int numBits, PimObjId src, PimObjId dest)
{
  // same as copy
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src, i);
    pimOpWriteSaToRow(dest, i);
  }
}

////////////////////////////////////////////////////////////////////////////////
// INTEGER POPCOUNT
////////////////////////////////////////////////////////////////////////////////

void
bitSerialBitsimd::bitSerialIntPopCount(int numBits, PimObjId src, PimObjId dest)
{
  if (numBits != 32) return; // todo

  // 2 bits -> 2-bit count
  for (int i = 0; i < 32; i += 2) {
    pimOpReadRowToSa(src, i);
    pimOpMove(src, PIM_RREG_SA, PIM_RREG_R1);
    pimOpReadRowToSa(src, i + 1);
    pimOpMove(src, PIM_RREG_SA, PIM_RREG_R2);
    pimOpXor(src, PIM_RREG_R1, PIM_RREG_R2, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
    pimOpAnd(src, PIM_RREG_R1, PIM_RREG_R2, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i + 1);
  }

  // aggregate from 2-bit count to 6-bit count
  for (int iter = 2; iter <= 5; ++iter) {
    for (int i = 0; i < 32; i += (1 << iter)) {
      pimOpSet(src, PIM_RREG_R1, 0);
      for (int j = 0; j < iter; ++j) {
        pimOpReadRowToSa(dest, i + j);
        pimOpXor(src, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
        pimOpReadRowToSa(dest, i + (1 << (iter - 1)) + j);
        pimOpSel(src, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
        pimOpXor(src, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_SA);
        pimOpWriteSaToRow(dest, i + j);
      }
      pimOpMove(src, PIM_RREG_R1, PIM_RREG_SA);
      pimOpWriteSaToRow(dest, i + iter);
    }
  }

  // set other bits to 0
  pimOpSet(src, PIM_RREG_SA, 0);
  for (int i = 6; i < numBits; ++i) {
    pimOpWriteSaToRow(dest, i);
  }
}

////////////////////////////////////////////////////////////////////////////////
// INTEGER ADD
////////////////////////////////////////////////////////////////////////////////
void
bitSerialBitsimd::bitSerialIntAdd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntAdd(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimd::bitSerialIntAddScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntAdd(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimd::implIntAdd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
    pimOpXor(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

////////////////////////////////////////////////////////////////////////////////
// INTEGER SUB
////////////////////////////////////////////////////////////////////////////////
void
bitSerialBitsimd::bitSerialIntSub(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntSub(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimd::bitSerialIntSubScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntSub(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimd::implIntSub(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_R1);
    pimOpXor(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

////////////////////////////////////////////////////////////////////////////////
// INTEGER MUL
////////////////////////////////////////////////////////////////////////////////
void
bitSerialBitsimd::bitSerialIntMul(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntMul4Reg(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimd::bitSerialIntMulScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntMul4Reg(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimd::implIntMul3Reg(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  if (numBits > 32) return; // todo

  std::cout << "BS-INFO: Allocate 32 temporary rows" << std::endl;
  PimObjId tmp = pimAllocAssociated(src1, PIM_INT32);

  // cond copy the first
  pimOpReadRowToSa(src1, 0);
  pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R2);
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits; ++i) {
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }

  // add and cond write
  for (int i = 1; i < numBits; ++i) {
    pimOpSet(src1, PIM_RREG_R1, 0);
    for (int j = 0; i + j < numBits; ++j) {
      pimOpReadRowToSa(dest, i + j);
      pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
      implReadRowOrScalar(src2, j, useScalar, scalarVal);
      pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
      pimOpXor(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_SA);
      pimOpWriteSaToRow(tmp, j);
    }
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R2);
    for (int j = 0; i + j < numBits; ++j) {
      pimOpReadRowToSa(dest, i + j);
      pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
      pimOpReadRowToSa(tmp, j);
      pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
      pimOpWriteSaToRow(dest, i + j);
    }
  }

  pimFree(tmp);
}

void
bitSerialBitsimd::implIntMul4Reg(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  if (numBits > 32) return; // todo

  // cond copy the first
  pimOpReadRowToSa(src1, 0);
  pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R2);
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits; ++i) {
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }

  // add and cond write
  for (int i = 1; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R3); // cond

    pimOpSet(src1, PIM_RREG_R1, 0); // carry
    for (int j = 0; i + j < numBits; ++j) {
      // add
      implReadRowOrScalar(src2, j, useScalar, scalarVal);
      pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
      pimOpReadRowToSa(dest, i + j);
      pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
      pimOpXor(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R2);
      // cond write
      pimOpSel(src1, PIM_RREG_R3, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_SA);
      pimOpWriteSaToRow(dest, i + j);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// INTEGER DIV
////////////////////////////////////////////////////////////////////////////////
void
bitSerialBitsimd::bitSerialIntDiv(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntDivRem(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimd::bitSerialIntDivScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntDivRem(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimd::bitSerialUIntDiv(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implUintDivRem(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimd::bitSerialUIntDivScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implUintDivRem(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimd::implIntDivRem(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  if (numBits > 32) return; // todo

  // compute abs
  std::cout << "BS-INFO: Allocate 64 temporary rows" << std::endl;
  PimObjId abs1 = pimAllocAssociated(src1, PIM_INT32);
  PimObjId abs2 = pimAllocAssociated(src1, PIM_INT32);
  bitSerialIntAbs(numBits, src1, abs1);
  if (useScalar) {
    // broadcast the scalar value for computing abs
    for (int i = 0; i < numBits; ++i) {
      pimOpSet(src1, PIM_RREG_SA, getBit(scalarVal, i));
      pimOpWriteSaToRow(abs2, i);
    }
    bitSerialIntAbs(numBits, abs2, abs2);
  } else {
    bitSerialIntAbs(numBits, src2, abs2);
  }

  // 31-bit uint div rem
  implUintDivRem(numBits - 1, abs1, abs2, dest, useScalar, scalarVal);
  pimOpSet(src1, PIM_RREG_SA, 0);
  pimOpWriteSaToRow(dest, numBits - 1);

  // check sign
  pimOpReadRowToSa(src1, numBits - 1);
  pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
  implReadRowOrScalar(src2, numBits - 1, useScalar, scalarVal);
  pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);

  // sign ext
  pimOpMove(src1, PIM_RREG_R1, PIM_RREG_R2);
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(dest, i);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R2, PIM_RREG_SA);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
    pimOpNot(src1, PIM_RREG_SA, PIM_RREG_SA);
    pimOpAnd(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }

  pimFree(abs1);
  pimFree(abs2);
}

void
bitSerialBitsimd::implUintDivRem(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  if (numBits > 32) return; // todo

  // quotient and remainder
  std::cout << "BS-INFO: Allocate 96 temporary rows" << std::endl;
  PimObjId qr = pimAllocAssociated(src1, PIM_INT64);
  PimObjId tmp = pimAllocAssociated(src1, PIM_INT32);

  // init 64-bit space
  pimOpSet(src1, PIM_RREG_SA, 0);
  for (int i = 0; i < numBits * 2; ++i) {
    pimOpWriteSaToRow(qr, i);
  }

  // compute
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, numBits - 1 - i);
    pimOpWriteSaToRow(qr, numBits - 1 - i);

    // qr[62-i:31-i] - b
    pimOpSet(src1, PIM_RREG_R1, 0);
    for (int j = 0; j < numBits; ++j) {
      pimOpReadRowToSa(qr, numBits - 1 - i + j);
      pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
      implReadRowOrScalar(src2, j, useScalar, scalarVal);
      pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_R1);
      pimOpXor(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_SA);
      pimOpWriteSaToRow(tmp, j);
    }

    // update quotient
    pimOpNot(src1, PIM_RREG_R1, PIM_RREG_R1);
    pimOpMove(src1, PIM_RREG_R1, PIM_RREG_SA);
    pimOpWriteSaToRow(qr, numBits * 2 - 1 - i);

    // update remainder
    pimOpMove(src1, PIM_RREG_R1, PIM_RREG_R2);
    for (int j = 0; j <= i; ++j) {
      pimOpReadRowToSa(tmp, j);
      pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
      pimOpReadRowToSa(qr, numBits - 1 - i + j);
      pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_SA);
      pimOpWriteSaToRow(qr, numBits - 1 - i + j);
    }
  }

  // copy results
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(qr, i + numBits);
    pimOpWriteSaToRow(dest, i);
  }

  pimFree(qr);
  pimFree(tmp);
}

////////////////////////////////////////////////////////////////////////////////
// INTEGER AND
////////////////////////////////////////////////////////////////////////////////
void
bitSerialBitsimd::bitSerialIntAnd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntAnd(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimd::bitSerialIntAndScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntAnd(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimd::implIntAnd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpAnd(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

////////////////////////////////////////////////////////////////////////////////
// INTEGER OR
////////////////////////////////////////////////////////////////////////////////
void
bitSerialBitsimd::bitSerialIntOr(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntOr(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimd::bitSerialIntOrScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntOr(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimd::implIntOr(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpOr(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

////////////////////////////////////////////////////////////////////////////////
// INTEGER XOR
////////////////////////////////////////////////////////////////////////////////
void
bitSerialBitsimd::bitSerialIntXor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntXor(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimd::bitSerialIntXorScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntXor(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimd::implIntXor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

////////////////////////////////////////////////////////////////////////////////
// INTEGER XNOR
////////////////////////////////////////////////////////////////////////////////
void
bitSerialBitsimd::bitSerialIntXnor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntXnor(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimd::bitSerialIntXnorScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntXnor(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimd::implIntXnor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    pimOpNot(src1, PIM_RREG_SA, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

////////////////////////////////////////////////////////////////////////////////
// INTEGER GT
////////////////////////////////////////////////////////////////////////////////
void
bitSerialBitsimd::bitSerialIntGT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntGT(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimd::bitSerialIntGTScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntGT(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimd::implIntGT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  // n-1-bit uint gt
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits - 1; ++i) {
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpReadRowToSa(src1, i);
    pimOpNot(src1, PIM_RREG_R2, PIM_RREG_R2);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
  // handle sign bit
  pimOpReadRowToSa(src1, numBits - 1);
  pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R2);
  implReadRowOrScalar(src2, numBits - 1, useScalar, scalarVal);
  pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R2, PIM_RREG_R2);
  pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
  pimOpWriteSaToRow(dest, 0);

  // set other bits of dest to 0
  pimOpSet(src1, PIM_RREG_SA, 0);
  for (int i = 1; i < numBits; ++i) {
    pimOpWriteSaToRow(dest, i);
  }
}

void
bitSerialBitsimd::bitSerialUIntGT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implUIntGT(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimd::bitSerialUIntGTScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implUIntGT(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimd::implUIntGT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  // n-bit uint gt
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits; ++i) {
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpReadRowToSa(src1, i);
    pimOpNot(src1, PIM_RREG_R2, PIM_RREG_R2);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
  pimOpMove(src1, PIM_RREG_R1, PIM_RREG_SA);
  pimOpWriteSaToRow(dest, 0);

  // set other bits of dest to 0
  pimOpSet(src1, PIM_RREG_SA, 0);
  for (int i = 1; i < numBits; ++i) {
    pimOpWriteSaToRow(dest, i);
  }
}

////////////////////////////////////////////////////////////////////////////////
// INTEGER LT
////////////////////////////////////////////////////////////////////////////////
void
bitSerialBitsimd::bitSerialIntLT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntLT(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimd::bitSerialIntLTScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntLT(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimd::implIntLT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  // n-1-bit uint lt
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits - 1; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpNot(src1, PIM_RREG_R2, PIM_RREG_R2);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
  // handle sign bit
  implReadRowOrScalar(src2, numBits - 1, useScalar, scalarVal);
  pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R2);
  pimOpReadRowToSa(src1, numBits - 1);
  pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R2, PIM_RREG_R2);
  pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
  pimOpWriteSaToRow(dest, 0);

  // set other bits of dest to 0
  pimOpSet(src1, PIM_RREG_SA, 0);
  for (int i = 1; i < numBits; ++i) {
    pimOpWriteSaToRow(dest, i);
  }
}

void
bitSerialBitsimd::bitSerialUIntLT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implUIntLT(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimd::bitSerialUIntLTScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implUIntLT(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimd::implUIntLT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  // n-bit uint lt
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpNot(src1, PIM_RREG_R2, PIM_RREG_R2);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
  pimOpMove(src1, PIM_RREG_R1, PIM_RREG_SA);
  pimOpWriteSaToRow(dest, 0);

  // set other bits of dest to 0
  pimOpSet(src1, PIM_RREG_SA, 0);
  for (int i = 1; i < numBits; ++i) {
    pimOpWriteSaToRow(dest, i);
  }
}

////////////////////////////////////////////////////////////////////////////////
// INTEGER EQ
////////////////////////////////////////////////////////////////////////////////
void
bitSerialBitsimd::bitSerialIntEQ(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntEQ(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimd::bitSerialIntEQScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntEQ(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimd::implIntEQ(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  pimOpSet(src1, PIM_RREG_R2, 0);
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
    pimOpOr(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_R2);
  }
  pimOpNot(src1, PIM_RREG_R2, PIM_RREG_SA);
  pimOpWriteSaToRow(dest, 0);

  // set other bits of dest to 0
  pimOpSet(src1, PIM_RREG_SA, 0);
  for (int i = 1; i < numBits; ++i) {
    pimOpWriteSaToRow(dest, i);
  }
}

////////////////////////////////////////////////////////////////////////////////
// INTEGER MIN
////////////////////////////////////////////////////////////////////////////////
void
bitSerialBitsimd::bitSerialIntMin(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntMin(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimd::bitSerialIntMinScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntMin(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimd::implIntMin(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  // n-bit int lt
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits - 1; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpNot(src1, PIM_RREG_R2, PIM_RREG_R2);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
  implReadRowOrScalar(src2, numBits - 1, useScalar, scalarVal);
  pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R2);
  pimOpReadRowToSa(src1, numBits - 1);
  pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R2, PIM_RREG_R2);
  pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);

  // if-else copy
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

void
bitSerialBitsimd::bitSerialUIntMin(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implUIntMin(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimd::bitSerialUIntMinScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implUIntMin(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimd::implUIntMin(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  // n-bit uint lt
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpNot(src1, PIM_RREG_R2, PIM_RREG_R2);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
  pimOpMove(src1, PIM_RREG_R1, PIM_RREG_R2);

  // if-else copy
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

////////////////////////////////////////////////////////////////////////////////
// INTEGER MAX
////////////////////////////////////////////////////////////////////////////////
void
bitSerialBitsimd::bitSerialIntMax(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntMax(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimd::bitSerialIntMaxScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntMax(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimd::implIntMax(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  // n-bit int gt
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits - 1; ++i) {
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpReadRowToSa(src1, i);
    pimOpNot(src1, PIM_RREG_R2, PIM_RREG_R2);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
  pimOpReadRowToSa(src1, numBits - 1);
  pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R2);
  implReadRowOrScalar(src2, numBits - 1, useScalar, scalarVal);
  pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R2, PIM_RREG_R2);
  pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);

  // if-else copy
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

void
bitSerialBitsimd::bitSerialUIntMax(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implUIntMax(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimd::bitSerialUIntMaxScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implUIntMax(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimd::implUIntMax(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{  // n-bit uint gt
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits; ++i) {
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpReadRowToSa(src1, i);
    pimOpNot(src1, PIM_RREG_R2, PIM_RREG_R2);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
  pimOpMove(src1, PIM_RREG_R1, PIM_RREG_R2);

  // if-else copy
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

////////////////////////////////////////////////////////////////////////////////
// HELPER
////////////////////////////////////////////////////////////////////////////////
void
bitSerialBitsimd::implReadRowOrScalar(PimObjId src, unsigned bitIdx, bool useScalar, uint64_t scalarVal)
{
  if (useScalar) {
    pimOpSet(src, PIM_RREG_SA, getBit(scalarVal, bitIdx));
  } else {
    pimOpReadRowToSa(src, bitIdx);
  }
}

