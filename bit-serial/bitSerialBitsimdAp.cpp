// Bit-Serial Performance Modeling - BitSIMD_V_AP
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "bitSerialBitsimdAp.h"
#include <iostream>
#include <cassert>

////////////////////////////////////////////////////////////////////////////////
// INTEGER ABS
////////////////////////////////////////////////////////////////////////////////
void
bitSerialBitsimdAp::bitSerialIntAbs(int numBits, PimObjId src, PimObjId dest)
{
  pimOpReadRowToSa(src, numBits - 1);
  pimOpMove(src, PIM_RREG_SA, PIM_RREG_R1);
  pimOpMove(src, PIM_RREG_SA, PIM_RREG_R2);
  pimOpSet(src, PIM_RREG_R3, 0); // XNOR with R3 to compute NOT
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src, i);
    pimOpXnor(src, PIM_RREG_SA, PIM_RREG_R2, PIM_RREG_SA);
    pimOpXnor(src, PIM_RREG_SA, PIM_RREG_R3, PIM_RREG_SA);
    pimOpXnor(src, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    pimOpXnor(src, PIM_RREG_SA, PIM_RREG_R3, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
    pimOpXnor(src, PIM_RREG_SA, PIM_RREG_R3, PIM_RREG_SA);
    pimOpAnd(src, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
}

void
bitSerialBitsimdAp::bitSerialUIntAbs(int numBits, PimObjId src, PimObjId dest)
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
bitSerialBitsimdAp::bitSerialIntPopCount(int numBits, PimObjId src, PimObjId dest)
{
  if (numBits != 32) return; // todo

  // 2 bits -> 2-bit count
  pimOpSet(src, PIM_RREG_R3, 0); // XNOR with R3 to compute NOT
  for (int i = 0; i < 32; i += 2) {
    pimOpReadRowToSa(src, i);
    pimOpMove(src, PIM_RREG_SA, PIM_RREG_R1);
    pimOpReadRowToSa(src, i + 1);
    pimOpMove(src, PIM_RREG_SA, PIM_RREG_R2);
    pimOpXnor(src, PIM_RREG_R1, PIM_RREG_R2, PIM_RREG_SA);
    pimOpXnor(src, PIM_RREG_SA, PIM_RREG_R3, PIM_RREG_SA);
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
        pimOpXnor(src, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
        pimOpXnor(src, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
        pimOpReadRowToSa(dest, i + (1 << (iter - 1)) + j);
        pimOpSel(src, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
        pimOpXnor(src, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_SA);
        pimOpXnor(src, PIM_RREG_SA, PIM_RREG_R3, PIM_RREG_SA);
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
bitSerialBitsimdAp::bitSerialIntAdd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntAdd(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimdAp::bitSerialIntAddScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntAdd(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimdAp::implIntAdd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_R1);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

////////////////////////////////////////////////////////////////////////////////
// INTEGER SUB
////////////////////////////////////////////////////////////////////////////////
void
bitSerialBitsimdAp::bitSerialIntSub(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntSub(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimdAp::bitSerialIntSubScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntSub(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimdAp::implIntSub(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

////////////////////////////////////////////////////////////////////////////////
// INTEGER MUL
////////////////////////////////////////////////////////////////////////////////
void
bitSerialBitsimdAp::bitSerialIntMul(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntMul(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimdAp::bitSerialIntMulScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntMul(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimdAp::implIntMul(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
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
      pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
      pimOpReadRowToSa(dest, i + j);
      pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_R1);
      pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R2);
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
bitSerialBitsimdAp::bitSerialIntDiv(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntDivRem(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimdAp::bitSerialIntDivScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntDivRem(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimdAp::bitSerialUIntDiv(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implUintDivRem(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimdAp::bitSerialUIntDivScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implUintDivRem(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimdAp::implIntDivRem(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
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
  pimOpSet(src1, PIM_RREG_R3, 0); // XNOR with R3 to compute NOT
  pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  pimOpXnor(src1, PIM_RREG_R1, PIM_RREG_R3, PIM_RREG_R1);

  // sign ext
  pimOpMove(src1, PIM_RREG_R1, PIM_RREG_R2);
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(dest, i);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R2, PIM_RREG_SA);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R3, PIM_RREG_SA);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R3, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R3, PIM_RREG_SA);
    pimOpAnd(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }

  pimFree(abs1);
  pimFree(abs2);
}

void
bitSerialBitsimdAp::implUintDivRem(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
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
    pimOpSet(src1, PIM_RREG_R3, 0); // XNOR with R3 to compute NOT
    for (int j = 0; j < numBits; ++j) {
      pimOpReadRowToSa(qr, numBits - 1 - i + j);
      pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
      pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
      implReadRowOrScalar(src2, j, useScalar, scalarVal);
      pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_R1);
      pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_SA);
      pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R3, PIM_RREG_SA);
      pimOpWriteSaToRow(tmp, j);
    }

    // update quotient
    pimOpXnor(src1, PIM_RREG_R1, PIM_RREG_R3, PIM_RREG_R1);
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
bitSerialBitsimdAp::bitSerialIntAnd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntAnd(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimdAp::bitSerialIntAndScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntAnd(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimdAp::implIntAnd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
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
bitSerialBitsimdAp::bitSerialIntOr(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntOr(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimdAp::bitSerialIntOrScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntOr(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimdAp::implIntOr(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  pimOpSet(src1, PIM_RREG_R2, 1);
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpSel(src1, PIM_RREG_R1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

////////////////////////////////////////////////////////////////////////////////
// INTEGER XOR
////////////////////////////////////////////////////////////////////////////////
void
bitSerialBitsimdAp::bitSerialIntXor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntXor(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimdAp::bitSerialIntXorScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntXor(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimdAp::implIntXor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  pimOpSet(src1, PIM_RREG_R2, 0); // XNOR with R2 to compute NOT
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R2, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

////////////////////////////////////////////////////////////////////////////////
// INTEGER XNOR
////////////////////////////////////////////////////////////////////////////////
void
bitSerialBitsimdAp::bitSerialIntXnor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntXnor(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimdAp::bitSerialIntXnorScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntXnor(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimdAp::implIntXnor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

////////////////////////////////////////////////////////////////////////////////
// INTEGER GT
////////////////////////////////////////////////////////////////////////////////
void
bitSerialBitsimdAp::bitSerialIntGT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntGT(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimdAp::bitSerialIntGTScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntGT(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimdAp::implIntGT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  // n-bit uint gt
  pimOpSet(src1, PIM_RREG_R3, 0); // XNOR with R3 to compute NOT
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits - 1; ++i) {
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
    pimOpReadRowToSa(src1, i);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
  // handle sign bit
  pimOpReadRowToSa(src1, numBits - 1);
  pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R2);
  implReadRowOrScalar(src2, numBits - 1, useScalar, scalarVal);
  pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R2, PIM_RREG_R2);
  pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_SA);
  pimOpWriteSaToRow(dest, 0);

  // set other bits of dest to 0
  pimOpSet(src1, PIM_RREG_SA, 0);
  for (int i = 1; i < numBits; ++i) {
    pimOpWriteSaToRow(dest, i);
  }
}

void
bitSerialBitsimdAp::bitSerialUIntGT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implUIntGT(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimdAp::bitSerialUIntGTScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implUIntGT(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimdAp::implUIntGT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  // n-bit uint gt
  pimOpSet(src1, PIM_RREG_R3, 0); // XNOR with R3 to compute NOT
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits; ++i) {
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
    pimOpReadRowToSa(src1, i);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
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
bitSerialBitsimdAp::bitSerialIntLT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntLT(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimdAp::bitSerialIntLTScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntLT(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimdAp::implIntLT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  // n-bit uint lt
  pimOpSet(src1, PIM_RREG_R3, 0); // XNOR with R3 to compute NOT
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits - 1; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
  // handle sign bit
  implReadRowOrScalar(src2, numBits - 1, useScalar, scalarVal);
  pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R2);
  pimOpReadRowToSa(src1, numBits - 1);
  pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R2, PIM_RREG_R2);
  pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_SA);
  pimOpWriteSaToRow(dest, 0);

  // set other bits of dest to 0
  pimOpSet(src1, PIM_RREG_SA, 0);
  for (int i = 1; i < numBits; ++i) {
    pimOpWriteSaToRow(dest, i);
  }
}

void
bitSerialBitsimdAp::bitSerialUIntLT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implUIntLT(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimdAp::bitSerialUIntLTScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implUIntLT(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimdAp::implUIntLT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  // n-bit uint lt
  pimOpSet(src1, PIM_RREG_R3, 0); // XNOR with R3 to compute NOT
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
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
bitSerialBitsimdAp::bitSerialIntEQ(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntEQ(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimdAp::bitSerialIntEQScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntEQ(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimdAp::implIntEQ(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  pimOpSet(src1, PIM_RREG_R2, 1);
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
    pimOpAnd(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_R2);
  }
  pimOpMove(src1, PIM_RREG_R2, PIM_RREG_SA);
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
bitSerialBitsimdAp::bitSerialIntMin(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntMin(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimdAp::bitSerialIntMinScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntMin(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimdAp::implIntMin(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  // n-bit int lt
  pimOpSet(src1, PIM_RREG_R3, 0); // XNOR with R3 to compute NOT
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits - 1; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
  implReadRowOrScalar(src2, numBits - 1, useScalar, scalarVal);
  pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R2);
  pimOpReadRowToSa(src1, numBits - 1);
  pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R2, PIM_RREG_R2);
  pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_R2);

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
bitSerialBitsimdAp::bitSerialUIntMin(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implUIntMin(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimdAp::bitSerialUIntMinScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implUIntMin(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimdAp::implUIntMin(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  // n-bit uint lt
  pimOpSet(src1, PIM_RREG_R3, 0); // XNOR with R3 to compute NOT
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
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
bitSerialBitsimdAp::bitSerialIntMax(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implIntMax(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimdAp::bitSerialIntMaxScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implIntMax(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimdAp::implIntMax(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  // n-bit int gt
  pimOpSet(src1, PIM_RREG_R3, 0); // XNOR with R3 to compute NOT
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits - 1; ++i) {
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
    pimOpReadRowToSa(src1, i);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
  pimOpReadRowToSa(src1, numBits - 1);
  pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R2);
  implReadRowOrScalar(src2, numBits - 1, useScalar, scalarVal);
  pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R2, PIM_RREG_R2);
  pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_R2);

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
bitSerialBitsimdAp::bitSerialUIntMax(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  implUIntMax(numBits, src1, src2, dest, false, 0);
}

void
bitSerialBitsimdAp::bitSerialUIntMaxScalar(int numBits, PimObjId src1, PimObjId dest, uint64_t scalarVal)
{
  implUIntMax(numBits, src1, src1, dest, true, scalarVal);
}

void
bitSerialBitsimdAp::implUIntMax(int numBits, PimObjId src1, PimObjId src2, PimObjId dest, bool useScalar, uint64_t scalarVal)
{
  // n-bit uint gt
  pimOpSet(src1, PIM_RREG_R3, 0); // XNOR with R3 to compute NOT
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits; ++i) {
    implReadRowOrScalar(src2, i, useScalar, scalarVal);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
    pimOpReadRowToSa(src1, i);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
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
bitSerialBitsimdAp::implReadRowOrScalar(PimObjId src, unsigned bitIdx, bool useScalar, uint64_t scalarVal)
{
  if (useScalar) {
    pimOpSet(src, PIM_RREG_SA, getBit(scalarVal, bitIdx));
  } else {
    pimOpReadRowToSa(src, bitIdx);
  }
}
