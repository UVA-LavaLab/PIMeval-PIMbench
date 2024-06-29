// Bit-Serial Performance Modeling - BitSIMD_V_AP
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "bitSerialBitsimdAp.h"
#include <iostream>
#include <cassert>

void
bitSerialBitsimdAp::bitSerialIntAdd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpReadRowToSa(src2, i);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_R1);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

void
bitSerialBitsimdAp::bitSerialIntSub(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpReadRowToSa(src2, i);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

void
bitSerialBitsimdAp::bitSerialIntMul(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  bitSerialIntMulHelper(numBits, src1, src2, dest);
}

void
bitSerialBitsimdAp::bitSerialIntDiv(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  bitSerialIntDivRemHelper(numBits, src1, src2, dest);
}

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
bitSerialBitsimdAp::bitSerialIntAnd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    pimOpReadRowToSa(src2, i);
    pimOpAnd(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

void
bitSerialBitsimdAp::bitSerialIntOr(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  pimOpSet(src1, PIM_RREG_R2, 1);
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    pimOpReadRowToSa(src2, i);
    pimOpSel(src1, PIM_RREG_R1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

void
bitSerialBitsimdAp::bitSerialIntXor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  pimOpSet(src1, PIM_RREG_R2, 0); // XNOR with R2 to compute NOT
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    pimOpReadRowToSa(src2, i);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R2, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

void
bitSerialBitsimdAp::bitSerialIntXnor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    pimOpReadRowToSa(src2, i);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

void
bitSerialBitsimdAp::bitSerialIntGT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  // n-bit uint gt
  pimOpSet(src1, PIM_RREG_R3, 0); // XNOR with R3 to compute NOT
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits - 1; ++i) {
    pimOpReadRowToSa(src2, i);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
    pimOpReadRowToSa(src1, i);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
  // handle sign bit
  pimOpReadRowToSa(src1, numBits - 1);
  pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R2);
  pimOpReadRowToSa(src2, numBits - 1);
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
bitSerialBitsimdAp::bitSerialIntLT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  // n-bit uint lt
  pimOpSet(src1, PIM_RREG_R3, 0); // XNOR with R3 to compute NOT
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits - 1; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
    pimOpReadRowToSa(src2, i);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
  // handle sign bit
  pimOpReadRowToSa(src2, numBits - 1);
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
bitSerialBitsimdAp::bitSerialIntEQ(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  pimOpSet(src1, PIM_RREG_R2, 1);
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    pimOpReadRowToSa(src2, i);
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

void
bitSerialBitsimdAp::bitSerialIntMin(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  // n-bit int lt
  pimOpSet(src1, PIM_RREG_R3, 0); // XNOR with R3 to compute NOT
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits - 1; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
    pimOpReadRowToSa(src2, i);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
  pimOpReadRowToSa(src2, numBits - 1);
  pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R2);
  pimOpReadRowToSa(src1, numBits - 1);
  pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R2, PIM_RREG_R2);
  pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_R2);

  // if-else copy
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    pimOpReadRowToSa(src2, i);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

void
bitSerialBitsimdAp::bitSerialIntMax(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  // n-bit int gt
  pimOpSet(src1, PIM_RREG_R3, 0); // XNOR with R3 to compute NOT
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits - 1; ++i) {
    pimOpReadRowToSa(src2, i);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
    pimOpReadRowToSa(src1, i);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
  pimOpReadRowToSa(src1, numBits - 1);
  pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R2);
  pimOpReadRowToSa(src2, numBits - 1);
  pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R2, PIM_RREG_R2);
  pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_R2);

  // if-else copy
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    pimOpReadRowToSa(src2, i);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

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

void
bitSerialBitsimdAp::bitSerialUIntAdd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  bitSerialIntAdd(numBits, src1, src2, dest);
}

void
bitSerialBitsimdAp::bitSerialUIntSub(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  bitSerialIntSub(numBits, src1, src2, dest);
}

void
bitSerialBitsimdAp::bitSerialUIntMul(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  bitSerialIntMulHelper(numBits, src1, src2, dest);
}

void
bitSerialBitsimdAp::bitSerialUIntDiv(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  bitSerialUintDivRemHelper(numBits, src1, src2, dest);
}

void
bitSerialBitsimdAp::bitSerialUIntAbs(int numBits, PimObjId src, PimObjId dest)
{
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src, i);
    pimOpWriteSaToRow(dest, i);
  }
}

void
bitSerialBitsimdAp::bitSerialUIntAnd(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  bitSerialIntAnd(numBits, src1, src2, dest);
}

void
bitSerialBitsimdAp::bitSerialUIntOr(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  bitSerialIntOr(numBits, src1, src2, dest);
}

void
bitSerialBitsimdAp::bitSerialUIntXor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  bitSerialIntXor(numBits, src1, src2, dest);
}

void
bitSerialBitsimdAp::bitSerialUIntXnor(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  bitSerialIntXnor(numBits, src1, src2, dest);
}

void
bitSerialBitsimdAp::bitSerialUIntGT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  // n-bit uint gt
  pimOpSet(src1, PIM_RREG_R3, 0); // XNOR with R3 to compute NOT
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src2, i);
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

void
bitSerialBitsimdAp::bitSerialUIntLT(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  // n-bit uint lt
  pimOpSet(src1, PIM_RREG_R3, 0); // XNOR with R3 to compute NOT
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
    pimOpReadRowToSa(src2, i);
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

void
bitSerialBitsimdAp::bitSerialUIntEQ(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  bitSerialIntEQ(numBits, src1, src2, dest);
}

void
bitSerialBitsimdAp::bitSerialUIntMin(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  // n-bit uint lt
  pimOpSet(src1, PIM_RREG_R3, 0); // XNOR with R3 to compute NOT
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpXnor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
    pimOpReadRowToSa(src2, i);
    pimOpXnor(src1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_R2);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
  pimOpMove(src1, PIM_RREG_R1, PIM_RREG_R2);

  // if-else copy
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    pimOpReadRowToSa(src2, i);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

void
bitSerialBitsimdAp::bitSerialUIntMax(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  // n-bit uint gt
  pimOpSet(src1, PIM_RREG_R3, 0); // XNOR with R3 to compute NOT
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src2, i);
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
    pimOpReadRowToSa(src2, i);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

void
bitSerialBitsimdAp::bitSerialUIntPopCount(int numBits, PimObjId src, PimObjId dest)
{
  bitSerialIntPopCount(numBits, src, dest);
}

void
bitSerialBitsimdAp::bitSerialIntMulHelper(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  if (numBits > 32) return; // todo

  // cond copy the first
  pimOpReadRowToSa(src1, 0);
  pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R2);
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < numBits; ++i) {
    pimOpReadRowToSa(src2, i);
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
      pimOpReadRowToSa(src2, j);
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

void
bitSerialBitsimdAp::bitSerialUIntMulHelper(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  // TODO
}

void
bitSerialBitsimdAp::bitSerialIntDivRemHelper(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  if (numBits > 32) return; // todo

  // compute abs
  std::cout << "BS-INFO: Allocate 64 temporary rows" << std::endl;
  PimObjId abs1 = pimAllocAssociated(32, src1, PIM_INT32);
  PimObjId abs2 = pimAllocAssociated(32, src2, PIM_INT32);
  bitSerialIntAbs(numBits, src1, abs1);
  bitSerialIntAbs(numBits, src2, abs2);

  // 31-bit uint div rem
  bitSerialUintDivRemHelper(numBits - 1, abs1, abs2, dest);
  pimOpSet(src1, PIM_RREG_SA, 0);
  pimOpWriteSaToRow(dest, numBits - 1);

  // check sign
  pimOpReadRowToSa(src1, numBits - 1);
  pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
  pimOpReadRowToSa(src2, numBits - 1);
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
bitSerialBitsimdAp::bitSerialUintDivRemHelper(int numBits, PimObjId src1, PimObjId src2, PimObjId dest)
{
  if (numBits > 32) return; // todo

  // quotient and remainder
  std::cout << "BS-INFO: Allocate 96 temporary rows" << std::endl;
  PimObjId qr = pimAllocAssociated(64, src1, PIM_INT64);
  PimObjId tmp = pimAllocAssociated(32, src1, PIM_INT32);

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
      pimOpReadRowToSa(src2, j);
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

