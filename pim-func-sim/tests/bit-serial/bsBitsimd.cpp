// Bit-Serial Performance Modeling - BitSIMD_V
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "bsBitsimd.h"
#include <iostream>

void
bsBitsimd::bsInt32Add(PimObjId src1, PimObjId src2, PimObjId dest)
{
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < 32; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpReadRowToSa(src2, i);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
    pimOpXor(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

void
bsBitsimd::bsInt32Sub(PimObjId src1, PimObjId src2, PimObjId dest)
{
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < 32; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpReadRowToSa(src2, i);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_R1);
    pimOpXor(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

void
bsBitsimd::bsInt32Mul(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bsInt32MulHelper4Reg(32, src1, src2, dest);
}

void
bsBitsimd::bsInt32Div(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bsInt32DivRemHelper(32, src1, src2, dest);
}

void
bsBitsimd::bsInt32Abs(PimObjId src, PimObjId dest)
{
  pimOpReadRowToSa(src, 31);
  pimOpMove(src, PIM_RREG_SA, PIM_RREG_R1);
  pimOpMove(src, PIM_RREG_SA, PIM_RREG_R2);
  for (int i = 0; i < 32; ++i) {
    pimOpReadRowToSa(src, i);
    pimOpXor(src, PIM_RREG_SA, PIM_RREG_R2, PIM_RREG_SA);
    pimOpXor(src, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
    pimOpNot(src, PIM_RREG_SA, PIM_RREG_SA);
    pimOpAnd(src, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
}

void
bsBitsimd::bsInt32And(PimObjId src1, PimObjId src2, PimObjId dest)
{
  for (int i = 0; i < 32; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    pimOpReadRowToSa(src2, i);
    pimOpAnd(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

void
bsBitsimd::bsInt32Or(PimObjId src1, PimObjId src2, PimObjId dest)
{
  for (int i = 0; i < 32; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    pimOpReadRowToSa(src2, i);
    pimOpOr(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

void
bsBitsimd::bsInt32Xor(PimObjId src1, PimObjId src2, PimObjId dest)
{
  for (int i = 0; i < 32; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    pimOpReadRowToSa(src2, i);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

void
bsBitsimd::bsInt32Xnor(PimObjId src1, PimObjId src2, PimObjId dest)
{
  for (int i = 0; i < 32; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    pimOpReadRowToSa(src2, i);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    pimOpNot(src1, PIM_RREG_SA, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

void
bsBitsimd::bsInt32GT(PimObjId src1, PimObjId src2, PimObjId dest)
{
  // 32-bit uint gt
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < 31; ++i) {
    pimOpReadRowToSa(src2, i);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpReadRowToSa(src1, i);
    pimOpNot(src1, PIM_RREG_R2, PIM_RREG_R2);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
  // handle sign bit
  pimOpReadRowToSa(src1, 31);
  pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R2);
  pimOpReadRowToSa(src2, 31);
  pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R2, PIM_RREG_R2);
  pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
  pimOpWriteSaToRow(dest, 0);

  // set other bits of dest to 0
  pimOpSet(src1, PIM_RREG_SA, 0);
  for (int i = 1; i < 32; ++i) {
    pimOpWriteSaToRow(dest, i);
  }
}

void
bsBitsimd::bsInt32LT(PimObjId src1, PimObjId src2, PimObjId dest)
{
  // 32-bit uint lt
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < 31; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpReadRowToSa(src2, i);
    pimOpNot(src1, PIM_RREG_R2, PIM_RREG_R2);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
  // handle sign bit
  pimOpReadRowToSa(src2, 31);
  pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R2);
  pimOpReadRowToSa(src1, 31);
  pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R2, PIM_RREG_R2);
  pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
  pimOpWriteSaToRow(dest, 0);

  // set other bits of dest to 0
  pimOpSet(src1, PIM_RREG_SA, 0);
  for (int i = 1; i < 32; ++i) {
    pimOpWriteSaToRow(dest, i);
  }
}

void
bsBitsimd::bsInt32EQ(PimObjId src1, PimObjId src2, PimObjId dest)
{
  pimOpSet(src1, PIM_RREG_R2, 0);
  for (int i = 0; i < 32; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    pimOpReadRowToSa(src2, i);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
    pimOpOr(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_R2);
  }
  pimOpNot(src1, PIM_RREG_R2, PIM_RREG_SA);
  pimOpWriteSaToRow(dest, 0);

  // set other bits of dest to 0
  pimOpSet(src1, PIM_RREG_SA, 0);
  for (int i = 1; i < 32; ++i) {
    pimOpWriteSaToRow(dest, i);
  }
}

void
bsBitsimd::bsInt32Min(PimObjId src1, PimObjId src2, PimObjId dest)
{
  // 32-bit int lt
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < 31; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpReadRowToSa(src2, i);
    pimOpNot(src1, PIM_RREG_R2, PIM_RREG_R2);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
  pimOpReadRowToSa(src2, 31);
  pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R2);
  pimOpReadRowToSa(src1, 31);
  pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R2, PIM_RREG_R2);
  pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);

  // if-else copy
  for (int i = 0; i < 32; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    pimOpReadRowToSa(src2, i);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

void
bsBitsimd::bsInt32Max(PimObjId src1, PimObjId src2, PimObjId dest)
{
  // 32-bit int gt
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < 31; ++i) {
    pimOpReadRowToSa(src2, i);
    pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
    pimOpReadRowToSa(src1, i);
    pimOpNot(src1, PIM_RREG_R2, PIM_RREG_R2);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
  }
  pimOpReadRowToSa(src1, 31);
  pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R2);
  pimOpReadRowToSa(src2, 31);
  pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R2, PIM_RREG_R2);
  pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);

  // if-else copy
  for (int i = 0; i < 32; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    pimOpReadRowToSa(src2, i);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }
}

void
bsBitsimd::bsInt32PopCount(PimObjId src, PimObjId dest)
{
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
  for (int i = 6; i < 32; ++i) {
    pimOpWriteSaToRow(dest, i);
  }
}

void
bsBitsimd::bsInt32MulHelper3Reg(int nBit, PimObjId src1, PimObjId src2, PimObjId dest)
{
  assert(nBit <= 32);
  std::cout << "BS-INFO: Allocate 32 temporary rows" << std::endl;
  PimObjId tmp = pimAllocAssociated(32, src1, PIM_INT32);

  // cond copy the first
  pimOpReadRowToSa(src1, 0);
  pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R2);
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < nBit; ++i) {
    pimOpReadRowToSa(src2, i);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }

  // add and cond write
  for (int i = 1; i < nBit; ++i) {
    pimOpSet(src1, PIM_RREG_R1, 0);
    for (int j = 0; i + j < nBit; ++j) {
      pimOpReadRowToSa(dest, i + j);
      pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
      pimOpReadRowToSa(src2, j);
      pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);
      pimOpXor(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_SA);
      pimOpWriteSaToRow(tmp, j);
    }
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R2);
    for (int j = 0; i + j < nBit; ++j) {
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
bsBitsimd::bsInt32MulHelper4Reg(int nBit, PimObjId src1, PimObjId src2, PimObjId dest)
{
  assert(nBit <= 32);
  // cond copy the first
  pimOpReadRowToSa(src1, 0);
  pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R2);
  pimOpSet(src1, PIM_RREG_R1, 0);
  for (int i = 0; i < nBit; ++i) {
    pimOpReadRowToSa(src2, i);
    pimOpSel(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    pimOpWriteSaToRow(dest, i);
  }

  // add and cond write
  for (int i = 1; i < nBit; ++i) {
    pimOpReadRowToSa(src1, i);
    pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R3); // cond

    pimOpSet(src1, PIM_RREG_R1, 0); // carry
    for (int j = 0; i + j < nBit; ++j) {
      // add
      pimOpReadRowToSa(src2, j);
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

void
bsBitsimd::bsInt32DivRemHelper(int nBit, PimObjId src1, PimObjId src2, PimObjId dest)
{
  // compute abs
  std::cout << "BS-INFO: Allocate 64 temporary rows" << std::endl;
  PimObjId abs1 = pimAllocAssociated(32, src1, PIM_INT32);
  PimObjId abs2 = pimAllocAssociated(32, src2, PIM_INT32);
  bsInt32Abs(src1, abs1);
  bsInt32Abs(src2, abs2);

  // 31-bit uint div rem
  bsUint32DivRemHelper(31, abs1, abs2, dest);
  pimOpSet(src1, PIM_RREG_SA, 0);
  pimOpWriteSaToRow(dest, 31);

  // check sign
  pimOpReadRowToSa(src1, 31);
  pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
  pimOpReadRowToSa(src2, 31);
  pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R1);

  // sign ext
  pimOpMove(src1, PIM_RREG_R1, PIM_RREG_R2);
  for (int i = 0; i < 32; ++i) {
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
bsBitsimd::bsUint32DivRemHelper(int nBit, PimObjId src1, PimObjId src2, PimObjId dest)
{
  assert(nBit <= 32);
  // quotient and remainder
  std::cout << "BS-INFO: Allocate 96 temporary rows" << std::endl;
  PimObjId qr = pimAllocAssociated(64, src1, PIM_INT64);
  PimObjId tmp = pimAllocAssociated(32, src1, PIM_INT32);

  // init 64-bit space
  pimOpSet(src1, PIM_RREG_SA, 0);
  for (int i = 0; i < nBit * 2; ++i) {
    pimOpWriteSaToRow(qr, i);
  }

  // compute
  for (int i = 0; i < nBit; ++i) {
    pimOpReadRowToSa(src1, nBit - 1 - i);
    pimOpWriteSaToRow(qr, nBit - 1 - i);

    // qr[62-i:31-i] - b
    pimOpSet(src1, PIM_RREG_R1, 0);
    for (int j = 0; j < nBit; ++j) {
      pimOpReadRowToSa(qr, nBit - 1 - i + j);
      pimOpXor(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_R2);
      pimOpReadRowToSa(src2, j);
      pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_R1);
      pimOpXor(src1, PIM_RREG_R2, PIM_RREG_SA, PIM_RREG_SA);
      pimOpWriteSaToRow(tmp, j);
    }

    // update quotient
    pimOpNot(src1, PIM_RREG_R1, PIM_RREG_R1);
    pimOpMove(src1, PIM_RREG_R1, PIM_RREG_SA);
    pimOpWriteSaToRow(qr, nBit * 2 - 1 - i);

    // update remainder
    pimOpMove(src1, PIM_RREG_R1, PIM_RREG_R2);
    for (int j = 0; j <= i; ++j) {
      pimOpReadRowToSa(tmp, j);
      pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
      pimOpReadRowToSa(qr, nBit - 1 - i + j);
      pimOpSel(src1, PIM_RREG_R2, PIM_RREG_R1, PIM_RREG_SA, PIM_RREG_SA);
      pimOpWriteSaToRow(qr, nBit - 1 - i + j);
    }
  }

  // copy results
  for (int i = 0; i < nBit; ++i) {
    pimOpReadRowToSa(qr, i + nBit);
    pimOpWriteSaToRow(dest, i);
  }

  pimFree(qr);
  pimFree(tmp);
}
