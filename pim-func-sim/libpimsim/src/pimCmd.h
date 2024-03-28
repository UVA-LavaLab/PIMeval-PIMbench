// File: pimCmd.h
// PIM Functional Simulator - PIM Commands
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef LAVA_PIM_CMD_H
#define LAVA_PIM_CMD_H

#include "libpimsim.h"
#include <vector>
#include <string>
#include <bit>

class pimDevice;
class pimResMgr;

enum class PimCmdEnum {
  NOOP = 0,
  ADD_INT32_V,
};


//! @class  pimCmd
//! @brief  Pim command base class
class pimCmd
{
public:
  pimCmd() {}
  virtual ~pimCmd() {}

  virtual bool execute(pimDevice* device) = 0;
  virtual std::string getName() const = 0;

protected:
  bool isCoreAligned(PimObjId objId1, PimObjId objId2, pimResMgr* resMgr);
  bool isVAligned(PimObjId objId1, PimObjId objId2, pimResMgr* resMgr);
  bool isHAligned(PimObjId objId1, PimObjId objId2, pimResMgr* resMgr);
};

//! @class  pimCmdAddV
//! @brief  Pim CMD: add v-layout
class pimCmdAddV : public pimCmd
{
public:
  pimCmdAddV(PimObjId src1, PimObjId src2, PimObjId dest)
    : m_src1(src1), m_src2(src2), m_dest(dest) {}
  virtual ~pimCmdAddV() {}

  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "add.v"; }

protected:
  PimObjId m_src1;
  PimObjId m_src2;
  PimObjId m_dest;
};

//! @class  pimCmdSubV
//! @brief  Pim CMD: sub v-layout
class pimCmdSubV : public pimCmd
{
public:
  pimCmdSubV(PimObjId src1, PimObjId src2, PimObjId dest)
    : m_src1(src1), m_src2(src2), m_dest(dest) {}
  virtual ~pimCmdSubV() {}

  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "sub.v"; }

protected:
  PimObjId m_src1;
  PimObjId m_src2;
  PimObjId m_dest;
};

//! @class  pimCmdDivV
//! @brief  Pim CMD: div v-layout
class pimCmdDivV : public pimCmd
{
public:
  pimCmdDivV(PimObjId src1, PimObjId src2, PimObjId dest)
    : m_src1(src1), m_src2(src2), m_dest(dest) {}
  virtual ~pimCmdDivV() {}

  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "div.v"; }

protected:
  PimObjId m_src1;
  PimObjId m_src2;
  PimObjId m_dest;
};

//! @class  pimCmdAbsV
//! @brief  Pim CMD: abs v-layout
class pimCmdAbsV : public pimCmd
{
public:
  pimCmdAbsV(PimObjId src, PimObjId dest)
    : m_src(src), m_dest(dest) {}
  virtual ~pimCmdAbsV() {}

  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "abs.v"; }

protected:
  PimObjId m_src;
  PimObjId m_dest;
};

//! @class  pimCmdAndV
//! @brief  Pim CMD: and v-layout
class pimCmdAndV : public pimCmd
{
public:
  pimCmdAndV(PimObjId src1, PimObjId src2, PimObjId dest)
    : m_src1(src1), m_src2(src2), m_dest(dest) {}
  virtual ~pimCmdAndV() {}

  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "and.v"; }

protected:
  PimObjId m_src1;
  PimObjId m_src2;
  PimObjId m_dest;
};

//! @class  pimCmdOrV
//! @brief  Pim CMD: or v-layout
class pimCmdOrV : public pimCmd
{
public:
  pimCmdOrV(PimObjId src1, PimObjId src2, PimObjId dest)
    : m_src1(src1), m_src2(src2), m_dest(dest) {}
  virtual ~pimCmdOrV() {}

  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "or.v"; }

protected:
  PimObjId m_src1;
  PimObjId m_src2;
  PimObjId m_dest;
};

//! @class  pimCmdXorV
//! @brief  Pim CMD: xor v-layout
class pimCmdXorV : public pimCmd
{
public:
  pimCmdXorV(PimObjId src1, PimObjId src2, PimObjId dest)
    : m_src1(src1), m_src2(src2), m_dest(dest) {}
  virtual ~pimCmdXorV() {}

  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "xor.v"; }

protected:
  PimObjId m_src1;
  PimObjId m_src2;
  PimObjId m_dest;
};

//! @class  pimCmdRedSum
//! @brief  Pim CMD: RedSum v-layout
class pimCmdRedSum : public pimCmd
{
public:
  pimCmdRedSum(PimObjId src, int& result)
    : m_src(src), m_result(result) {}
  virtual ~pimCmdRedSum() {}

  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "redsum.v"; }

protected:
  PimObjId m_src;
  int& m_result;
};

//! @class  pimCmdRedSumRanged
//! @brief  Pim CMD: RedSumRanged v-layout
class pimCmdRedSumRanged : public pimCmd
{
public:
  pimCmdRedSumRanged(PimObjId src, int& result, unsigned idxBegin, unsigned idxEnd)
    : m_src(src), m_result(result), m_idxBegin(idxBegin), m_idxEnd(idxEnd) {}
  virtual ~pimCmdRedSumRanged() {}

  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "redsum_range.v"; }

protected:
  PimObjId m_src;
  int& m_result;
  unsigned m_idxBegin; 
  unsigned m_idxEnd;
};

//! @class  pimCmdMulV
//! @brief  Pim CMD: mul v-layout
class pimCmdMulV : public pimCmd
{
public:
  pimCmdMulV(PimObjId src1, PimObjId src2, PimObjId dest)
    : m_src1(src1), m_src2(src2), m_dest(dest) {}
  virtual ~pimCmdMulV() {}

  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "mul.v"; }

protected:
  PimObjId m_src1;
  PimObjId m_src2;
  PimObjId m_dest;
};

//! @class  pimCmdGTV
//! @brief  Pim CMD: gt v-layout
class pimCmdGTV : public pimCmd
{
public:
  pimCmdGTV(PimObjId src1, PimObjId src2, PimObjId dest)
    : m_src1(src1), m_src2(src2), m_dest(dest) {}
  virtual ~pimCmdGTV() {}

  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "gt.v"; }

protected:
  PimObjId m_src1;
  PimObjId m_src2;
  PimObjId m_dest;
};

//! @class  pimCmdLTV
//! @brief  Pim CMD: lt v-layout
class pimCmdLTV : public pimCmd
{
public:
  pimCmdLTV(PimObjId src1, PimObjId src2, PimObjId dest)
    : m_src1(src1), m_src2(src2), m_dest(dest) {}
  virtual ~pimCmdLTV() {}

  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "lt.v"; }

protected:
  PimObjId m_src1;
  PimObjId m_src2;
  PimObjId m_dest;
};

//! @class  pimCmdEQV
//! @brief  Pim CMD: eq v-layout
class pimCmdEQV : public pimCmd
{
public:
  pimCmdEQV(PimObjId src1, PimObjId src2, PimObjId dest)
    : m_src1(src1), m_src2(src2), m_dest(dest) {}
  virtual ~pimCmdEQV() {}

  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "eq.v"; }

protected:
  PimObjId m_src1;
  PimObjId m_src2;
  PimObjId m_dest;
};

//! @class  pimCmdMinV
//! @brief  Pim CMD: eq v-layout
class pimCmdMinV : public pimCmd
{
public:
  pimCmdMinV(PimObjId src1, PimObjId src2, PimObjId dest)
    : m_src1(src1), m_src2(src2), m_dest(dest) {}
  virtual ~pimCmdMinV() {}

  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "min.v"; }

protected:
  PimObjId m_src1;
  PimObjId m_src2;
  PimObjId m_dest;
};

//! @class  pimCmdMaxV
//! @brief  Pim CMD: max v-layout
class pimCmdMaxV : public pimCmd
{
public:
  pimCmdMaxV(PimObjId src1, PimObjId src2, PimObjId dest)
    : m_src1(src1), m_src2(src2), m_dest(dest) {}
  virtual ~pimCmdMaxV() {}

  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "max.v"; }

protected:
  PimObjId m_src1;
  PimObjId m_src2;
  PimObjId m_dest;
};

//! @class  pimCmdPopCountV
//! @brief  Pim CMD: pop count v-layout
class pimCmdPopCountV : public pimCmd
{
public:
  pimCmdPopCountV(PimObjId src, PimObjId dest)
    : m_src(src), m_dest(dest) {}
  virtual ~pimCmdPopCountV() {}

  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "popcount.v"; }

protected:
  PimObjId m_src;
  PimObjId m_dest;
};

//! @class  pimCmdRotateRightV
//! @brief  Pim CMD: rotate right v-layout
class pimCmdRotateRightV : public pimCmd
{
public:
  pimCmdRotateRightV(PimObjId src)
    : m_src(src) {}
  virtual ~pimCmdRotateRightV() {}

  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "rotate.right.v"; }

protected:
  PimObjId m_src;
};

//! @class  pimCmdRotateLeftV
//! @brief  Pim CMD: rotate left v-layout
class pimCmdRotateLeftV : public pimCmd
{
public:
  pimCmdRotateLeftV(PimObjId src)
    : m_src(src) {}
  virtual ~pimCmdRotateLeftV() {}

  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "rotate.left.v"; }

protected:
  PimObjId m_src;
};

//! @class  pimCmdReadRowToSa
//! @brief  Pim CMD: BitSIMD-V: Read a row to SA
class pimCmdReadRowToSa : public pimCmd
{
public:
  pimCmdReadRowToSa(PimObjId objId, unsigned ofst)
    : m_objId(objId), m_ofst(ofst) {}
  virtual ~pimCmdReadRowToSa() {}
  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "bitsimd.read_row.v"; }
protected:
  PimObjId m_objId;
  unsigned m_ofst;
};

//! @class  pimCmdWriteSaToRow
//! @brief  Pim CMD: BitSIMD-V: Write SA to a row
class pimCmdWriteSaToRow : public pimCmd
{
public:
  pimCmdWriteSaToRow(PimObjId objId, unsigned ofst)
    : m_objId(objId), m_ofst(ofst) {}
  virtual ~pimCmdWriteSaToRow() {}
  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "bitsimd.write_row.v"; }
protected:
  PimObjId m_objId;
  unsigned m_ofst;
};

//! @class  pimCmdRRegMove
//! @brief  Pim CMD: BitSIMD-V: Move row reg to reg
class pimCmdRRegMove : public pimCmd
{
public:
  pimCmdRRegMove(PimObjId objId, PimRowReg src, PimRowReg dest)
    : m_objId(objId), m_src(src), m_dest(dest) {}
  virtual ~pimCmdRRegMove() {}
  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "bitsimd.mov.v"; }
protected:
  PimObjId m_objId;
  PimRowReg m_src;
  PimRowReg m_dest;
};

//! @class  pimCmdRRegSet
//! @brief  Pim CMD: BitSIMD-V: Set row reg to 0/1
class pimCmdRRegSet : public pimCmd
{
public:
  pimCmdRRegSet(PimObjId objId, PimRowReg dest, bool val)
    : m_objId(objId), m_dest(dest), m_val(val) {}
  virtual ~pimCmdRRegSet() {}
  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "bitsimd.set.v"; }
protected:
  PimObjId m_objId;
  PimRowReg m_dest;
  bool m_val;
};

//! @class  pimCmdRRegNot
//! @brief  Pim CMD: BitSIMD-V: row reg dest = !src
class pimCmdRRegNot : public pimCmd
{
public:
  pimCmdRRegNot(PimObjId objId, PimRowReg src, PimRowReg dest)
    : m_objId(objId), m_src(src), m_dest(dest) {}
  virtual ~pimCmdRRegNot() {}
  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "bitsimd.not.v"; }
protected:
  PimObjId m_objId;
  PimRowReg m_src;
  PimRowReg m_dest;
};

//! @class  pimCmdRRegAnd
//! @brief  Pim CMD: BitSIMD-V: row reg dest = src1 & src2
class pimCmdRRegAnd : public pimCmd
{
public:
  pimCmdRRegAnd(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
    : m_objId(objId), m_src1(src1), m_src2(src2), m_dest(dest) {}
  virtual ~pimCmdRRegAnd() {}
  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "bitsimd.and.v"; }
protected:
  PimObjId m_objId;
  PimRowReg m_src1;
  PimRowReg m_src2;
  PimRowReg m_dest;
};

//! @class  pimCmdRRegOr
//! @brief  Pim CMD: BitSIMD-V: row reg dest = src1 | src2
class pimCmdRRegOr : public pimCmd
{
public:
  pimCmdRRegOr(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
    : m_objId(objId), m_src1(src1), m_src2(src2), m_dest(dest) {}
  virtual ~pimCmdRRegOr() {}
  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "bitsimd.or.v"; }
protected:
  PimObjId m_objId;
  PimRowReg m_src1;
  PimRowReg m_src2;
  PimRowReg m_dest;
};

//! @class  pimCmdRRegNand
//! @brief  Pim CMD: BitSIMD-V: row reg dest = src1 nand src2
class pimCmdRRegNand : public pimCmd
{
public:
  pimCmdRRegNand(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
    : m_objId(objId), m_src1(src1), m_src2(src2), m_dest(dest) {}
  virtual ~pimCmdRRegNand() {}
  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "bitsimd.nand.v"; }
protected:
  PimObjId m_objId;
  PimRowReg m_src1;
  PimRowReg m_src2;
  PimRowReg m_dest;
};

//! @class  pimCmdRRegNor
//! @brief  Pim CMD: BitSIMD-V: row reg dest = src1 nor src2
class pimCmdRRegNor : public pimCmd
{
public:
  pimCmdRRegNor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
    : m_objId(objId), m_src1(src1), m_src2(src2), m_dest(dest) {}
  virtual ~pimCmdRRegNor() {}
  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "bitsimd.nor.v"; }
protected:
  PimObjId m_objId;
  PimRowReg m_src1;
  PimRowReg m_src2;
  PimRowReg m_dest;
};

//! @class  pimCmdRRegXor
//! @brief  Pim CMD: BitSIMD-V: row reg dest = src1 xor src2
class pimCmdRRegXor : public pimCmd
{
public:
  pimCmdRRegXor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
    : m_objId(objId), m_src1(src1), m_src2(src2), m_dest(dest) {}
  virtual ~pimCmdRRegXor() {}
  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "bitsimd.xor.v"; }
protected:
  PimObjId m_objId;
  PimRowReg m_src1;
  PimRowReg m_src2;
  PimRowReg m_dest;
};

//! @class  pimCmdRRegXnor
//! @brief  Pim CMD: BitSIMD-V: row reg dest = src1 xnor src2
class pimCmdRRegXnor : public pimCmd
{
public:
  pimCmdRRegXnor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
    : m_objId(objId), m_src1(src1), m_src2(src2), m_dest(dest) {}
  virtual ~pimCmdRRegXnor() {}
  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "bitsimd.xnor.v"; }
protected:
  PimObjId m_objId;
  PimRowReg m_src1;
  PimRowReg m_src2;
  PimRowReg m_dest;
};

//! @class  pimCmdRRegMaj
//! @brief  Pim CMD: BitSIMD-V: row reg dest = maj(src1, src2, src3)
class pimCmdRRegMaj : public pimCmd
{
public:
  pimCmdRRegMaj(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg src3, PimRowReg dest)
    : m_objId(objId), m_src1(src1), m_src2(src2), m_src3(src3), m_dest(dest) {}
  virtual ~pimCmdRRegMaj() {}
  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "bitsimd.maj.v"; }
protected:
  PimObjId m_objId;
  PimRowReg m_src1;
  PimRowReg m_src2;
  PimRowReg m_src3;
  PimRowReg m_dest;
};

//! @class  pimCmdRRegSel
//! @brief  Pim CMD: BitSIMD-V: row reg dest = cond ? src1 : src2;
class pimCmdRRegSel : public pimCmd
{
public:
  pimCmdRRegSel(PimObjId objId, PimRowReg cond, PimRowReg src1, PimRowReg src2, PimRowReg dest)
    : m_objId(objId), m_cond(cond), m_src1(src1), m_src2(src2), m_dest(dest) {}
  virtual ~pimCmdRRegSel() {}
  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "bitsimd.sel.v"; }
protected:
  PimObjId m_objId;
  PimRowReg m_cond;
  PimRowReg m_src1;
  PimRowReg m_src2;
  PimRowReg m_dest;
};

//! @class  pimCmdRRegRotateR
//! @brief  Pim CMD: BitSIMD-V: row reg rotate right by one step
class pimCmdRRegRotateR : public pimCmd
{
public:
  pimCmdRRegRotateR(PimObjId objId, PimRowReg dest) : m_objId(objId), m_dest(dest) {}
  virtual ~pimCmdRRegRotateR() {}
  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "bitsimd.rotate_r.v"; }
protected:
  PimObjId m_objId;
  PimRowReg m_dest;
};

//! @class  pimCmdRRegRotateL
//! @brief  Pim CMD: BitSIMD-V: row reg rotate left by one step
class pimCmdRRegRotateL : public pimCmd
{
public:
  pimCmdRRegRotateL(PimObjId objId, PimRowReg dest) : m_objId(objId), m_dest(dest) {}
  virtual ~pimCmdRRegRotateL() {}
  virtual bool execute(pimDevice* device) override;
  virtual std::string getName() const override { return "bitsimd.rotate_l.v"; }
protected:
  PimObjId m_objId;
  PimRowReg m_dest;
};

#endif

