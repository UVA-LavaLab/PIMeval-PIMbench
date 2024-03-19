// File: pimCmd.h
// PIM Functional Simulator - PIM Commands
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef LAVA_PIM_CMD_H
#define LAVA_PIM_CMD_H

#include "libpimsim.h"
#include <vector>

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

protected:
  bool isCoreAligned(PimObjId objId1, PimObjId objId2, pimResMgr* resMgr);
  bool isVAligned(PimObjId objId1, PimObjId objId2, pimResMgr* resMgr);
  bool isHAligned(PimObjId objId1, PimObjId objId2, pimResMgr* resMgr);
};

//! @class  pimCmdInt32AddV
//! @brief  Pim CMD: int32 add v-layout
class pimCmdInt32AddV : public pimCmd
{
public:
  pimCmdInt32AddV(PimObjId src1, PimObjId src2, PimObjId dest)
    : m_src1(src1), m_src2(src2), m_dest(dest) {}
  virtual ~pimCmdInt32AddV() {}

  virtual bool execute(pimDevice* device) override;

protected:
  PimObjId m_src1;
  PimObjId m_src2;
  PimObjId m_dest;
};

//! @class  pimCmdInt32AbsV
//! @brief  Pim CMD: int32 abs v-layout
class pimCmdInt32AbsV : public pimCmd
{
public:
  pimCmdInt32AbsV(PimObjId src, PimObjId dest)
    : m_src(src), m_dest(dest) {}
  virtual ~pimCmdInt32AbsV() {}

  virtual bool execute(pimDevice* device) override;

protected:
  PimObjId m_src;
  PimObjId m_dest;
};

//! @class  pimCmdInt32RedSum
//! @brief  Pim CMD: int32 RedSum v-layout
class pimCmdInt32RedSum : public pimCmd
{
public:
  pimCmdInt32RedSum(PimObjId src, int& result)
    : m_src(src), m_result(result) {}
  virtual ~pimCmdInt32RedSum() {}

  virtual bool execute(pimDevice* device) override;

protected:
  PimObjId m_src;
  int& m_result;
};

//! @class  pimCmdInt32RedSumRanged
//! @brief  Pim CMD: int32 RedSumRanged v-layout
class pimCmdInt32RedSumRanged : public pimCmd
{
public:
  pimCmdInt32RedSumRanged(PimObjId src, int& result, unsigned idxBegin, unsigned idxEnd)
    : m_src(src), m_result(result), m_idxBegin(idxBegin), m_idxEnd(idxEnd) {}
  virtual ~pimCmdInt32RedSumRanged() {}

  virtual bool execute(pimDevice* device) override;

protected:
  PimObjId m_src;
  int& m_result;
  unsigned m_idxBegin; 
  unsigned m_idxEnd;
};

//! @class  pimCmdInt32MulV
//! @brief  Pim CMD: int32 add v-layout
class pimCmdInt32MulV : public pimCmd
{
public:
  pimCmdInt32MulV(PimObjId src1, PimObjId src2, PimObjId dest)
    : m_src1(src1), m_src2(src2), m_dest(dest) {}
  virtual ~pimCmdInt32MulV() {}

  virtual bool execute(pimDevice* device) override;

protected:
  PimObjId m_src1;
  PimObjId m_src2;
  PimObjId m_dest;
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
protected:
  PimObjId m_objId;
  PimRowReg m_dest;
};

#endif

