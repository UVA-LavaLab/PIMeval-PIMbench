// File: pimCmd.h
// PIM Functional Simulator - PIM Commands
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef LAVA_PIM_CMD_H
#define LAVA_PIM_CMD_H

#include "libpimsim.h"
#include <vector>
#include <string>
#include <bit>
#include <limits>
#include <cassert>

class pimDevice;
class pimResMgr;

enum class PimCmdEnum {
  NOOP = 0,
  // Functional 1-operand v-layout
  ABS_V,
  POPCOUNT_V,
  BROADCAST_V,
  BROADCAST_H,
  // Functional 2-operand v-layout
  ADD_V,
  SUB_V,
  MUL_V,
  DIV_V,
  AND_V,
  OR_V,
  XOR_V,
  GT_V,
  LT_V,
  EQ_V,
  MIN_V,
  MAX_V,
  // Functional special v-layout
  REDSUM_V,
  REDSUM_RANGE_V,
  ROTATE_R_V,
  ROTATE_L_V,
  // BitSIMD v-layout commands
  ROW_R,
  ROW_W,
  RREG_MOV,
  RREG_SET,
  RREG_NOT,
  RREG_AND,
  RREG_OR,
  RREG_NAND,
  RREG_NOR,
  RREG_XOR,
  RREG_XNOR,
  RREG_MAJ,
  RREG_SEL,
  RREG_ROTATE_R,
  RREG_ROTATE_L,
};


//! @class  pimCmd
//! @brief  Pim command base class
class pimCmd
{
public:
  pimCmd(PimCmdEnum cmdType) : m_cmdType(cmdType) {}
  virtual ~pimCmd() {}

  virtual bool execute(pimDevice* device) = 0;

  std::string getName() const { return getName(m_cmdType); }
  static std::string getName(PimCmdEnum cmdType);

  virtual void updateStats(int numPass);

protected:
  bool isCoreAligned(PimObjId objId1, PimObjId objId2, pimResMgr* resMgr);
  bool isVAligned(PimObjId objId1, PimObjId objId2, pimResMgr* resMgr);
  bool isHAligned(PimObjId objId1, PimObjId objId2, pimResMgr* resMgr);

  PimCmdEnum m_cmdType;
};

//! @class  pimCmdFunc1V
//! @brief  Pim CMD: Functional 1-operand v-layout 
class pimCmdFunc1V : public pimCmd
{
public:
  pimCmdFunc1V(PimCmdEnum cmdType, PimObjId src, PimObjId dest)
    : pimCmd(cmdType), m_src(src), m_dest(dest) {}
  virtual ~pimCmdFunc1V() {}
  virtual bool execute(pimDevice* device) override;
  virtual void updateStats(int numPass) override;
protected:
  PimObjId m_src;
  PimObjId m_dest;
};

//! @class  pimCmdFunc2V
//! @brief  Pim CMD: Functional 2-operand v-layout 
class pimCmdFunc2V : public pimCmd
{
public:
  pimCmdFunc2V(PimCmdEnum cmdType, PimObjId src1, PimObjId src2, PimObjId dest)
    : pimCmd(cmdType), m_src1(src1), m_src2(src2), m_dest(dest) {}
  virtual ~pimCmdFunc2V() {}
  virtual bool execute(pimDevice* device) override;
  virtual void updateStats(int numPass) override;
protected:
  PimObjId m_src1;
  PimObjId m_src2;
  PimObjId m_dest;
};

//! @class  pimCmdedSum
//! @brief  Pim CMD: RedSum non-ranged/ranged v-layout
class pimCmdRedSumV : public pimCmd
{
public:
  pimCmdRedSumV(PimCmdEnum cmdType, PimObjId src, int* result)
    : pimCmd(cmdType), m_src(src), m_result(result) {}
  pimCmdRedSumV(PimCmdEnum cmdType, PimObjId src, int* result, unsigned idxBegin, unsigned idxEnd)
    : pimCmd(cmdType), m_src(src), m_result(result), m_idxBegin(idxBegin), m_idxEnd(idxEnd) {}
  virtual ~pimCmdRedSumV() {}
  virtual bool execute(pimDevice* device) override;
  virtual void updateStats(int numPass) override;
protected:
  PimObjId m_src;
  int* m_result;
  unsigned m_idxBegin = 0;
  unsigned m_idxEnd = std::numeric_limits<unsigned>::max();
  uint64_t m_numElements = 0;
  uint64_t m_totalBytes = 0;
};

//! @class  pimCmdBroadcast
//! @brief  Pim CMD: Broadcast a value to all elements, v/h-layout
class pimCmdBroadcast : public pimCmd
{
public:
  pimCmdBroadcast(PimCmdEnum cmdType, PimObjId dest, unsigned val)
    : pimCmd(cmdType), m_dest(dest), m_val(val) {}
  virtual ~pimCmdBroadcast() {}
  virtual bool execute(pimDevice* device) override;
  virtual void updateStats(int numPass) override;
protected:
  PimObjId m_dest;
  unsigned m_val;
  unsigned m_bitsPerElement = 0;
  unsigned m_numElements = 0;
  unsigned m_numRegions = 0;
  unsigned m_maxElementsPerRegion = 0;
};

//! @class  pimCmdRotateV
//! @brief  Pim CMD: rotate right/left v-layout
class pimCmdRotateV : public pimCmd
{
public:
  pimCmdRotateV(PimCmdEnum cmdType, PimObjId src)
    : pimCmd(cmdType), m_src(src) {}
  virtual ~pimCmdRotateV() {}
  virtual bool execute(pimDevice* device) override;
  virtual void updateStats(int numPass) override;
protected:
  PimObjId m_src;
  unsigned m_numRegions = 0;
  unsigned m_bitsPerElement = 0;
  unsigned m_numElements = 0;
};

//! @class  pimCmdReadRowToSa
//! @brief  Pim CMD: BitSIMD-V: Read a row to SA
class pimCmdReadRowToSa : public pimCmd
{
public:
  pimCmdReadRowToSa(PimCmdEnum cmdType, PimObjId objId, unsigned ofst)
    : pimCmd(cmdType), m_objId(objId), m_ofst(ofst) {}
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
  pimCmdWriteSaToRow(PimCmdEnum cmdType, PimObjId objId, unsigned ofst)
    : pimCmd(cmdType), m_objId(objId), m_ofst(ofst) {}
  virtual ~pimCmdWriteSaToRow() {}
  virtual bool execute(pimDevice* device) override;
protected:
  PimObjId m_objId;
  unsigned m_ofst;
};

//! @class  pimCmdRRegOp : public pimCmd
//! @brief  Pim CMD: BitSIMD-V: Row reg operations
class pimCmdRRegOp : public pimCmd
{
public:
  pimCmdRRegOp(PimCmdEnum cmdType, PimObjId objId, PimRowReg dest, bool val)
    : pimCmd(cmdType), m_objId(objId), m_dest(dest), m_val(val)
  {
    assert(cmdType == PimCmdEnum::RREG_SET);
  }
  pimCmdRRegOp(PimCmdEnum cmdType, PimObjId objId, PimRowReg dest, PimRowReg src1)
    : pimCmd(cmdType), m_objId(objId), m_dest(dest), m_src1(src1)
  {
    assert(cmdType == PimCmdEnum::RREG_MOV || cmdType == PimCmdEnum::RREG_NOT);
  }
  pimCmdRRegOp(PimCmdEnum cmdType, PimObjId objId, PimRowReg dest, PimRowReg src1, PimRowReg src2)
    : pimCmd(cmdType), m_objId(objId), m_dest(dest), m_src1(src1), m_src2(src2)
  {
  }
  pimCmdRRegOp(PimCmdEnum cmdType, PimObjId objId, PimRowReg dest, PimRowReg src1, PimRowReg src2, PimRowReg src3)
    : pimCmd(cmdType), m_objId(objId), m_dest(dest), m_src1(src1), m_src2(src2), m_src3(src3)
  {
    assert(cmdType == PimCmdEnum::RREG_MAJ || cmdType == PimCmdEnum::RREG_SEL);
  }
  virtual ~pimCmdRRegOp() {}
  virtual bool execute(pimDevice* device) override;
protected:
  PimObjId m_objId;
  PimRowReg m_dest;
  bool m_val = 0;
  PimRowReg m_src1 = PIM_RREG_NONE;
  PimRowReg m_src2 = PIM_RREG_NONE;
  PimRowReg m_src3 = PIM_RREG_NONE;
};

//! @class  pimCmdRRegRotate
//! @brief  Pim CMD: BitSIMD-V: row reg rotate right by one step
class pimCmdRRegRotate : public pimCmd
{
public:
  pimCmdRRegRotate(PimCmdEnum cmdType, PimObjId objId, PimRowReg dest)
    : pimCmd(cmdType), m_objId(objId), m_dest(dest) {}
  virtual ~pimCmdRRegRotate() {}
  virtual bool execute(pimDevice* device) override;
protected:
  PimObjId m_objId;
  PimRowReg m_dest;
};


#endif

