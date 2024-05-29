// File: pimCmd.h
// PIM Functional Simulator - PIM Commands
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef LAVA_PIM_CMD_H
#define LAVA_PIM_CMD_H

#include "libpimsim.h"
#include "pimResMgr.h"
#include "pimCore.h"
#include <vector>
#include <string>
#include <bit>
#include <limits>
#include <cassert>

class pimDevice;
class pimResMgr;

enum class PimCmdEnum {
  NOOP = 0,
  // Functional 1-operand
  ABS,
  POPCOUNT,
  BROADCAST,
  // Functional 2-operand
  ADD,
  SUB,
  MUL,
  DIV,
  AND,
  OR,
  XOR,
  XNOR,
  GT,
  LT,
  EQ,
  MIN,
  MAX,
  // Functional special
  REDSUM,
  REDSUM_RANGE,
  ROTATE_R,
  ROTATE_L,
  SHIFT_R,
  SHIFT_L,
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

  std::string getName() const {
    return getName(m_cmdType, "");
  }
  std::string getName(bool isVLayout) const {
    return getName(m_cmdType, isVLayout ? ".v" : ".h");
  }
  static std::string getName(PimCmdEnum cmdType, const std::string& suffix);

protected:
  bool isValidObjId(pimResMgr* resMgr, PimObjId objId) const;
  bool isAssociated(const pimObjInfo& obj1, const pimObjInfo& obj2) const;

  unsigned getNumElementsInRegion(const pimRegion& region, unsigned bitsPerElement) const;
  std::pair<unsigned, unsigned> locateNthB32(const pimRegion& region, bool isVLayout, unsigned nth) const;
  unsigned getB32(const pimCore& core, bool isVLayout, unsigned rowLoc, unsigned colLoc) const;
  void setB32(pimCore& core, bool isVLayout, unsigned rowLoc, unsigned colLoc, unsigned val) const;

  PimCmdEnum m_cmdType;
};

//! @class  pimCmdFunc1
//! @brief  Pim CMD: Functional 1-operand
class pimCmdFunc1 : public pimCmd
{
public:
  pimCmdFunc1(PimCmdEnum cmdType, PimObjId src, PimObjId dest)
    : pimCmd(cmdType), m_src(src), m_dest(dest) {}
  virtual ~pimCmdFunc1() {}
  virtual bool execute(pimDevice* device) override;
protected:
  PimObjId m_src;
  PimObjId m_dest;
};

//! @class  pimCmdFunc2
//! @brief  Pim CMD: Functional 2-operand
class pimCmdFunc2 : public pimCmd
{
public:
  pimCmdFunc2(PimCmdEnum cmdType, PimObjId src1, PimObjId src2, PimObjId dest)
    : pimCmd(cmdType), m_src1(src1), m_src2(src2), m_dest(dest) {}
  virtual ~pimCmdFunc2() {}
  virtual bool execute(pimDevice* device) override;
protected:
  PimObjId m_src1;
  PimObjId m_src2;
  PimObjId m_dest;
};

//! @class  pimCmdedSum
//! @brief  Pim CMD: RedSum non-ranged/ranged
class pimCmdRedSum : public pimCmd
{
public:
  pimCmdRedSum(PimCmdEnum cmdType, PimObjId src, int* result)
    : pimCmd(cmdType), m_src(src), m_result(result)
  {
    assert(cmdType == PimCmdEnum::REDSUM);
  }
  pimCmdRedSum(PimCmdEnum cmdType, PimObjId src, int* result, unsigned idxBegin, unsigned idxEnd)
    : pimCmd(cmdType), m_src(src), m_result(result), m_idxBegin(idxBegin), m_idxEnd(idxEnd)
  {
    assert(cmdType == PimCmdEnum::REDSUM_RANGE);
  }
  virtual ~pimCmdRedSum() {}
  virtual bool execute(pimDevice* device) override;
protected:
  PimObjId m_src;
  int* m_result;
  unsigned m_idxBegin = 0;
  unsigned m_idxEnd = std::numeric_limits<unsigned>::max();
};

//! @class  pimCmdBroadcast
//! @brief  Pim CMD: Broadcast a value to all elements
class pimCmdBroadcast : public pimCmd
{
public:
  pimCmdBroadcast(PimCmdEnum cmdType, PimObjId dest, unsigned val)
    : pimCmd(cmdType), m_dest(dest), m_val(val)
  {
    assert(cmdType == PimCmdEnum::BROADCAST);
  }
  virtual ~pimCmdBroadcast() {}
  virtual bool execute(pimDevice* device) override;
protected:
  PimObjId m_dest;
  unsigned m_val;
};

//! @class  pimCmdRotate
//! @brief  Pim CMD: rotate right/left
class pimCmdRotate : public pimCmd
{
public:
  pimCmdRotate(PimCmdEnum cmdType, PimObjId src)
    : pimCmd(cmdType), m_src(src)
  {
    assert(cmdType == PimCmdEnum::ROTATE_R || cmdType == PimCmdEnum::ROTATE_L ||
           cmdType == PimCmdEnum::SHIFT_R || cmdType == PimCmdEnum::SHIFT_L);
  }
  virtual ~pimCmdRotate() {}
  virtual bool execute(pimDevice* device) override;
protected:
  PimObjId m_src;
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

