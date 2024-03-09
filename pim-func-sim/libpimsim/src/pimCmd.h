// File: pimCmd.h
// PIM Functional Simulator - PIM Commands
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef LAVA_PIM_CMD_H
#define LAVA_PIM_CMD_H

#include "libpimsim.h"
#include <vector>

class pimCore;

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
  ~pimCmd() {}

  virtual bool execute() = 0;

protected:
};

//! @class  pimAddInt32V
//! @brief  Pim OP: V add int32 --> need to avoid confusion from vec add
class pimAddInt32V : public pimCmd
{
public:
  pimAddInt32V(PimObjId src1, PimObjId src2, PimObjId dest)
    : m_src1(src1), m_src2(src2), m_dest(dest) {}
  ~pimAddInt32V() {}

  virtual bool execute();

protected:
  PimObjId m_src1;
  PimObjId m_src2;
  PimObjId m_dest;
};


#endif

