// File: pimCmdFuse.h
// PIMeval Simulator - PIM API Fusion
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef LAVA_PIM_CMD_FUSE_H
#define LAVA_PIM_CMD_FUSE_H

#include "libpimeval.h"
#include "pimCmd.h"


//! @class  pimCmdFuse
//! @brief  Pim CMD: PIM API Fusion
class pimCmdFuse : public pimCmd
{
public:
  pimCmdFuse(PimProg prog) : pimCmd(PimCmdEnum::NOOP), m_prog(prog) {}
  virtual ~pimCmdFuse() {}
  virtual bool execute() override;
  virtual bool updateStats() const override;
private:
  PimProg m_prog;
};

#endif

