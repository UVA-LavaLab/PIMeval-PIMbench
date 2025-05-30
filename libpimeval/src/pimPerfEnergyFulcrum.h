// File: pimPerfEnergyFulcrum.h
// PIMeval Simulator - Performance Energy Models
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef LAVA_PIM_PERF_ENERGY_FULCRUM_H
#define LAVA_PIM_PERF_ENERGY_FULCRUM_H

#include "libpimeval.h"                // for PimDeviceEnum, PimDataType
#include "pimParamsDram.h"             // for pimParamsDram
#include "pimCmd.h"                    // for PimCmdEnum
#include "pimResMgr.h"                 // for pimObjInfo
#include "pimPerfEnergyBase.h"         // for pimPerfEnergyBase


//! @class  pimPerfEnergyBitFulcrum
//! @brief  PIM performance energy model for Fulcrum family
class pimPerfEnergyFulcrum : public pimPerfEnergyBase
{
public:
  pimPerfEnergyFulcrum(const pimPerfEnergyModelParams& params) : pimPerfEnergyBase(params) {}
  virtual ~pimPerfEnergyFulcrum() {}

  virtual pimeval::perfEnergy getPerfEnergyForFunc1(PimCmdEnum cmdType, const pimObjInfo& objSrc, const pimObjInfo& objDest) const override;
  virtual pimeval::perfEnergy getPerfEnergyForFunc2(PimCmdEnum cmdType, const pimObjInfo& objSrc1, const pimObjInfo& objSrc2, const pimObjInfo& objDest) const override;
  virtual pimeval::perfEnergy getPerfEnergyForReduction(PimCmdEnum cmdType, const pimObjInfo& obj, unsigned numPass) const override;
  virtual pimeval::perfEnergy getPerfEnergyForBroadcast(PimCmdEnum cmdType, const pimObjInfo& obj) const override;
  virtual pimeval::perfEnergy getPerfEnergyForRotate(PimCmdEnum cmdType, const pimObjInfo& obj) const override;
  virtual pimeval::perfEnergy getPerfEnergyForPrefixSum(PimCmdEnum cmdType, const pimObjInfo& obj) const override;

protected:
  double m_fulcrumMulLatency = 0.00000609; // 6.09ns
  double m_fulcrumAddLatency = 0.00000120; // 1.20ns
  unsigned m_fulcrumAluBitWidth = 32;
  // Following values are taken from fulcrum paper.
  double m_fulcrumMulEnergy = 0.0000000004992329586; // mJ
  double m_fulcrumAddEnergy = 0.0000000001467846411; // mJ
  double m_fulcrumShiftEnergy = 0.0000000075; // mJ
};

#endif

