// File: pimPerfEnergyBitSerial.h
// PIMeval Simulator - Performance Energy Models
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef LAVA_PIM_PERF_ENERGY_BIT_SERIAL_H
#define LAVA_PIM_PERF_ENERGY_BIT_SERIAL_H

#include "libpimeval.h"                // for PimDeviceEnum, PimDataType
#include "pimParamsDram.h"             // for pimParamsDram
#include "pimCmd.h"                    // for PimCmdEnum
#include "pimResMgr.h"                 // for pimObjInfo
#include "pimPerfEnergyBase.h"         // for pimPerfEnergyBase


//! @class  pimPerfEnergyBitSerial
//! @brief  PIM performance energy model for bit-serial PIM family
class pimPerfEnergyBitSerial : public pimPerfEnergyBase
{
public:
  pimPerfEnergyBitSerial(const pimPerfEnergyModelParams& params) : pimPerfEnergyBase(params) {}
  virtual ~pimPerfEnergyBitSerial() {}

  virtual pimeval::perfEnergy getPerfEnergyForFunc1(PimCmdEnum cmdType, const pimObjInfo& obj) const override;
  virtual pimeval::perfEnergy getPerfEnergyForFunc2(PimCmdEnum cmdType, const pimObjInfo& obj) const override;
  virtual pimeval::perfEnergy getPerfEnergyForRedSum(PimCmdEnum cmdType, const pimObjInfo& obj, unsigned numPass) const override;
  virtual pimeval::perfEnergy getPerfEnergyForBroadcast(PimCmdEnum cmdType, const pimObjInfo& obj) const override;
  virtual pimeval::perfEnergy getPerfEnergyForRotate(PimCmdEnum cmdType, const pimObjInfo& obj) const override;

protected:
  pimeval::perfEnergy getPerfEnergyBitSerial(PimDeviceEnum deviceType, PimCmdEnum cmdType, PimDataType dataType, unsigned bitsPerElement, unsigned numPass, const pimObjInfo& obj) const;

  // Popcount logc Params from DRAM-CAM paper
  double m_pclNsDelay = 0.76; // 64-bit popcount logic ns delay, using LUT no pipeline design
  double m_pclUwPower = 0.03; // 64-bit popcount logic uW power, using LUT no pipeline design
};

#endif

