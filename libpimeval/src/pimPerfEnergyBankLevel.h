// File: pimPerfEnergyBankLevel.h
// PIMeval Simulator - Performance Energy Models
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef LAVA_PIM_PERF_ENERGY_BANK_LEVEL_H
#define LAVA_PIM_PERF_ENERGY_BANK_LEVEL_H

#include "libpimeval.h"                // for PimDeviceEnum, PimDataType
#include "pimParamsDram.h"             // for pimParamsDram
#include "pimCmd.h"                    // for PimCmdEnum
#include "pimResMgr.h"                 // for pimObjInfo
#include "pimPerfEnergyBase.h"         // for pimPerfEnergyBase


//! @class  pimPerfEnergyBankLevel
//! @brief  PIM performance energy model for bank-level PIM family
class pimPerfEnergyBankLevel : public pimPerfEnergyBase
{
public:
  pimPerfEnergyBankLevel(const pimPerfEnergyModelParams& params) : pimPerfEnergyBase(params) {}
  virtual ~pimPerfEnergyBankLevel() {}

  virtual pimeval::perfEnergy getPerfEnergyForFunc1(PimCmdEnum cmdType, const pimObjInfo& objSrc, const pimObjInfo& objDest) const override;
  virtual pimeval::perfEnergy getPerfEnergyForFunc2(PimCmdEnum cmdType, const pimObjInfo& objSrc1, const pimObjInfo& objSrc2, const pimObjInfo& objDest) const override;
  virtual pimeval::perfEnergy getPerfEnergyForReduction(PimCmdEnum cmdType, const pimObjInfo& obj, unsigned numPass) const override;
  virtual pimeval::perfEnergy getPerfEnergyForBroadcast(PimCmdEnum cmdType, const pimObjInfo& obj) const override;
  virtual pimeval::perfEnergy getPerfEnergyForRotate(PimCmdEnum cmdType, const pimObjInfo& obj) const override;

protected:
  double m_blimpCoreOriginalLatency = 0.000005; // ms; 200 MHz. Reference: BLIMP paper
  double m_blimpCoreLatency = m_blimpCoreOriginalLatency > m_tGDL ? m_blimpCoreOriginalLatency : m_tGDL; // if compute unit is faster than GDL, compute unit needs to sit idly for next data 
  unsigned m_blimpCoreBitWidth = m_GDLWidth;
  unsigned m_simdUnitCount = m_blimpCoreBitWidth / 32; // 32-bit SIMD unit
  // Following values are taken from fulcrum paper as BLIMP paper does not model energy
  double m_blimpArithmeticEnergy = 0.0000000004992329586 * m_simdUnitCount; // mJ
  double m_blimpLogicalEnergy = 0.0000000001467846411 * m_simdUnitCount; // mJ
};

#endif

