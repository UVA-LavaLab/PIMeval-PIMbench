// File: pimPerfEnergyModels.cc
// PIMeval Simulator - Performance Energy Models
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimPerfEnergyBase.h"
#include "pimSim.h"
#include "pimCmd.h"
#include "pimPerfEnergyBitSerial.h"
#include "pimPerfEnergyFulcrum.h"
#include "pimPerfEnergyBankLevel.h"
#include <cstdio>


//! @brief  A factory function to get perf energy model for sim target
pimPerfEnergyBase*
pimPerfEnergyFactory::getPerfEnergyModel(PimDeviceEnum simTarget, pimParamsDram* paramsDram)
{
  pimPerfEnergyBase* model = nullptr;
  switch (simTarget) {
    case PIM_DEVICE_BITSIMD_V:
    case PIM_DEVICE_BITSIMD_V_AP:
    case PIM_DEVICE_BITSIMD_H:
    case PIM_DEVICE_SIMDRAM:
      std::printf("PIM-Info: Created performance energy model for bit-serial PIM\n");
      model = new pimPerfEnergyBitSerial(paramsDram);
      break;
    case PIM_DEVICE_FULCRUM:
      std::printf("PIM-Info: Created performance energy model for Fulcrum\n");
      model = new pimPerfEnergyFulcrum(paramsDram);
      break;
    case PIM_DEVICE_BANK_LEVEL:
      std::printf("PIM-Info: Created performance energy model for bank-level PIM\n");
      model = new pimPerfEnergyBankLevel(paramsDram);
      break;
    default:
      std::printf("PIM-Info: Created performance energy model for base\n");
      model = new pimPerfEnergyBase(paramsDram);
  }
  return model;
}

///////////////////////////////////////////////////////////////////////////////
// MODEL: Base
///////////////////////////////////////////////////////////////////////////////

//! @brief  pimPerfEnergyBase ctor
pimPerfEnergyBase::pimPerfEnergyBase(pimParamsDram* paramsDram)
  : m_paramsDram(paramsDram)
{
  m_tR = m_paramsDram->getNsRowRead() / m_nano_to_milli;
  m_tW = m_paramsDram->getNsRowWrite() / m_nano_to_milli;
  m_tL = m_paramsDram->getNsTCCD_S() / m_nano_to_milli;
  m_tGDL = m_paramsDram->getNsTCAS() / m_nano_to_milli;
  m_eAP = m_paramsDram->getPjRowRead() / m_pico_to_milli; // Convert pJ to mJ
  m_eL = m_paramsDram->getPjLogic() / m_pico_to_milli; // Convert pJ to mJ
  m_eR = m_paramsDram->getMwRead() / 1000.0;
  m_eW = m_paramsDram->getMwWrite() / 1000.0;
  // m_pBCore = (m_paramsDram->getMwIDD3N() - m_paramsDram->getMwIDD2N()) / 1000.0; // Convert mW to W, so that W * ms = mJ
  m_pBChip = m_paramsDram->getMwIDD3N() / 1000.0; // Convert mW to W, so that W * ms = mJ
  m_GDLWidth = m_paramsDram->getBurstLength() * m_paramsDram->getDeviceWidth();
  m_numChipsPerRank = m_paramsDram->getNumChipsPerRank();
}

//! @brief  Perf energy model of data transfer between CPU memory and PIM memory
pimeval::perfEnergy
pimPerfEnergyBase::getPerfEnergyForBytesTransfer(PimCmdEnum cmdType, uint64_t numBytes) const
{
  double mjEnergy = 0.0;
  unsigned numRanks = pimSim::get()->getNumRanks();
  double typicalRankBW = m_paramsDram->getTypicalRankBW(); // GB/s
  double msRuntime = static_cast<double>(numBytes) / (typicalRankBW * numRanks * 1024 * 1024 * 1024 / 1000);
  switch (cmdType) {
    case PimCmdEnum::COPY_H2D:
    {
      mjEnergy = m_eW * msRuntime * m_numChipsPerRank * numRanks;
      mjEnergy += m_pBChip * m_numChipsPerRank * numRanks * msRuntime;
      break;
    }
    case PimCmdEnum::COPY_D2H:
    {
      mjEnergy = m_eR * msRuntime * m_numChipsPerRank * numRanks;
      mjEnergy += m_pBChip * m_numChipsPerRank * numRanks * msRuntime;
      break;
    }
    case PimCmdEnum::COPY_D2D:
    {
      // One row read, one row write within a subarray
      mjEnergy = m_eAP * 2 * msRuntime * m_numChipsPerRank * numRanks;
      mjEnergy += m_pBChip * m_numChipsPerRank * numRanks * msRuntime;
      break;
    }
    default:
    {
      std::printf("PIM-Warning: Perf energy model not available for PIM command %s\n", pimCmd::getName(cmdType, "").c_str());
      break;
    }
  }
  return pimeval::perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Perf energy model of base class for func1 (placeholder)
pimeval::perfEnergy
pimPerfEnergyBase::getPerfEnergyForFunc1(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 1e10;
  double mjEnergy = 999999999.9;
  return pimeval::perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Perf energy model of base class for func2 (placeholder)
pimeval::perfEnergy
pimPerfEnergyBase::getPerfEnergyForFunc2(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 1e10;
  double mjEnergy = 999999999.9;
  return pimeval::perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Perf energy model of base class for reduction sum (placeholder)
pimeval::perfEnergy
pimPerfEnergyBase::getPerfEnergyForRedSum(PimCmdEnum cmdType, const pimObjInfo& obj, unsigned numPass) const
{
  double msRuntime = 1e10;
  double mjEnergy = 999999999.9;
  return pimeval::perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Perf energy model of base class for broadcast (placeholder)
pimeval::perfEnergy
pimPerfEnergyBase::getPerfEnergyForBroadcast(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 1e10;
  double mjEnergy = 999999999.9;
  return pimeval::perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Perf energy model of base class for rotate (placeholder)
pimeval::perfEnergy
pimPerfEnergyBase::getPerfEnergyForRotate(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 1e10;
  double mjEnergy = 999999999.9;
  return pimeval::perfEnergy(msRuntime, mjEnergy);
}

