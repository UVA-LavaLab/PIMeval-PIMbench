// File: pimPerfEnergyBitSerial.cc
// PIMeval Simulator - Performance Energy Models
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimPerfEnergyBitSerial.h"
#include "pimCmd.h"
#include "pimPerfEnergyTables.h"
#include <iostream>


//! @brief  Get performance and energy for bit-serial PIM
//!         BitSIMD and SIMDRAM need different fields
pimeval::perfEnergy
pimPerfEnergyBitSerial::getPerfEnergyBitSerial(PimDeviceEnum deviceType, PimCmdEnum cmdType, PimDataType dataType, unsigned bitsPerElement, unsigned numPass, const pimObjInfo& obj) const
{
  bool ok = false;
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned numCores = obj.getNumCoresUsed();

  switch (deviceType) {
    case PIM_DEVICE_BITSIMD_V:
    case PIM_DEVICE_BITSIMD_V_AP:
    case PIM_DEVICE_BITSIMD_H:
    {
      // BitSIMD-H reuse BitISMD-V perf for now
      if (deviceType == PIM_DEVICE_BITSIMD_H) {
        deviceType = PIM_DEVICE_BITSIMD_V;
      }
      // look up perf params from table
      auto it1 = pimPerfEnergyTables::bitsimdPerfTable.find(deviceType);
      if (it1 != pimPerfEnergyTables::bitsimdPerfTable.end()) {
        auto it2 = it1->second.find(dataType);
        if (it2 != it1->second.end()) {
          auto it3 = it2->second.find(cmdType);
          if (it3 != it2->second.end()) {
            const auto& [numR, numW, numL] = it3->second;
            msRuntime += m_tR * numR + m_tW * numW + m_tL * numL;
            mjEnergy += ((m_eL * numL * obj.getMaxElementsPerRegion()) + (m_eAP * numR + m_eAP * numW)) * numCores;
            mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
            ok = true;
          }
        }
      }
      // handle bit-shift specially
      if (cmdType == PimCmdEnum::SHIFT_BITS_L || cmdType == PimCmdEnum::SHIFT_BITS_R) {
        msRuntime += m_tR * (bitsPerElement - 1) + m_tW * bitsPerElement + m_tL;
        ok = true;
      }
      break;
    }
    case PIM_DEVICE_SIMDRAM:
    {
      break;
    }
    default:
      assert(0);
  }

  if (!ok) {
    std::cout << "PIM-Warning: Unimplemented bit-serial runtime estimation for"
              << " device=" << pimUtils::pimDeviceEnumToStr(deviceType)
              << " cmd=" << pimCmd::getName(cmdType, "")
              << " dataType=" << pimUtils::pimDataTypeEnumToStr(dataType)
              << std::endl;
    msRuntime = 1000000;
  }
  msRuntime *= numPass;
  mjEnergy *= numPass;

  return pimeval::perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Perf energy model of bit-serial PIM for func1
pimeval::perfEnergy
pimPerfEnergyBitSerial::getPerfEnergyForFunc1(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  PimDataType dataType = obj.getDataType();

  switch (m_simTarget) {
    case PIM_DEVICE_BITSIMD_V:
    case PIM_DEVICE_BITSIMD_V_AP:
    case PIM_DEVICE_BITSIMD_H:
    case PIM_DEVICE_SIMDRAM:
    {
      pimeval::perfEnergy perfEnergyBS = getPerfEnergyBitSerial(m_simTarget, cmdType, dataType, bitsPerElement, numPass, obj);
      msRuntime += perfEnergyBS.m_msRuntime;
      mjEnergy += perfEnergyBS.m_mjEnergy;
      break;
    }
    default:
      assert(0);
  }

  return pimeval::perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Perf energy model of bit-serial PIM for func2
pimeval::perfEnergy
pimPerfEnergyBitSerial::getPerfEnergyForFunc2(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  PimDataType dataType = obj.getDataType();

  switch (m_simTarget) {
    case PIM_DEVICE_BITSIMD_V:
    case PIM_DEVICE_BITSIMD_V_AP:
    case PIM_DEVICE_BITSIMD_H:
    case PIM_DEVICE_SIMDRAM:
    {
      pimeval::perfEnergy perfEnergyBS = getPerfEnergyBitSerial(m_simTarget, cmdType, dataType, bitsPerElement, numPass, obj);
      msRuntime += perfEnergyBS.m_msRuntime;
      mjEnergy += perfEnergyBS.m_mjEnergy;
      break;
    }
    default:
      assert(0);
  }

  return pimeval::perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Perf energy model of bit-serial PIM for reduction sum
pimeval::perfEnergy
pimPerfEnergyBitSerial::getPerfEnergyForRedSum(PimCmdEnum cmdType, const pimObjInfo& obj, unsigned numPass) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  PimDataType dataType = obj.getDataType();
  unsigned bitsPerElement = obj.getBitsPerElement();
  uint64_t numElements = obj.getNumElements();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned numCore = obj.getNumCoresUsed();
  double cpuTDP = 200; // W; AMD EPYC 9124 16 core

  switch (m_simTarget) {
    case PIM_DEVICE_BITSIMD_V:
    case PIM_DEVICE_BITSIMD_V_AP:
    {
      if (dataType == PIM_INT8 || dataType == PIM_INT16 || dataType == PIM_INT64 || dataType == PIM_INT32 || dataType == PIM_UINT8 || dataType == PIM_UINT16 || dataType == PIM_UINT32 || dataType == PIM_UINT64) {
        // Assume row-wide popcount capability for integer reduction, with a 64-bit popcount logic unit per PIM core
        // For a single row, popcount is calculated per 64-bit chunks, and result is shifted then added to an 64-bit accumulator register
        // If there are multiple regions per core, the multi-region reduction sum is stored in the accumulator
        double mjEnergyPerPcl = m_pclNsDelay * m_pclUwPower * 1e-12;
        int numPclPerCore = (maxElementsPerRegion + 63) / 64; // number of 64-bit popcount needed for a row
        msRuntime = m_tR + (m_pclNsDelay * 1e-6) * numPclPerCore;
        msRuntime *= bitsPerElement * numPass;
        mjEnergy = m_eAP * numCore + mjEnergyPerPcl * numPclPerCore * numCore; // energy of one row read and row-wide popcount
        mjEnergy *= bitsPerElement * numPass;
        // reduction for all regions
        double aggregateMs = static_cast<double>(numCore) / 3200000;
        msRuntime += aggregateMs;
        mjEnergy += aggregateMs * cpuTDP;
        mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      } else {
        assert(0);
      }
      break;
    }
    case PIM_DEVICE_SIMDRAM:
      // todo
      std::cout << "PIM-Warning: SIMDRAM performance stats not implemented yet." << std::endl;
      break;
    case PIM_DEVICE_BITSIMD_H:
      // Sequentially process all elements per CPU cycle
      msRuntime = static_cast<double>(numElements) / 3200000; // typical 3.2 GHz CPU
      mjEnergy = 999999999.9; // todo
      // consider PCL
      break;
    default:
      assert(0);
  }

  return pimeval::perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Perf energy model of bit-serial PIM for broadcast
pimeval::perfEnergy
pimPerfEnergyBitSerial::getPerfEnergyForBroadcast(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned numCore = obj.getNumCoresUsed();
  switch (m_simTarget) {
    case PIM_DEVICE_BITSIMD_V:
    case PIM_DEVICE_BITSIMD_V_AP:
    {
      // For one pass: For every bit: Set SA to bit value; Write SA to row;
      msRuntime = (m_tL + m_tW) * bitsPerElement;
      msRuntime *= numPass;
      mjEnergy = m_eAP * numCore * numPass ;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      break;
    }
    case PIM_DEVICE_SIMDRAM:
    {
      // todo
      msRuntime *= numPass;
      mjEnergy *= numPass;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      break;
    }
    case PIM_DEVICE_BITSIMD_H:
    {
      // For one pass: For every element: 1 tCCD per byte
      uint64_t maxBytesPerRegion = (uint64_t)maxElementsPerRegion * (bitsPerElement / 8);
      msRuntime = m_tW + m_tL * maxBytesPerRegion; // for one pass
      msRuntime *= numPass;
      mjEnergy = (m_eAP + (m_tL * maxBytesPerRegion)) * numCore * numPass;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      break;
    }
    default:
      assert(0);
  }

  return pimeval::perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Perf energy model of bit-serial PIM for rotate
pimeval::perfEnergy
pimPerfEnergyBitSerial::getPerfEnergyForRotate(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned numRegions = obj.getRegions().size();
  // boundary handling
  pimeval::perfEnergy perfEnergyBT = getPerfEnergyForBytesTransfer(cmdType, numRegions * bitsPerElement / 8);

  switch (m_simTarget) {
    case PIM_DEVICE_BITSIMD_V:
    case PIM_DEVICE_BITSIMD_V_AP:
      // rotate within subarray:
      // For every bit: Read row to SA; move SA to R1; Shift R1; Move R1 to SA; Write SA to row
      msRuntime = (m_tR + 3 * m_tL + m_tW) * bitsPerElement; // for one pass
      msRuntime *= numPass;
      mjEnergy = (m_eAP + 3 * m_eL) * bitsPerElement * numPass; // for one pass
      msRuntime += 2 * perfEnergyBT.m_msRuntime;
      mjEnergy += 2 * perfEnergyBT.m_mjEnergy;
      break;
    case PIM_DEVICE_SIMDRAM:
      // todo
      break;
    case PIM_DEVICE_BITSIMD_H:
      // rotate within subarray:
      // For every bit: Read row to SA; move SA to R1; Shift R1 by N steps; Move R1 to SA; Write SA to row
      // TODO: separate bank level and GDL
      // TODO: energy unimplemented
      msRuntime = (m_tR + (bitsPerElement + 2) * m_tL + m_tW); // for one pass
      msRuntime *= numPass;
      mjEnergy = (m_eAP + (bitsPerElement + 2) * m_eL) * numPass;
      msRuntime += 2 * perfEnergyBT.m_msRuntime;
      mjEnergy += 2 * perfEnergyBT.m_mjEnergy;
      break;
    default:
      assert(0);
  }

  return pimeval::perfEnergy(msRuntime, mjEnergy);
}

