// File: pimParamsPerf.cc
// PIMeval Simulator - Performance parameters
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimPerfEnergyModels.h"
#include "pimSim.h"
#include "pimCmd.h"
#include "pimPerfEnergyTables.h"
#include <cstdio>

//! @brief  pimParamsPerf ctor
pimParamsPerf::pimParamsPerf(pimParamsDram* paramsDram)
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

//! @brief  Get ms runtime for bytes transferred between host and device
pimParamsPerf::perfEnergy
pimParamsPerf::getPerfEnergyForBytesTransfer(PimCmdEnum cmdType, uint64_t numBytes) const
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
  return  perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Get ms runtime for bit-serial PIM devices
//!         BitSIMD and SIMDRAM need different fields
pimParamsPerf::perfEnergy
pimParamsPerf::getPerfEnergyBitSerial(PimDeviceEnum deviceType, PimCmdEnum cmdType, PimDataType dataType, unsigned bitsPerElement, unsigned numPass, const pimObjInfo& obj) const
{
  bool ok = false;
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned numRanks = pimSim::get()->getNumRanks();

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
          mjEnergy += ((m_eL * numL * obj.getMaxElementsPerRegion()) + (m_eAP * numR + m_eAP * numW)) * m_numChipsPerRank * numRanks;
          mjEnergy += m_pBChip * m_numChipsPerRank * numRanks * msRuntime;
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
    std::printf("PIM-Warning: Unimplemented bit-serial runtime estimation for device=%s cmd=%s dataType=%s\n",
            pimUtils::pimDeviceEnumToStr(deviceType).c_str(),
            pimCmd::getName(cmdType, "").c_str(),
            pimUtils::pimDataTypeEnumToStr(dataType).c_str());
    msRuntime = 1000000;
  }
  msRuntime *= numPass;
  mjEnergy *= numPass;

  return perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Get ms runtime for func1
pimParamsPerf::perfEnergy
pimParamsPerf::getPerfEnergyForFunc1(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  PimDeviceEnum simTarget = pimSim::get()->getSimTarget();
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned numCores = obj.getNumCoresUsed();
  unsigned numRanks = pimSim::get()->getNumRanks();
  PimDataType dataType = obj.getDataType();
  switch (simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_AP:
  case PIM_DEVICE_BITSIMD_H:
  case PIM_DEVICE_SIMDRAM:
  {
    pimParamsPerf::perfEnergy perfEnergyBS = getPerfEnergyBitSerial(simTarget, cmdType, dataType, bitsPerElement, numPass, obj);
    msRuntime += perfEnergyBS.m_msRuntime;
    mjEnergy += perfEnergyBS.m_mjEnergy;
    break;
  }
  case PIM_DEVICE_FULCRUM:
  {
    // Fulcrum utilizes three walkers: two for input operands and one for the output operand.
    // For instructions that operate on a single operand, the next operand is fetched by the walker.
    // Consequently, only one row read operation is required in this case.
    // Additionally, using the walker-renaming technique (refer to the Fulcrum paper for details),
    // the write operation is also pipelined. Thus, only one row write operation is needed.

    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double numberOfALUOperationPerElement = ((double)bitsPerElement / m_flucrumAluBitWidth);
    switch (cmdType)
    {
    case PimCmdEnum::POPCOUNT:
    {
      numberOfALUOperationPerElement *= 12; // 4 shifts, 4 ands, 3 add/sub, 1 mul
      msRuntime = m_tR + m_tW + (maxElementsPerRegion * m_fulcrumAluLatency * numberOfALUOperationPerElement * numPass);
      double energyArithmetic = ((maxElementsPerRegion - 1) * 2 *  m_fulcrumShiftEnergy) + ((maxElementsPerRegion) * m_fulcrumALUArithmeticEnergy * 4);
      double energyLogical = ((maxElementsPerRegion - 1) * 2 *  m_fulcrumShiftEnergy) + ((maxElementsPerRegion) * m_fulcrumALULogicalEnergy * 8);
      mjEnergy = ((energyArithmetic + energyLogical) + m_eAP) * numCores * numPass;
      mjEnergy += m_pBChip * m_numChipsPerRank * numRanks * msRuntime;
      break;
    }
    case PimCmdEnum::ADD_SCALAR:
    case PimCmdEnum::SUB_SCALAR:
    case PimCmdEnum::MUL_SCALAR:
    case PimCmdEnum::DIV_SCALAR:
    case PimCmdEnum::ABS:
    {
      msRuntime = m_tR + m_tW + (maxElementsPerRegion * m_fulcrumAluLatency * numberOfALUOperationPerElement * numPass);
      mjEnergy = numPass * numCores * ((m_eAP * 2) + ((maxElementsPerRegion - 1) * 2 *  m_fulcrumShiftEnergy) + ((maxElementsPerRegion) * m_fulcrumALUArithmeticEnergy * numberOfALUOperationPerElement));
      mjEnergy += m_pBChip * m_numChipsPerRank * numRanks * msRuntime;
      break;
    }
    case PimCmdEnum::AND_SCALAR:
    case PimCmdEnum::OR_SCALAR:
    case PimCmdEnum::XOR_SCALAR:
    case PimCmdEnum::XNOR_SCALAR:
    case PimCmdEnum::GT_SCALAR:
    case PimCmdEnum::LT_SCALAR:
    case PimCmdEnum::EQ_SCALAR:
    case PimCmdEnum::MIN_SCALAR:
    case PimCmdEnum::MAX_SCALAR:
    case PimCmdEnum::SHIFT_BITS_L:
    case PimCmdEnum::SHIFT_BITS_R:
    {
      msRuntime = m_tR + m_tW + (maxElementsPerRegion * m_fulcrumAluLatency * numberOfALUOperationPerElement * numPass);
      mjEnergy = numPass * numCores * ((m_eAP * 2) + ((maxElementsPerRegion - 1) * 2 *  m_fulcrumShiftEnergy) + ((maxElementsPerRegion) * m_fulcrumALULogicalEnergy * numberOfALUOperationPerElement));
      mjEnergy += m_pBChip * m_numChipsPerRank * numRanks * msRuntime;
      break;
    }
    default:
      std::printf("PIM-Warning: Perf energy model not available for PIM command %s\n", pimCmd::getName(cmdType, "").c_str());
      break;
    }
    break;
  }
  case PIM_DEVICE_BANK_LEVEL:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double numberOfOperationPerElement = ((double)bitsPerElement / m_blimpCoreBitWidth);
    switch (cmdType)
    {
    case PimCmdEnum::POPCOUNT:
    case PimCmdEnum::ABS:
    case PimCmdEnum::ADD_SCALAR:
    case PimCmdEnum::SUB_SCALAR:
    case PimCmdEnum::MUL_SCALAR:
    case PimCmdEnum::DIV_SCALAR:
    {
      // How many iteration require to read / write max elements per region
      unsigned numGDLItr = maxElementsPerRegion * bitsPerElement / m_GDLWidth;
      double totalGDLOverhead = m_tGDL * numGDLItr; // read can be pipelined and write cannot be pipelined
      // Refer to fulcrum documentation
      msRuntime = m_tR + m_tW + totalGDLOverhead + (maxElementsPerRegion * m_blimpCoreLatency * numberOfOperationPerElement * numPass);
      mjEnergy = (m_eAP * 2 + (m_eGDL * numGDLItr * 2 + (maxElementsPerRegion * m_blimpLogicalEnergy * numberOfOperationPerElement))) * numCores * numPass;
      mjEnergy += m_pBChip * m_numChipsPerRank * numRanks * msRuntime;
      break;
    }
    case PimCmdEnum::AND_SCALAR:
    case PimCmdEnum::OR_SCALAR:
    case PimCmdEnum::XOR_SCALAR:
    case PimCmdEnum::XNOR_SCALAR:
    case PimCmdEnum::GT_SCALAR:
    case PimCmdEnum::LT_SCALAR:
    case PimCmdEnum::EQ_SCALAR:
    case PimCmdEnum::MIN_SCALAR:
    case PimCmdEnum::MAX_SCALAR:
    case PimCmdEnum::SHIFT_BITS_L:
    case PimCmdEnum::SHIFT_BITS_R:
    {
      // How many iteration require to read / write max elements per region
      unsigned numGDLItr = maxElementsPerRegion * bitsPerElement / m_GDLWidth;
      double totalGDLOverhead = m_tGDL * numGDLItr; // read can be pipelined and write cannot be pipelined
      // Refer to fulcrum documentation
      msRuntime = m_tR + m_tW + totalGDLOverhead + (maxElementsPerRegion * m_blimpCoreLatency * numberOfOperationPerElement * numPass);
      mjEnergy = ((m_eAP * 2) + (m_eGDL * numGDLItr * 2 + (maxElementsPerRegion * m_blimpLogicalEnergy * numberOfOperationPerElement))) * numCores * numPass ;
      mjEnergy += m_pBChip * m_numChipsPerRank * numRanks * msRuntime;
      break;
    }
    default:
      std::printf("PIM-Warning: Perf energy model not available for PIM command %s\n", pimCmd::getName(cmdType, "").c_str());
      break;
    }
    break;
  }
  default:
  {
    msRuntime = 1000000;
    mjEnergy = 999999999.9;
  }
  break;
  }
  return perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Get ms runtime for func2
pimParamsPerf::perfEnergy
pimParamsPerf::getPerfEnergyForFunc2(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  PimDeviceEnum simTarget = pimSim::get()->getSimTarget();
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned numRanks = pimSim::get()->getNumRanks();
  PimDataType dataType = obj.getDataType();
  unsigned numCoresUsed = obj.getNumCoresUsed();
  switch (simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_AP:
  case PIM_DEVICE_BITSIMD_H:
  case PIM_DEVICE_SIMDRAM:
  {
    pimParamsPerf::perfEnergy perfEnergyBS = getPerfEnergyBitSerial(simTarget, cmdType, dataType, bitsPerElement, numPass, obj);
    msRuntime += perfEnergyBS.m_msRuntime;
    mjEnergy += perfEnergyBS.m_mjEnergy;
    break;
  }
  case PIM_DEVICE_FULCRUM:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double numberOfALUOperationPerElement = ((double)bitsPerElement / m_flucrumAluBitWidth);
    switch (cmdType)
    {
    case PimCmdEnum::ADD:
    case PimCmdEnum::SUB:
    case PimCmdEnum::MUL:
    case PimCmdEnum::DIV:
    {
      msRuntime = 2 * m_tR + m_tW + maxElementsPerRegion * numberOfALUOperationPerElement * m_fulcrumAluLatency;
      msRuntime *= numPass;
      mjEnergy = numCoresUsed * numPass * ((m_eAP * 3) + ((maxElementsPerRegion - 1) * 3 *  m_fulcrumShiftEnergy) + ((maxElementsPerRegion) * m_fulcrumALUArithmeticEnergy * numberOfALUOperationPerElement));
      mjEnergy += m_pBChip * m_numChipsPerRank * numRanks * msRuntime;
      break;
    }
    case PimCmdEnum::SCALED_ADD:
    {
      /**
       * Performs a multiply-add operation on rows in DRAM.
       *
       * This command executes the following steps:
       * 1. Multiply the elements of a source row by a scalar value.
       * 2. Add the result of the multiplication to the elements of another row.
       * 3. Write the final result back to a row in DRAM.
       *
       * Performance Optimizations:
       * - While performing the multiplication, the next row to be added can be fetched without any additional overhead.
       * - During the addition, the next row to be multiplied can be fetched concurrently.
       * - Total execution time for one region of multiplication and addition >>>> reading/writing three DRAM rows as a result using walker renaming, row write is also pipelined
       *
       * As a result, only one read operation and one write operation is necessary for the entire pass.
      */
      msRuntime = m_tR + m_tW + (maxElementsPerRegion * numberOfALUOperationPerElement * m_fulcrumAluLatency * 2) * numPass;
      mjEnergy = numCoresUsed * numPass * ((m_eAP * 3) + ((maxElementsPerRegion - 1) * 3 *  m_fulcrumShiftEnergy) + ((maxElementsPerRegion) * m_fulcrumALUArithmeticEnergy * numberOfALUOperationPerElement));
      mjEnergy += m_pBChip * m_numChipsPerRank * numRanks * msRuntime;
      break;
    }
    case PimCmdEnum::AND:
    case PimCmdEnum::OR:
    case PimCmdEnum::XOR:
    case PimCmdEnum::XNOR:
    case PimCmdEnum::GT:
    case PimCmdEnum::LT:
    case PimCmdEnum::EQ:
    case PimCmdEnum::MIN:
    case PimCmdEnum::MAX:
    {
      msRuntime = 2 * m_tR + m_tW + maxElementsPerRegion * numberOfALUOperationPerElement * m_fulcrumAluLatency;
      msRuntime *= numPass;
      mjEnergy = numCoresUsed * numPass * (((maxElementsPerRegion - 1) * 3 *  m_fulcrumShiftEnergy) + ((maxElementsPerRegion) * m_fulcrumALULogicalEnergy * numberOfALUOperationPerElement));
      mjEnergy += m_eAP * 3 * m_numChipsPerRank * numRanks;
      mjEnergy += m_pBChip * m_numChipsPerRank * numRanks * msRuntime;
      break;
    }
    default:
      std::printf("PIM-Warning: Perf energy model not available for PIM command %s\n", pimCmd::getName(cmdType, "").c_str());
      break;
    }
    break;
  }
  case PIM_DEVICE_BANK_LEVEL:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double numberOfOperationPerElement = ((double)bitsPerElement / m_blimpCoreBitWidth);
    // How many iteration require to read / write max elements per region
    unsigned numGDLItr = maxElementsPerRegion * bitsPerElement / m_GDLWidth;

    switch (cmdType)
    {
    case PimCmdEnum::ADD:
    case PimCmdEnum::SUB:
    case PimCmdEnum::MUL:
    case PimCmdEnum::DIV:
    {
      double totalGDLOverhead = m_tGDL * numGDLItr * 2; // one read can be pipelined
      msRuntime = 2 * m_tR + m_tW + totalGDLOverhead + maxElementsPerRegion * m_blimpCoreLatency * numberOfOperationPerElement;
      msRuntime *= numPass;
      mjEnergy = ((m_eAP * 3) + (m_eGDL * numGDLItr * 3 + (maxElementsPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement))) * numCoresUsed * numPass;
      mjEnergy += m_pBChip * m_numChipsPerRank * numRanks * msRuntime;
      break;
    }
    case PimCmdEnum::SCALED_ADD:
    {
      /**
       * Performs a multiply-add operation on rows in DRAM.
       *
       * This command executes the following steps:
       * 1. Multiply the elements of a source row by a scalar value.
       * 2. Add the result of the multiplication to the elements of another row.
       * 3. Write the final result back to a row in DRAM.
       *
       * Performance Optimizations:
       * - While performing the multiplication, the next row to be added can be fetched without any additional overhead.
       * - During the addition, the next row to be multiplied can be fetched concurrently.
       *
       * As a result, only one read operation is necessary for the entire pass.
      */
      double totalGDLOverhead = m_tGDL * numGDLItr; // both read can be pipelined as multiplication and addition takes twice the time to execute.
      msRuntime = m_tR + (m_tW + totalGDLOverhead + maxElementsPerRegion * m_blimpCoreLatency * numberOfOperationPerElement * 2) * numPass;
      mjEnergy = ((m_eAP * 3) + (m_eGDL * numGDLItr * 3 + (maxElementsPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement))) * numCoresUsed;
      mjEnergy += maxElementsPerRegion * numberOfOperationPerElement * m_blimpArithmeticEnergy * numCoresUsed;
      mjEnergy *= numPass;
      mjEnergy += m_pBChip * m_numChipsPerRank * numRanks * msRuntime;
      break;
    }
    case PimCmdEnum::AND:
    case PimCmdEnum::OR:
    case PimCmdEnum::XOR:
    case PimCmdEnum::XNOR:
    case PimCmdEnum::GT:
    case PimCmdEnum::LT:
    case PimCmdEnum::EQ:
    case PimCmdEnum::MIN:
    case PimCmdEnum::MAX:
    {
      double totalGDLOverhead = m_tGDL * numGDLItr * 2; // one read can be pipelined
      msRuntime = 2 * m_tR + m_tW + totalGDLOverhead + maxElementsPerRegion * m_blimpCoreLatency * numberOfOperationPerElement;
      msRuntime *= numPass;
      mjEnergy = ((m_eAP * 3) + (m_eGDL * numGDLItr * 3 + (maxElementsPerRegion * m_blimpLogicalEnergy * numberOfOperationPerElement))) * numCoresUsed;
      mjEnergy *= numPass;
      mjEnergy += m_pBChip * m_numChipsPerRank * numRanks * msRuntime;
      break;
    }
    default:
      std::printf("PIM-Warning: Perf energy model not available for PIM command %s\n", pimCmd::getName(cmdType, "").c_str());
      break;
    }
    break;
  }
  default:
    msRuntime = 1e10;
    mjEnergy = 999999999.9;
  }

  return perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Get ms runtime for reduction sum
pimParamsPerf::perfEnergy
pimParamsPerf::getPerfEnergyForRedSum(PimCmdEnum cmdType, const pimObjInfo& obj, unsigned numPass) const
{
  PimDeviceEnum simTarget = pimSim::get()->getSimTarget();
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  PimDataType dataType = obj.getDataType();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned numRegions = obj.getRegions().size();
  uint64_t numElements = obj.getNumElements();
  unsigned numRanks = pimSim::get()->getNumRanks();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned numCore = obj.getNumCoresUsed();
  double cpuTDP = 200; // W; AMD EPYC 9124 16 core

  switch (simTarget) {
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
      mjEnergy += m_pBChip * m_numChipsPerRank * numRanks * msRuntime;
    } else {
      assert(0);
    }
    break;
  }
  case PIM_DEVICE_SIMDRAM:
    // todo
    std::printf("PIM-Warning: SIMDRAM performance stats not implemented yet.\n");
    break;
  case PIM_DEVICE_BITSIMD_H:
    // Sequentially process all elements per CPU cycle
    msRuntime = static_cast<double>(numElements) / 3200000; // typical 3.2 GHz CPU
    mjEnergy = 999999999.9; // todo
    // consider PCL
    break;
  case PIM_DEVICE_FULCRUM:
  {
    // read a row to walker, then reduce in serial
    double numberOfOperationPerElement = ((double)bitsPerElement / m_flucrumAluBitWidth);
    msRuntime = m_tR + (maxElementsPerRegion * m_fulcrumAluLatency * numberOfOperationPerElement * numPass);
    mjEnergy = numPass * numCore * (m_eAP * ((maxElementsPerRegion - 1) *  m_fulcrumShiftEnergy) + ((maxElementsPerRegion) * m_fulcrumALUArithmeticEnergy * numberOfOperationPerElement));
    // reduction for all regions
    double aggregateMs = static_cast<double>(numCore) / 3200000;
    msRuntime += aggregateMs;
    mjEnergy += aggregateMs * cpuTDP;
    mjEnergy += m_pBChip * m_numChipsPerRank * numRanks * msRuntime;
    break;
  }
  case PIM_DEVICE_BANK_LEVEL:
  {
    // How many iteration require to read / write max elements per region
    unsigned numGDLItr = maxElementsPerRegion * bitsPerElement / m_GDLWidth;
    double numberOfOperationPerElement = ((double)bitsPerElement / m_blimpCoreBitWidth);
    msRuntime = m_tR + m_tGDL + (maxElementsPerRegion * m_blimpCoreLatency * numberOfOperationPerElement * numPass);

    // Refer to fulcrum documentation
    mjEnergy = (m_eAP * 1 + (m_eGDL * numGDLItr + (maxElementsPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement))) * numPass * numCore;
    // reduction for all regions
    double aggregateMs = static_cast<double>(numCore) / 3200000;
    msRuntime += aggregateMs;
    mjEnergy += aggregateMs * cpuTDP;
    mjEnergy += m_pBChip * m_numChipsPerRank * numRanks * msRuntime;
    break;
  }
  default:
    msRuntime = 1e10;
    mjEnergy = 999999999.9; // todo
  }

  return perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Get ms runtime for broadcast
pimParamsPerf::perfEnergy
pimParamsPerf::getPerfEnergyForBroadcast(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  PimDeviceEnum simTarget = pimSim::get()->getSimTarget();
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned numRanks = pimSim::get()->getNumRanks();
  unsigned numCore = obj.getNumCoresUsed();
  switch (simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_AP:
  {
    // For one pass: For every bit: Set SA to bit value; Write SA to row;
    msRuntime = (m_tL + m_tW) * bitsPerElement;
    msRuntime *= numPass;
    mjEnergy = m_eAP * numCore * numPass ;
    mjEnergy += m_pBChip * m_numChipsPerRank * numRanks * msRuntime;
    break;
  }
  case PIM_DEVICE_SIMDRAM:
  {
    // todo
    msRuntime *= numPass;
    mjEnergy *= numPass;
    mjEnergy += m_pBChip * m_numChipsPerRank * numRanks * msRuntime;
    break;
  }
  case PIM_DEVICE_BITSIMD_H:
  {
    // For one pass: For every element: 1 tCCD per byte
    uint64_t maxBytesPerRegion = (uint64_t)maxElementsPerRegion * (bitsPerElement / 8);
    msRuntime = m_tW + m_tL * maxBytesPerRegion; // for one pass
    msRuntime *= numPass;
    mjEnergy = (m_eAP + (m_tL * maxBytesPerRegion)) * numCore * numPass;
    mjEnergy += m_pBChip * m_numChipsPerRank * numRanks * msRuntime;
    break;
  }
  case PIM_DEVICE_FULCRUM:
  {
    // assume taking 1 ALU latency to write an element
    double numberOfOperationPerElement = ((double)bitsPerElement / m_flucrumAluBitWidth);
    msRuntime = m_tW + m_fulcrumAluLatency * maxElementsPerRegion * numberOfOperationPerElement;
    msRuntime *= numPass;
    mjEnergy = numPass * numCore * (m_eAP + ((maxElementsPerRegion - 1) *  m_fulcrumShiftEnergy) + ((maxElementsPerRegion) * m_fulcrumALULogicalEnergy * numberOfOperationPerElement));
    mjEnergy += m_pBChip * m_numChipsPerRank * numRanks * msRuntime;
    break;
  }
  case PIM_DEVICE_BANK_LEVEL:
  {
    // assume taking 1 ALU latency to write an element
    double numberOfOperationPerElement = ((double)bitsPerElement / m_blimpCoreBitWidth);
    msRuntime = m_tW + m_tGDL + (m_blimpCoreLatency * maxElementsPerRegion * numberOfOperationPerElement);
    msRuntime *= numPass;
    msRuntime = (m_eAP + (m_blimpCoreLatency * maxElementsPerRegion * numberOfOperationPerElement)) * numPass; // todo: change m_eR to write energy
    unsigned numGDLItr = maxElementsPerRegion * bitsPerElement / m_GDLWidth;
    mjEnergy = (m_eAP + (m_eGDL * numGDLItr + (maxElementsPerRegion * m_blimpLogicalEnergy * numberOfOperationPerElement))) * numPass * numCore;
    mjEnergy += m_pBChip * m_numChipsPerRank * numRanks * msRuntime;
    break;
  }
  default:
    msRuntime = 1e10;
    mjEnergy = 999999999.9;
  }

  return perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Get ms runtime for rotate
pimParamsPerf::perfEnergy
pimParamsPerf::getPerfEnergyForRotate(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  PimDeviceEnum simTarget = pimSim::get()->getSimTarget();
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned numRegions = obj.getRegions().size();
  // boundary handling
  pimParamsPerf::perfEnergy perfEnergyBT = getPerfEnergyForBytesTransfer(cmdType, numRegions * bitsPerElement / 8);
  switch (simTarget) {
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
  case PIM_DEVICE_FULCRUM:
  case PIM_DEVICE_BANK_LEVEL:
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
    msRuntime = 1e10;
    mjEnergy = 999999999.9; // todo
  }

  return perfEnergy(msRuntime, mjEnergy);
}

