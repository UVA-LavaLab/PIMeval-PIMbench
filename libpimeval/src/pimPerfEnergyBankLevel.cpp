// File: pimPerfEnergyBankLevel.cc
// PIMeval Simulator - Performance Energy Models
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimPerfEnergyBankLevel.h"
#include "pimCmd.h"
#include <iostream>


//! @brief  Perf energy model of bank-level PIM for func1
pimeval::perfEnergy
pimPerfEnergyBankLevel::getPerfEnergyForFunc1(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned numCores = obj.getNumCoresUsed();

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
      mjEnergy = (m_eAP * 2 + (m_eGDL * 2 + (maxElementsPerRegion * m_blimpLogicalEnergy * numberOfOperationPerElement))) * numCores * numPass;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
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
      mjEnergy = ((m_eAP * 2) + (m_eGDL * 2 + (maxElementsPerRegion * m_blimpLogicalEnergy * numberOfOperationPerElement))) * numCores * numPass ;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      break;
    }
    default:
      std::cout << "PIM-Warning: Perf energy model not available for PIM command " << pimCmd::getName(cmdType, "") << std::endl;
      break;
  }

  return pimeval::perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Perf energy model of bank-level PIM for func2
pimeval::perfEnergy
pimPerfEnergyBankLevel::getPerfEnergyForFunc2(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned numCoresUsed = obj.getNumCoresUsed();

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
      mjEnergy = ((m_eAP * 3) + (m_eGDL * 3 + (maxElementsPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement))) * numCoresUsed * numPass;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
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
      mjEnergy = ((m_eAP * 3) + (m_eGDL * 3 + (maxElementsPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement))) * numCoresUsed;
      mjEnergy += maxElementsPerRegion * numberOfOperationPerElement * m_blimpArithmeticEnergy * numCoresUsed;
      mjEnergy *= numPass;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
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
      mjEnergy = ((m_eAP * 3) + (m_eGDL * 3 + (maxElementsPerRegion * m_blimpLogicalEnergy * numberOfOperationPerElement))) * numCoresUsed;
      mjEnergy *= numPass;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      break;
    }
    default:
      std::cout << "PIM-Warning: Perf energy model not available for PIM command " << pimCmd::getName(cmdType, "") << std::endl;
      break;
  }

  return pimeval::perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Perf energy model of bank-level PIM for reduction sum
pimeval::perfEnergy
pimPerfEnergyBankLevel::getPerfEnergyForRedSum(PimCmdEnum cmdType, const pimObjInfo& obj, unsigned numPass) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned numCore = obj.getNumCoresUsed();
  double cpuTDP = 200; // W; AMD EPYC 9124 16 core

  // How many iteration require to read / write max elements per region
  double numberOfOperationPerElement = ((double)bitsPerElement / m_blimpCoreBitWidth);
  msRuntime = m_tR + m_tGDL + (maxElementsPerRegion * m_blimpCoreLatency * numberOfOperationPerElement * numPass);

  // Refer to fulcrum documentation
  mjEnergy = (m_eAP + (m_eGDL + (maxElementsPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement))) * numPass * numCore;
  // reduction for all regions
  double aggregateMs = static_cast<double>(numCore) / 3200000;
  msRuntime += aggregateMs;
  mjEnergy += aggregateMs * cpuTDP;
  mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;

  return pimeval::perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Perf energy model of bank-level PIM for broadcast
pimeval::perfEnergy
pimPerfEnergyBankLevel::getPerfEnergyForBroadcast(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned numCore = obj.getNumCoresUsed();

  // assume taking 1 ALU latency to write an element
  double numberOfOperationPerElement = ((double)bitsPerElement / m_blimpCoreBitWidth);
  msRuntime = m_tW + m_tGDL + (m_blimpCoreLatency * maxElementsPerRegion * numberOfOperationPerElement);
  msRuntime *= numPass;
  msRuntime = (m_eAP + (m_blimpCoreLatency * maxElementsPerRegion * numberOfOperationPerElement)) * numPass; // todo: change m_eR to write energy
  mjEnergy = (m_eAP + (m_eGDL + (maxElementsPerRegion * m_blimpLogicalEnergy * numberOfOperationPerElement))) * numPass * numCore;
  mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;

  return pimeval::perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Perf energy model of bank-level PIM for rotate
pimeval::perfEnergy
pimPerfEnergyBankLevel::getPerfEnergyForRotate(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned numRegions = obj.getRegions().size();
  // boundary handling
  pimeval::perfEnergy perfEnergyBT = getPerfEnergyForBytesTransfer(cmdType, numRegions * bitsPerElement / 8);

  // rotate within subarray:
  // For every bit: Read row to SA; move SA to R1; Shift R1 by N steps; Move R1 to SA; Write SA to row
  // TODO: separate bank level and GDL
  // TODO: energy unimplemented
  msRuntime = (m_tR + (bitsPerElement + 2) * m_tL + m_tW); // for one pass
  msRuntime *= numPass;
  mjEnergy = (m_eAP + (bitsPerElement + 2) * m_eL) * numPass;
  msRuntime += 2 * perfEnergyBT.m_msRuntime;
  mjEnergy += 2 * perfEnergyBT.m_mjEnergy;

  return pimeval::perfEnergy(msRuntime, mjEnergy);
}

