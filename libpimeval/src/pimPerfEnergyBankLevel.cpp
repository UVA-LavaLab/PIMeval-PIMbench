// File: pimPerfEnergyBankLevel.cc
// PIMeval Simulator - Performance Energy Models
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimPerfEnergyBankLevel.h"
#include "pimCmd.h"
#include <iostream>
#include <cmath>


//! @brief  Perf energy model of bank-level PIM for func1
pimeval::perfEnergy
pimPerfEnergyBankLevel::getPerfEnergyForFunc1(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement(PimBitWidth::ACTUAL);
  unsigned numCores = obj.isLoadBalanced() ? obj.getNumCoreAvailable() : obj.getNumCoresUsed();

  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  double numberOfOperationPerElement = ((double)bitsPerElement / m_blimpCoreBitWidth);
  unsigned minElementPerRegion = obj.isLoadBalanced() ? (std::ceil(obj.getNumElements() * 1.0 / numCores) - (maxElementsPerRegion * (numPass - 1))) : maxElementsPerRegion;
  // How many iteration require to read / write max elements per region
  unsigned maxGDLItr = maxElementsPerRegion * bitsPerElement / m_GDLWidth;
  unsigned minGDLItr = minElementPerRegion * bitsPerElement / m_GDLWidth;

  switch (cmdType)
  {
    case PimCmdEnum::COPY_O2O:
    {
      msRead = ((m_tR  + maxGDLItr * m_tGDL) * (numPass - 1)) + (m_tR + (minGDLItr * m_tGDL));
      msWrite = ((m_tW + maxGDLItr * m_tGDL) * (numPass - 1)) + (m_tW + (minGDLItr * m_tGDL));
      msCompute = 0;
      msRuntime = msRead + msWrite + msCompute;
      mjEnergy = numPass * numCores * ((m_eAP * 2) + (m_eGDL * 2));
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      break;
    }
    case PimCmdEnum::POPCOUNT:
    case PimCmdEnum::ABS:
    case PimCmdEnum::ADD_SCALAR:
    case PimCmdEnum::SUB_SCALAR:
    case PimCmdEnum::MUL_SCALAR:
    case PimCmdEnum::DIV_SCALAR:
    case PimCmdEnum::BIT_SLICE_EXTRACT:
    case PimCmdEnum::BIT_SLICE_INSERT:
    {
      if (cmdType == PimCmdEnum::BIT_SLICE_EXTRACT) {
        numberOfOperationPerElement *= 2; // 1 shift, 1 and
      } else if (cmdType == PimCmdEnum::BIT_SLICE_INSERT) {
        numberOfOperationPerElement *= 5; // 2 shifts, 1 not, 1 and, 1 or
      }
      // Refer to fulcrum documentation
      msRead = m_tR;
      msWrite = ((m_tW + maxGDLItr * m_tGDL) * (numPass - 1)) + (m_tW + (minGDLItr * m_tGDL));
      msCompute = (maxElementsPerRegion * m_blimpCoreLatency * numberOfOperationPerElement * (numPass - 1)) + (minElementPerRegion * m_blimpCoreLatency * numberOfOperationPerElement);
      msRuntime = msRead + msWrite + msCompute;
      mjEnergy = (m_eAP * 2 + (m_eGDL * 2 + (maxElementsPerRegion * m_blimpLogicalEnergy * numberOfOperationPerElement))) * numCores * (numPass - 1);
      mjEnergy += (m_eAP * 2 + (m_eGDL * 2 + (minElementPerRegion * m_blimpLogicalEnergy * numberOfOperationPerElement))) * numCores;
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
    case PimCmdEnum::NE_SCALAR:
    case PimCmdEnum::MIN_SCALAR:
    case PimCmdEnum::MAX_SCALAR:
    case PimCmdEnum::SHIFT_BITS_L:
    case PimCmdEnum::SHIFT_BITS_R:
    {
      msRead = m_tR;
      msWrite = ((m_tW + maxGDLItr * m_tGDL) * (numPass - 1)) + (m_tW + (minGDLItr * m_tGDL));
      msCompute = (maxElementsPerRegion * m_blimpCoreLatency * numberOfOperationPerElement * (numPass - 1)) + (minElementPerRegion * m_blimpCoreLatency * numberOfOperationPerElement);
      msRuntime = msRead + msWrite + msCompute;
      mjEnergy = ((m_eAP * 2) + (m_eGDL * 2 + (maxElementsPerRegion * m_blimpLogicalEnergy * numberOfOperationPerElement))) * numCores * (numPass - 1);
      mjEnergy += ((m_eAP * 2) + (m_eGDL * 2 + (minElementPerRegion * m_blimpLogicalEnergy * numberOfOperationPerElement))) * numCores;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      break;
    }
    default:
      std::cout << "PIM-Warning: Perf energy model not available for PIM command " << pimCmd::getName(cmdType, "") << std::endl;
      break;
  }

  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute);
}

//! @brief  Perf energy model of bank-level PIM for func2
pimeval::perfEnergy
pimPerfEnergyBankLevel::getPerfEnergyForFunc2(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement(PimBitWidth::ACTUAL);
  unsigned numCoresUsed = obj.isLoadBalanced() ? obj.getNumCoreAvailable() : obj.getNumCoresUsed();

  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  double numberOfOperationPerElement = ((double)bitsPerElement / m_blimpCoreBitWidth);
  unsigned minElementPerRegion = obj.isLoadBalanced() ? (std::ceil(obj.getNumElements() * 1.0 / numCoresUsed) - (maxElementsPerRegion * (numPass - 1))) : maxElementsPerRegion;
  // How many iteration require to read / write max elements per region
  unsigned maxGDLItr = maxElementsPerRegion * bitsPerElement / m_GDLWidth;
  unsigned minGDLItr = minElementPerRegion * bitsPerElement / m_GDLWidth;

  switch (cmdType)
  {
    case PimCmdEnum::ADD:
    case PimCmdEnum::SUB:
    case PimCmdEnum::MUL:
    case PimCmdEnum::DIV:
    {
      msRead = ((2 * m_tR) + (maxGDLItr * m_tGDL)) * (numPass - 1) + ((2 * m_tR) + (minGDLItr * m_tGDL));
      msWrite = (m_tW + (maxGDLItr * m_tGDL)) * (numPass - 1) + (m_tW + (minGDLItr * m_tGDL));
      msCompute = (maxElementsPerRegion * m_blimpCoreLatency * numberOfOperationPerElement * (numPass - 1)) + (minElementPerRegion * m_blimpCoreLatency * numberOfOperationPerElement);
      msRuntime = msRead + msWrite + msCompute;
      mjEnergy = ((m_eAP * 3) + (m_eGDL * 3 + (maxElementsPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement))) * numCoresUsed * (numPass - 1);
      mjEnergy = ((m_eAP * 3) + (m_eGDL * 3 + (minElementPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement))) * numCoresUsed;
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
      msRead = m_tR + (maxGDLItr * m_tGDL);
      msWrite = (m_tW + (maxGDLItr * m_tGDL)) * (numPass - 1) + (m_tW + (minGDLItr * m_tGDL));
      msCompute = (maxElementsPerRegion * m_blimpCoreLatency * numberOfOperationPerElement * 2 * (numPass - 1)) + (minElementPerRegion * m_blimpCoreLatency * numberOfOperationPerElement * 2);
      msRuntime = msRead + msWrite + msCompute;
      mjEnergy = ((m_eAP * 3) + (m_eGDL * 3 + (maxElementsPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement * 2))) * numCoresUsed * (numPass - 1);
      mjEnergy = ((m_eAP * 3) + (m_eGDL * 3 + (minElementPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement * 2))) * numCoresUsed;
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
    case PimCmdEnum::NE:
    case PimCmdEnum::MIN:
    case PimCmdEnum::MAX:
    {
      msRead = ((2 * m_tR) + (maxGDLItr * m_tGDL)) * (numPass - 1) + ((2 * m_tR) + (minGDLItr * m_tGDL));
      msWrite = (m_tW + (maxGDLItr * m_tGDL)) * (numPass - 1) + (m_tW + (minGDLItr * m_tGDL));
      msCompute = (maxElementsPerRegion * m_blimpCoreLatency * numberOfOperationPerElement * (numPass - 1)) + (minElementPerRegion * m_blimpCoreLatency * numberOfOperationPerElement);
      msRuntime = msRead + msWrite + msCompute;
      mjEnergy = ((m_eAP * 3) + (m_eGDL * 3 + (maxElementsPerRegion * m_blimpLogicalEnergy * numberOfOperationPerElement))) * numCoresUsed * (numPass - 1);
      mjEnergy += ((m_eAP * 3) + (m_eGDL * 3 + (minElementPerRegion * m_blimpLogicalEnergy * numberOfOperationPerElement))) * numCoresUsed;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      break;
    }
    default:
      std::cout << "PIM-Warning: Perf energy model not available for PIM command " << pimCmd::getName(cmdType, "") << std::endl;
      break;
  }

  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute);
}

//! @brief  Perf energy model of bank-level PIM for reduction sum
pimeval::perfEnergy
pimPerfEnergyBankLevel::getPerfEnergyForReduction(PimCmdEnum cmdType, const pimObjInfo& obj, unsigned numPass) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  unsigned bitsPerElement = obj.getBitsPerElement(PimBitWidth::ACTUAL);
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned numCore = obj.isLoadBalanced() ? obj.getNumCoreAvailable() : obj.getNumCoresUsed();
  double cpuTDP = 200; // W; AMD EPYC 9124 16 core
  unsigned minElementPerRegion = obj.isLoadBalanced() ? (std::ceil(obj.getNumElements() * 1.0 / numCore) - (maxElementsPerRegion * (numPass - 1))) : maxElementsPerRegion;
 
  switch (cmdType) {
    case PimCmdEnum::REDSUM:
    case PimCmdEnum::REDSUM_RANGE:
    case PimCmdEnum::REDMIN:
    case PimCmdEnum::REDMIN_RANGE:
    case PimCmdEnum::REDMAX:
    case PimCmdEnum::REDMAX_RANGE:
    {
      // How many iteration require to read / write max elements per region
      double numberOfOperationPerElement = ((double)bitsPerElement / m_blimpCoreBitWidth);
      msRead = m_tR + m_tGDL;
      // reduction for all regions assuming 16 core AMD EPYC 9124
      double aggregateMs = static_cast<double>(numCore) / (3200000 * 16);
      msCompute = (maxElementsPerRegion * m_blimpCoreLatency * numberOfOperationPerElement * (numPass - 1)) + (minElementPerRegion * m_blimpCoreLatency * numberOfOperationPerElement) + aggregateMs;
      msRuntime = msRead + msWrite + msCompute;

      // Refer to fulcrum documentation
      mjEnergy = (m_eAP + (m_eGDL + (maxElementsPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement))) * (numPass - 1) * numCore;
      mjEnergy += (m_eAP + (m_eGDL + (minElementPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement))) * numCore;
      mjEnergy += aggregateMs * cpuTDP;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      break;
    }
    default:
      std::cout << "PIM-Warning: Unsupported reduction command for bank-level PIM: " 
                << pimCmd::getName(cmdType, "") << std::endl;
      break;
    }

  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute);
}

//! @brief  Perf energy model of bank-level PIM for broadcast
pimeval::perfEnergy
pimPerfEnergyBankLevel::getPerfEnergyForBroadcast(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned numCore = obj.isLoadBalanced() ? obj.getNumCoreAvailable() : obj.getNumCoresUsed();
  unsigned bitsPerElement = obj.getBitsPerElement(PimBitWidth::ACTUAL);
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned minElementPerRegion = obj.isLoadBalanced() ? (std::ceil(obj.getNumElements() * 1.0 / obj.getNumCoreAvailable()) - (maxElementsPerRegion * (numPass - 1))) : maxElementsPerRegion;
   // How many iteration require to read / write max elements per region
  unsigned maxGDLItr = maxElementsPerRegion * bitsPerElement / m_GDLWidth;
  unsigned minGDLItr = minElementPerRegion * bitsPerElement / m_GDLWidth;
  msWrite = (m_tW + (maxGDLItr * m_tGDL)) * (numPass - 1) + (m_tW + (minGDLItr * m_tGDL));
  msRuntime = msRead + msWrite + msCompute;
  mjEnergy = (m_eAP + m_eGDL) * numPass * numCore;
  mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;

  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute);
}

// TODO: This needs to be revisited
//! @brief  Perf energy model of bank-level PIM for rotate
pimeval::perfEnergy
pimPerfEnergyBankLevel::getPerfEnergyForRotate(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement(PimBitWidth::ACTUAL);
  unsigned numRegions = obj.getRegions().size();
  // boundary handling - assume two times copying between device and host for boundary elements
  pimeval::perfEnergy perfEnergyBT = getPerfEnergyForBytesTransfer(PimCmdEnum::COPY_D2H, numRegions * bitsPerElement / 8);

  // rotate within subarray:
  // For every bit: Read row to SA; move SA to R1; Shift R1 by N steps; Move R1 to SA; Write SA to row
  // TODO: separate bank level and GDL
  // TODO: energy unimplemented
  msRuntime = (m_tR + (bitsPerElement + 2) * m_tL + m_tW); // for one pass
  msRuntime *= numPass;
  mjEnergy = (m_eAP + (bitsPerElement + 2) * m_eL) * numPass;
  msRuntime += 2 * perfEnergyBT.m_msRuntime;
  mjEnergy += 2 * perfEnergyBT.m_mjEnergy;
  std::cout << "PIM-Warning: Perf energy model is not precise for PIM command " << pimCmd::getName(cmdType, "") << std::endl;

  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute);
}

