// File: pimPerfEnergyAquabolt.cc
// PIMeval Simulator - Performance Energy Models
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimPerfEnergyAquabolt.h"
#include "pimCmd.h"
#include <iostream>
#include <cmath>

//! @brief  Perf energy model of aquabolt PIM for func1
pimeval::perfEnergy
pimPerfEnergyAquabolt::getPerfEnergyForFunc1(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned numCores = obj.getNumCoresUsed();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  double numberOfOperationPerElement = ((double)bitsPerElement / m_aquaboltFPUBitWidth);
  unsigned minElementPerRegion = std::ceil(obj.getNumElements() * 1.0 / obj.getNumCoreAvailable()) - (maxElementsPerRegion * (numPass - 1));
  unsigned maxGDLItr = maxElementsPerRegion * bitsPerElement / m_GDLWidth;
  unsigned minGDLItr = minElementPerRegion * bitsPerElement / m_GDLWidth;
  switch (cmdType)
  {
    case PimCmdEnum::POPCOUNT:
    case PimCmdEnum::ABS:
    case PimCmdEnum::ADD_SCALAR:
    case PimCmdEnum::SUB_SCALAR:
    case PimCmdEnum::MUL_SCALAR:
    case PimCmdEnum::DIV_SCALAR:
    { 
      // multiplying by 2 as both read and write needs to consider GDL overhead; Cannot be pipeline because two rows cannot be open simultaeneously
      msRuntime = (m_tR + m_tW + (maxGDLItr * m_tGDL * numberOfOperationPerElement * 2)) * (numPass - 1);
      msRuntime += (m_tR + m_tW + (minGDLItr * m_tGDL * numberOfOperationPerElement * 2));
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
      // multiplying by 2 as both read and write needs to consider GDL overhead; Cannot be pipeline because two rows cannot be open simultaeneously
      msRuntime = (m_tR + m_tW + (maxGDLItr * m_tGDL * numberOfOperationPerElement * 2)) * (numPass - 1);
      msRuntime += (m_tR + m_tW + (minGDLItr * m_tGDL * numberOfOperationPerElement * 2));
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

//! @brief  Perf energy model of aquabolt PIM for func2
pimeval::perfEnergy
pimPerfEnergyAquabolt::getPerfEnergyForFunc2(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned numCoresUsed = obj.getNumCoresUsed();

  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned minElementPerRegion = std::ceil(obj.getNumElements() * 1.0 / obj.getNumCoreAvailable()) - (maxElementsPerRegion * (numPass - 1));
  unsigned maxGDLItr = std::ceil(maxElementsPerRegion * bitsPerElement * 1.0 / m_GDLWidth);
  unsigned minGDLItr = std::ceil(minElementPerRegion * bitsPerElement * 1.0 / m_GDLWidth);
  switch (cmdType)
  {
    case PimCmdEnum::ADD:
    case PimCmdEnum::SUB:
    case PimCmdEnum::MUL:
    case PimCmdEnum::DIV:
    {
      double numberOfOperationPerElement = ((double)bitsPerElement / (m_aquaboltFPUBitWidth));
      msRuntime = (2 * m_tR + m_tW + (maxGDLItr * m_tGDL * numberOfOperationPerElement * 3)) * (numPass - 1);
      msRuntime += (2 * m_tR + m_tW + (minGDLItr * m_tGDL * numberOfOperationPerElement * 3));
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
      double numberOfOperationPerElement = ((double)bitsPerElement / (m_aquaboltFPUBitWidth * m_aquaboltFPUUnit)) * 2; // multiplying by 2 as one addition and one multiplication is needed
      msRuntime = m_tR + m_tW;
      msRuntime += (maxGDLItr * m_tGDL * numberOfOperationPerElement * 3) * (numPass - 1);
      msRuntime += (minGDLItr * m_tGDL * numberOfOperationPerElement * 3);
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
      double numberOfOperationPerElement = ((double)bitsPerElement / (m_aquaboltFPUBitWidth * m_aquaboltFPUUnit));
      msRuntime = (2 * m_tR + m_tW + (maxGDLItr * m_tGDL * numberOfOperationPerElement * 3)) * (numPass - 1);
      msRuntime += (2 * m_tR + m_tW + (minGDLItr * m_tGDL * numberOfOperationPerElement * 3));
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

//! @brief  Perf energy model of aquabolt PIM for reduction sum
pimeval::perfEnergy
pimPerfEnergyAquabolt::getPerfEnergyForReduction(PimCmdEnum cmdType, const pimObjInfo& obj, unsigned numPass) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned numCore = obj.getNumCoresUsed();
  double cpuTDP = 200; // W; AMD EPYC 9124 16 coredouble numberOfOperationPerElement = ((double)bitsPerElement / (m_aquaboltFPUBitWidth * m_aquaboltFPUUnit));
  unsigned minElementPerRegion = std::ceil(obj.getNumElements() * 1.0 / obj.getNumCoreAvailable()) - (maxElementsPerRegion * (numPass - 1));
  unsigned maxGDLItr = maxElementsPerRegion * bitsPerElement / m_GDLWidth;
  unsigned minGDLItr = minElementPerRegion * bitsPerElement / m_GDLWidth;
  double numberOfOperationPerElement = ((double)bitsPerElement / (m_aquaboltFPUBitWidth * m_aquaboltFPUUnit));

  switch (cmdType) {
        case PimCmdEnum::REDSUM:
        case PimCmdEnum::REDSUM_RANGE:
        case PimCmdEnum::REDMIN:
        case PimCmdEnum::REDMIN_RANGE:
        case PimCmdEnum::REDMAX:
        case PimCmdEnum::REDMAX_RANGE:
        {
          // How many iteration require to read / write max elements per region
          msRuntime = (m_tR + (maxGDLItr * m_tGDL * numberOfOperationPerElement)) * (numPass - 1);
          msRuntime += (m_tR + (minGDLItr * m_tGDL * numberOfOperationPerElement));

          // Refer to fulcrum documentation
          mjEnergy = (m_eAP + (m_eGDL + (maxElementsPerRegion * m_blimpArithmeticEnergy * numberOfOperationPerElement))) * numPass * numCore;
          // reduction for all regions
          double aggregateMs = static_cast<double>(numCore) / (3200000 * 16);
          msRuntime += aggregateMs;
          mjEnergy += aggregateMs * cpuTDP;
          mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
          break;
        }
        default:
          std::cout << "PIM-Warning: Unsupported reduction command for aquabolt PIM: " 
                    << pimCmd::getName(cmdType, "") << std::endl;
          break;
    }

  return pimeval::perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Perf energy model of aquabolt PIM for broadcast
pimeval::perfEnergy
pimPerfEnergyAquabolt::getPerfEnergyForBroadcast(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned numCore = obj.getNumCoresUsed();

  unsigned minElementPerRegion = std::ceil(obj.getNumElements() * 1.0 / obj.getNumCoreAvailable()) - (maxElementsPerRegion * (numPass - 1));
  unsigned maxGDLItr = maxElementsPerRegion * bitsPerElement / m_GDLWidth;
  unsigned minGDLItr = minElementPerRegion * bitsPerElement / m_GDLWidth;
  msRuntime = (m_tW + maxGDLItr * m_tGDL) * (numPass - 1);
  msRuntime *= (m_tW + minGDLItr * m_tGDL);
  mjEnergy = (m_eAP + m_eGDL * maxGDLItr) * numPass * numCore;
  mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;

  return pimeval::perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Perf energy model of aquabolt PIM for rotate
pimeval::perfEnergy
pimPerfEnergyAquabolt::getPerfEnergyForRotate(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
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

  return pimeval::perfEnergy(msRuntime, mjEnergy);
}
