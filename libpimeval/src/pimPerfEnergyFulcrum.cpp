// File: pimPerfEnergyFulcrum.cc
// PIMeval Simulator - Performance Energy Models
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimPerfEnergyFulcrum.h"
#include "pimCmd.h"
#include <iostream>


//! @brief  Perf energy model of Fulcrum for func1
pimeval::perfEnergy
pimPerfEnergyFulcrum::getPerfEnergyForFunc1(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned numCores = obj.getNumCoresUsed();

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
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
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
      msRuntime = m_tR + m_tW + (maxElementsPerRegion * m_fulcrumAluLatency * numberOfALUOperationPerElement * numPass);
      mjEnergy = numPass * numCores * ((m_eAP * 2) + ((maxElementsPerRegion - 1) * 2 *  m_fulcrumShiftEnergy) + ((maxElementsPerRegion) * m_fulcrumALULogicalEnergy * numberOfALUOperationPerElement));
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      break;
    }
    default:
      std::cout << "PIM-Warning: Perf energy model not available for PIM command " << pimCmd::getName(cmdType, "") << std::endl;
      break;
  }

  return pimeval::perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Perf energy model of Fulcrum for func2
pimeval::perfEnergy
pimPerfEnergyFulcrum::getPerfEnergyForFunc2(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned numCoresUsed = obj.getNumCoresUsed();

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
       * - Total execution time for one region of multiplication and addition >>>> reading/writing three DRAM rows as a result using walker renaming, row write is also pipelined
       *
       * As a result, only one read operation and one write operation is necessary for the entire pass.
      */
      msRuntime = m_tR + m_tW + (maxElementsPerRegion * numberOfALUOperationPerElement * m_fulcrumAluLatency * 2) * numPass;
      mjEnergy = numCoresUsed * numPass * ((m_eAP * 3) + ((maxElementsPerRegion - 1) * 3 *  m_fulcrumShiftEnergy) + ((maxElementsPerRegion) * m_fulcrumALUArithmeticEnergy * numberOfALUOperationPerElement));
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
      msRuntime = 2 * m_tR + m_tW + maxElementsPerRegion * numberOfALUOperationPerElement * m_fulcrumAluLatency;
      msRuntime *= numPass;
      mjEnergy = numCoresUsed * numPass * (((maxElementsPerRegion - 1) * 3 *  m_fulcrumShiftEnergy) + ((maxElementsPerRegion) * m_fulcrumALULogicalEnergy * numberOfALUOperationPerElement));
      mjEnergy += m_eAP * 3 * m_numChipsPerRank * m_numRanks;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      break;
    }
    default:
      std::cout << "PIM-Warning: Perf energy model not available for PIM command " << pimCmd::getName(cmdType, "") << std::endl;
      break;
  }

  return pimeval::perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Perf energy model of Fulcrum for reduction sum
pimeval::perfEnergy
pimPerfEnergyFulcrum::getPerfEnergyForRedSum(PimCmdEnum cmdType, const pimObjInfo& obj, unsigned numPass) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned numCore = obj.getNumCoresUsed();
  double cpuTDP = 200; // W; AMD EPYC 9124 16 core

  // read a row to walker, then reduce in serial
  double numberOfOperationPerElement = ((double)bitsPerElement / m_flucrumAluBitWidth);
  msRuntime = m_tR + (maxElementsPerRegion * m_fulcrumAluLatency * numberOfOperationPerElement * numPass);
  mjEnergy = numPass * numCore * (m_eAP * ((maxElementsPerRegion - 1) *  m_fulcrumShiftEnergy) + ((maxElementsPerRegion) * m_fulcrumALUArithmeticEnergy * numberOfOperationPerElement));
  // reduction for all regions
  double aggregateMs = static_cast<double>(numCore) / 3200000;
  msRuntime += aggregateMs;
  mjEnergy += aggregateMs * cpuTDP;
  mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;

  return pimeval::perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Perf energy model of Fulcrum for broadcast
pimeval::perfEnergy
pimPerfEnergyFulcrum::getPerfEnergyForBroadcast(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned numCore = obj.getNumCoresUsed();

  // assume taking 1 ALU latency to write an element
  double numberOfOperationPerElement = ((double)bitsPerElement / m_flucrumAluBitWidth);
  msRuntime = m_tW + m_fulcrumAluLatency * maxElementsPerRegion * numberOfOperationPerElement;
  msRuntime *= numPass;
  mjEnergy = numPass * numCore * (m_eAP + ((maxElementsPerRegion - 1) *  m_fulcrumShiftEnergy) + ((maxElementsPerRegion) * m_fulcrumALULogicalEnergy * numberOfOperationPerElement));
  mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;

  return pimeval::perfEnergy(msRuntime, mjEnergy);
}

//! @brief  Perf energy model of Fulcrum for rotate
pimeval::perfEnergy
pimPerfEnergyFulcrum::getPerfEnergyForRotate(PimCmdEnum cmdType, const pimObjInfo& obj) const
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

