// File: pimPerfEnergyFulcrum.cc
// PIMeval Simulator - Performance Energy Models
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimPerfEnergyFulcrum.h"
#include "pimCmd.h"
#include <cmath>
#include <iostream>


//! @brief  Perf energy model of Fulcrum for func1
pimeval::perfEnergy
pimPerfEnergyFulcrum::getPerfEnergyForFunc1(PimCmdEnum cmdType, const pimObjInfo& obj) const
{

  // Fulcrum utilizes three walkers: two for input operands and one for the output operand.
  // For instructions that operate on a single operand, the next operand is fetched by the walker.
  // Consequently, only one row read operation is required in this case.
  // Additionally, using the walker-renaming technique (refer to the Fulcrum paper for details),
  // the write operation is also pipelined. Thus, only one row write operation is needed.

  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msALU = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement(PimBitWidth::ACTUAL);
  unsigned numCores =  obj.isLoadBalanced() ? obj.getNumCoreAvailable() : obj.getNumCoresUsed();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  double numberOfALUOperationPerElement = ((double)bitsPerElement / m_fulcrumAluBitWidth);
  unsigned minElementPerRegion = obj.isLoadBalanced() ? (std::ceil(obj.getNumElements() * 1.0 / numCores) - (maxElementsPerRegion * (numPass - 1))) : maxElementsPerRegion;
  switch (cmdType)
  {
    case PimCmdEnum::COPY_O2O:
    {
      msRead = m_tR * numPass;
      msWrite = m_tW * numPass;
      msRuntime = msRead + msWrite + msALU;
      mjEnergy = numPass * numCores * m_eAP * 2;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      break;
    }
    case PimCmdEnum::POPCOUNT:
    {
      numberOfALUOperationPerElement *= 12; // 4 shifts, 4 ands, 3 add/sub, 1 mul
      msRead = m_tR;
      msWrite = m_tW;
      msALU = ((maxElementsPerRegion * m_fulcrumAluLatency * numberOfALUOperationPerElement) * (numPass - 1)) + (minElementPerRegion * m_fulcrumAluLatency * numberOfALUOperationPerElement);
      msRuntime = msRead + msWrite + msALU;
      double energyArithmetic = (((maxElementsPerRegion - 1) * 2 *  m_fulcrumShiftEnergy) + (maxElementsPerRegion * m_fulcrumALUArithmeticEnergy * 4)) * (numPass - 1);
      energyArithmetic += (((minElementPerRegion - 1) * 2 *  m_fulcrumShiftEnergy) + (minElementPerRegion * m_fulcrumALUArithmeticEnergy * 4));
      double energyLogical = ((m_eAP * 2) + (((maxElementsPerRegion - 1) * 2 *  m_fulcrumShiftEnergy) + (maxElementsPerRegion * m_fulcrumALULogicalEnergy * 8))) * (numPass - 1);
      energyLogical += (m_eAP * 2) + (((minElementPerRegion - 1) * 2 *  m_fulcrumShiftEnergy) + (minElementPerRegion * m_fulcrumALULogicalEnergy * 8));
      mjEnergy = (energyArithmetic + energyLogical) * numCores ;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      break;
    }
    case PimCmdEnum::BIT_SLICE_EXTRACT:
    case PimCmdEnum::BIT_SLICE_INSERT:
    {
      if (cmdType == PimCmdEnum::BIT_SLICE_EXTRACT) {
        numberOfALUOperationPerElement *= 2; // 1 shift, 1 and
      } else if (cmdType == PimCmdEnum::BIT_SLICE_INSERT) {
        numberOfALUOperationPerElement *= 5; // 2 shifts, 1 not, 1 and, 1 or
      }
      msRead = m_tR;
      msWrite = m_tW;
      msALU = ((maxElementsPerRegion * m_fulcrumAluLatency * numberOfALUOperationPerElement) * (numPass - 1)) + (minElementPerRegion * m_fulcrumAluLatency * numberOfALUOperationPerElement);
      msRuntime = msRead + msWrite + msALU;
      double energyLogical = ((m_eAP * 2) + ((maxElementsPerRegion - 1) * 2 *  m_fulcrumShiftEnergy) + (maxElementsPerRegion * m_fulcrumALULogicalEnergy * numberOfALUOperationPerElement)) * (numPass - 1);
      energyLogical += ((m_eAP * 2) + (minElementPerRegion - 1) * 2 *  m_fulcrumShiftEnergy) + (minElementPerRegion * m_fulcrumALULogicalEnergy * numberOfALUOperationPerElement);
      mjEnergy = energyLogical * numCores;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      break;
    }
    case PimCmdEnum::ADD_SCALAR:
    case PimCmdEnum::SUB_SCALAR:
    case PimCmdEnum::MUL_SCALAR:
    case PimCmdEnum::DIV_SCALAR:
    case PimCmdEnum::ABS:
    {
      msRead = m_tR;
      msWrite = m_tW;
      msALU = ((maxElementsPerRegion * m_fulcrumAluLatency * numberOfALUOperationPerElement) * (numPass - 1)) + (minElementPerRegion * m_fulcrumAluLatency * numberOfALUOperationPerElement);
      msRuntime = msRead + msWrite + msALU;
      mjEnergy = (numPass - 1) * numCores * ((m_eAP * 2) + ((maxElementsPerRegion - 1) * 2 *  m_fulcrumShiftEnergy) + (maxElementsPerRegion * m_fulcrumALUArithmeticEnergy * numberOfALUOperationPerElement));
      mjEnergy += numCores * ((m_eAP * 2) + ((minElementPerRegion - 1) * 2 *  m_fulcrumShiftEnergy) + (minElementPerRegion * m_fulcrumALUArithmeticEnergy * numberOfALUOperationPerElement));
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
      msWrite = m_tW;
      msALU = ((maxElementsPerRegion * m_fulcrumAluLatency * numberOfALUOperationPerElement) * (numPass - 1)) + (minElementPerRegion * m_fulcrumAluLatency * numberOfALUOperationPerElement);
      msRuntime = msRead + msWrite + msALU;
      mjEnergy = (numPass - 1) * numCores * ((m_eAP * 2) + ((maxElementsPerRegion - 1) * 2 *  m_fulcrumShiftEnergy) + (maxElementsPerRegion * m_fulcrumALULogicalEnergy * numberOfALUOperationPerElement));
      mjEnergy = numCores * ((m_eAP * 2) + ((minElementPerRegion - 1) * 2 *  m_fulcrumShiftEnergy) + (minElementPerRegion * m_fulcrumALULogicalEnergy * numberOfALUOperationPerElement));
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      break;
    }
    default:
      std::cout << "PIM-Warning: Perf energy model not available for PIM command " << pimCmd::getName(cmdType, "") << std::endl;
      break;
  }

  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msALU);
}

//! @brief  Perf energy model of Fulcrum for func2
pimeval::perfEnergy
pimPerfEnergyFulcrum::getPerfEnergyForFunc2(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msALU = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement(PimBitWidth::ACTUAL);
  unsigned numCoresUsed = obj.isLoadBalanced() ? obj.getNumCoreAvailable() : obj.getNumCoresUsed();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned minElementPerRegion = obj.isLoadBalanced() ? (std::ceil(obj.getNumElements() * 1.0 / obj.getNumCoreAvailable()) - (maxElementsPerRegion * (numPass - 1))) : maxElementsPerRegion;
  double numberOfALUOperationPerElement = ((double)bitsPerElement / m_fulcrumAluBitWidth);
  switch (cmdType)
  {
    case PimCmdEnum::ADD:
    case PimCmdEnum::SUB:
    case PimCmdEnum::MUL:
    case PimCmdEnum::DIV:
    {
      msRead = 2 * m_tR * numPass;
      msWrite = m_tW * numPass;
      msALU = (maxElementsPerRegion * numberOfALUOperationPerElement * m_fulcrumAluLatency * (numPass - 1)) +  (minElementPerRegion * numberOfALUOperationPerElement * m_fulcrumAluLatency);
      msRuntime = msRead + msRead + msALU;
      mjEnergy = numCoresUsed * (numPass - 1) * ((m_eAP * 3) + ((maxElementsPerRegion - 1) * 3 *  m_fulcrumShiftEnergy) + ((maxElementsPerRegion) * m_fulcrumALUArithmeticEnergy * numberOfALUOperationPerElement));
      mjEnergy += numCoresUsed * ((m_eAP * 3) + ((minElementPerRegion - 1) * 3 *  m_fulcrumShiftEnergy) + ((minElementPerRegion) * m_fulcrumALUArithmeticEnergy * numberOfALUOperationPerElement));
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
      msRead = m_tR;
      msWrite = m_tW;
      msALU = (maxElementsPerRegion * numberOfALUOperationPerElement * m_fulcrumAluLatency * 2 * (numPass - 1)) +  (minElementPerRegion * numberOfALUOperationPerElement * m_fulcrumAluLatency * 2);
      msRuntime = msRead + msRead + msALU;
      mjEnergy = numCoresUsed * (numPass - 1) * ((m_eAP * 3) + ((maxElementsPerRegion - 1) * 3 *  m_fulcrumShiftEnergy * 2) + ((maxElementsPerRegion) * m_fulcrumALUArithmeticEnergy * 2 * numberOfALUOperationPerElement));
      mjEnergy += numCoresUsed * ((m_eAP * 3) + ((minElementPerRegion - 1) * 3 *  m_fulcrumShiftEnergy * 2) + ((minElementPerRegion) * m_fulcrumALUArithmeticEnergy * numberOfALUOperationPerElement * 2));
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
      msRead = 2 * m_tR * numPass;
      msWrite = m_tW * numPass;
      msALU = (maxElementsPerRegion * numberOfALUOperationPerElement * m_fulcrumAluLatency * (numPass - 1)) +  (minElementPerRegion * numberOfALUOperationPerElement * m_fulcrumAluLatency);
      msRuntime = msRead + msRead + msALU;
      mjEnergy = numCoresUsed * (numPass - 1) * ((m_eAP * 3) + ((maxElementsPerRegion - 1) * 3 *  m_fulcrumShiftEnergy) + ((maxElementsPerRegion) * m_fulcrumALULogicalEnergy * numberOfALUOperationPerElement));
      mjEnergy += numCoresUsed * ((m_eAP * 3) + ((minElementPerRegion - 1) * 3 *  m_fulcrumShiftEnergy) + ((minElementPerRegion) * m_fulcrumALULogicalEnergy * numberOfALUOperationPerElement));
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      break;
    }
    default:
      std::cout << "PIM-Warning: Perf energy model not available for PIM command " << pimCmd::getName(cmdType, "") << std::endl;
      break;
  }
  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msALU);
}

//! @brief  Perf energy model of Fulcrum for reduction sum
pimeval::perfEnergy
pimPerfEnergyFulcrum::getPerfEnergyForReduction(PimCmdEnum cmdType, const pimObjInfo& obj, unsigned numPass) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  unsigned bitsPerElement = obj.getBitsPerElement(PimBitWidth::ACTUAL);
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned numCore = obj.isLoadBalanced() ? obj.getNumCoreAvailable() : obj.getNumCoresUsed();
  unsigned minElementPerRegion = obj.isLoadBalanced() ? (std::ceil(obj.getNumElements() * 1.0 / numCore) - (maxElementsPerRegion * (numPass - 1))) : maxElementsPerRegion;
  double cpuTDP = 200; // W; AMD EPYC 9124 16 core

  switch (cmdType)
  {
  case PimCmdEnum::REDSUM:
  case PimCmdEnum::REDSUM_RANGE:
  case PimCmdEnum::REDMIN:
  case PimCmdEnum::REDMIN_RANGE:
  case PimCmdEnum::REDMAX:
  case PimCmdEnum::REDMAX_RANGE:
  {
    // read a row to walker, then reduce in serial
    double numberOfOperationPerElement = ((double)bitsPerElement / m_fulcrumAluBitWidth);
    // reduction for all regions assuming 16 core AMD EPYC 9124
    double aggregateMs = static_cast<double>(obj.getNumCoreAvailable()) / (3200000 * 16);
    
    msRead = m_tR;
    msWrite = 0;
    msCompute = aggregateMs + (maxElementsPerRegion * m_fulcrumAluLatency * numberOfOperationPerElement * (numPass  - 1)) + (minElementPerRegion * m_fulcrumAluLatency * numberOfOperationPerElement);
    msRuntime = msRead + msWrite + msCompute;
    mjEnergy = (numPass - 1) * numCore * (m_eAP * ((maxElementsPerRegion - 1) *  m_fulcrumShiftEnergy) + (maxElementsPerRegion * m_fulcrumALUArithmeticEnergy * numberOfOperationPerElement));
    mjEnergy += numCore * (m_eAP * ((minElementPerRegion - 1) *  m_fulcrumShiftEnergy) + (minElementPerRegion * m_fulcrumALUArithmeticEnergy * numberOfOperationPerElement));
    mjEnergy += aggregateMs * cpuTDP;
    mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
    break;
  }
  default:
    std::cout << "PIM-Warning: Perf energy model not available for PIM command " << pimCmd::getName(cmdType, "") << std::endl;
    break;
  }
  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute);
}

//! @brief  Perf energy model of Fulcrum for broadcast
pimeval::perfEnergy
pimPerfEnergyFulcrum::getPerfEnergyForBroadcast(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement(PimBitWidth::ACTUAL);
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned numCore = obj.isLoadBalanced() ? obj.getNumCoreAvailable() : obj.getNumCoresUsed();
  unsigned minElementPerRegion = obj.isLoadBalanced() ? (std::ceil(obj.getNumElements() * 1.0 / numCore) - (maxElementsPerRegion * (numPass - 1))) : maxElementsPerRegion;
  
  // assume taking 1 ALU latency to write an element
  double numberOfOperationPerElement = ((double)bitsPerElement / m_fulcrumAluBitWidth);
  msWrite = m_tW * numPass;
  msCompute = (m_fulcrumAluLatency * maxElementsPerRegion * numberOfOperationPerElement * (numPass - 1)) + (m_fulcrumAluLatency * minElementPerRegion * numberOfOperationPerElement);
  msRuntime = msRead + msWrite + msCompute;
  mjEnergy = (numPass - 1) * numCore * (m_eAP + ((maxElementsPerRegion - 1) *  m_fulcrumShiftEnergy) + ((maxElementsPerRegion) * m_fulcrumALULogicalEnergy * numberOfOperationPerElement));
  mjEnergy += numCore * (m_eAP + ((minElementPerRegion - 1) *  m_fulcrumShiftEnergy) + (minElementPerRegion * m_fulcrumALULogicalEnergy * numberOfOperationPerElement));
  mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;

  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute);
}

//! @brief  Perf energy model of Fulcrum for rotate
pimeval::perfEnergy
pimPerfEnergyFulcrum::getPerfEnergyForRotate(PimCmdEnum cmdType, const pimObjInfo& obj) const
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
  msRead = m_tR * numPass;
  msCompute = (bitsPerElement + 2) * m_tL * numPass;
  msWrite = m_tW * numPass;
  msRuntime = msRead + msWrite + msCompute;
  mjEnergy = (m_eAP + (bitsPerElement + 2) * m_eL) * numPass;
  msRuntime += 2 * perfEnergyBT.m_msRuntime;
  mjEnergy += 2 * perfEnergyBT.m_mjEnergy;
  std::cout << "PIM-Warning: Perf energy model is not precise for PIM command " << pimCmd::getName(cmdType, "") << std::endl;

  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute);
}

