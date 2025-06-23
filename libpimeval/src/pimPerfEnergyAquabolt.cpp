// File: pimPerfEnergyAquabolt.cc
// PIMeval Simulator - Performance Energy Models
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimPerfEnergyAquabolt.h"
#include "pimCmd.h"
#include <iostream>
#include <cmath>

// Aquabolt adds a SIMD FPU shared between two banks, with only one bank accessing it at a time.
// The supported FPU instructions are: ADD, MUL, MAC, and RELU. However, RELU is currently not implemented in the simulator.
// This model assumes that each FPU operation (ADD, MUL, MAC, or RELU) takes `tCCD_L * 3` cycles to execute.
// Additionally, for simplicity, the SIMD lane width is assumed to be determined by the GDL width of the HBM/DDR memory. 
// This analytical model has been validated against the Aquabolt for vector addition and multiplication using a 100M-element vector of 16-bit integers. 
// The model demonstrates a 1.5x speedup compared to the original Aquabolt.
// NOTE: The energy model is approximated. 

//! @brief  Perf energy model of aquabolt PIM for func1
pimeval::perfEnergy
pimPerfEnergyAquabolt::getPerfEnergyForFunc1(PimCmdEnum cmdType, const pimObjInfo& obj, const pimObjInfo& objDest) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement(PimBitWidth::ACTUAL);
  unsigned numCores = obj.getNumCoresUsed();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned numberOfOperationPerElement = std::ceil(bitsPerElement * 1.0 / m_aquaboltFPUBitWidth);
  unsigned minElementPerRegion = std::ceil(obj.getNumElements() * 1.0 / obj.getNumCoreAvailable()) - (maxElementsPerRegion * (numPass - 1));
  unsigned maxGDLItr = std::ceil(maxElementsPerRegion * bitsPerElement * 1.0 / m_GDLWidth);
  unsigned minGDLItr = std::ceil(minElementPerRegion * bitsPerElement * 1.0 / m_GDLWidth);
  double aquaboltCoreCycle = m_tGDL * 3;
  uint64_t totalOp = 0;
  switch (cmdType)
  {
    case PimCmdEnum::ADD_SCALAR:
    case PimCmdEnum::MUL_SCALAR:
    { 
      msRead = m_tR;
      msWrite = m_tW;
      msCompute = (minGDLItr * aquaboltCoreCycle * numberOfOperationPerElement) + ((maxGDLItr * aquaboltCoreCycle * numberOfOperationPerElement) * (numPass - 1));
      msRuntime = msRead + msWrite + msCompute;
      mjEnergy = (m_eAP * 2 + (m_eR * 2 + (maxElementsPerRegion * m_aquaboltArithmeticEnergy * numberOfOperationPerElement))) * numCores * numPass;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      totalOp = obj.getNumElements();
      break;
    }
    case PimCmdEnum::AES_SBOX:
    case PimCmdEnum::AES_INVERSE_SBOX:
    {
      numberOfOperationPerElement = 1; // Assuming each 8-bit element allocates one ALU word
      msRead = m_tR;
      msWrite = m_tW;
      msCompute = (minGDLItr * aquaboltCoreCycle * numberOfOperationPerElement) + ((maxGDLItr * aquaboltCoreCycle * numberOfOperationPerElement) * (numPass - 1));
      msRuntime = msRead + msWrite + msCompute;
      mjEnergy = (m_eAP * 2 + (m_eR * 2 + (maxElementsPerRegion * m_aquaboltArithmeticEnergy * numberOfOperationPerElement))) * numCores * numPass;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      totalOp = obj.getNumElements();
      break;
    }
    case PimCmdEnum::POPCOUNT:
    case PimCmdEnum::ABS:
    case PimCmdEnum::SUB_SCALAR:
    case PimCmdEnum::DIV_SCALAR:
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
    default:
      std::cout << "PIM-Warning: Perf energy model not available for PIM command " << pimCmd::getName(cmdType, "") << std::endl;
      break;
  }

  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute, totalOp);
}

//! @brief  Perf energy model of aquabolt PIM for func2
pimeval::perfEnergy
pimPerfEnergyAquabolt::getPerfEnergyForFunc2(PimCmdEnum cmdType, const pimObjInfo& obj, const pimObjInfo& objSrc2, const pimObjInfo& objDest) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement(PimBitWidth::ACTUAL);
  unsigned numCoresUsed = obj.getNumCoresUsed();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned minElementPerRegion = std::ceil(obj.getNumElements() * 1.0 / obj.getNumCoreAvailable()) - (maxElementsPerRegion * (numPass - 1));
  unsigned maxGDLItr = std::ceil(maxElementsPerRegion * bitsPerElement * 1.0 / m_GDLWidth);
  unsigned minGDLItr = std::ceil(minElementPerRegion * bitsPerElement * 1.0 / m_GDLWidth);
  double aquaboltCoreCycle = m_tGDL * 3;
  uint64_t totalOp = 0;

  switch (cmdType)
  {
    case PimCmdEnum::ADD:
    case PimCmdEnum::MUL:
    {
      unsigned numberOfOperationPerElement = std::ceil(bitsPerElement * 1.0 / m_aquaboltFPUBitWidth);
      msRead = 2 * m_tR * numPass;
      msWrite = m_tW * numPass;
      msCompute = (maxGDLItr * numberOfOperationPerElement * aquaboltCoreCycle) * (numPass - 1);
      msCompute += (minGDLItr * numberOfOperationPerElement * aquaboltCoreCycle);
      msRuntime = msRead + msWrite + msCompute;
      mjEnergy = ((m_eAP * 3) + (m_eR * 3 + (maxElementsPerRegion * m_aquaboltArithmeticEnergy * numberOfOperationPerElement))) * numCoresUsed * numPass;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      totalOp = obj.getNumElements();
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
      unsigned numberOfOperationPerElement = std::ceil(bitsPerElement * 1.0 / m_aquaboltFPUBitWidth) * 2; // multiplying by 2 as one addition and one multiplication is needed
      msRead = m_tR;
      msWrite = m_tW;
      msCompute = (maxGDLItr * aquaboltCoreCycle * numberOfOperationPerElement) * (numPass - 1);
      msCompute += (minGDLItr * aquaboltCoreCycle * numberOfOperationPerElement);
      msRuntime = msRead + msWrite + msCompute;
      mjEnergy = ((m_eAP * 3) + (m_eR * 3 + (maxElementsPerRegion * m_aquaboltArithmeticEnergy * numberOfOperationPerElement))) * numCoresUsed;
      mjEnergy += maxElementsPerRegion * numberOfOperationPerElement * m_aquaboltArithmeticEnergy * numCoresUsed;
      mjEnergy *= numPass;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      totalOp = obj.getNumElements() * 2;
      break;
    }
    case PimCmdEnum::DIV:
    case PimCmdEnum::SUB:
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
    default:
      std::cout << "PIM-Warning: Unsupported for Aquabolt: " << pimCmd::getName(cmdType, "") << std::endl;
      break;
  }

  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute, totalOp);
}

//! @brief  Perf energy model of aquabolt PIM for reduction sum
pimeval::perfEnergy
pimPerfEnergyAquabolt::getPerfEnergyForReduction(PimCmdEnum cmdType, const pimObjInfo& obj, unsigned numPass) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  unsigned bitsPerElement = obj.getBitsPerElement(PimBitWidth::ACTUAL);
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned numCore = obj.getNumCoresUsed();
  double cpuTDP = 200; // W; AMD EPYC 9124 16 core
  unsigned minElementPerRegion = std::ceil(obj.getNumElements() * 1.0 / obj.getNumCoreAvailable()) - (maxElementsPerRegion * (numPass - 1));
  unsigned maxGDLItr = std::ceil(maxElementsPerRegion * bitsPerElement * 1.0 / m_GDLWidth);
  unsigned minGDLItr = std::ceil(minElementPerRegion * bitsPerElement * 1.0 / m_GDLWidth);
  unsigned numberOfOperationPerElement = std::ceil(bitsPerElement * 1.0 / m_aquaboltFPUBitWidth);
  double aquaboltCoreCycle = m_tGDL * 3;
  uint64_t totalOp = 0;

  switch (cmdType) {
    case PimCmdEnum::REDSUM:
    case PimCmdEnum::REDSUM_RANGE:
    {
      msRead = m_tR;
      msCompute = (maxGDLItr * aquaboltCoreCycle * numberOfOperationPerElement) * (numPass - 1);
      msCompute += (m_tR + (minGDLItr * aquaboltCoreCycle * numberOfOperationPerElement));
      msRuntime = msRead + msWrite + msCompute;
      // Refer to fulcrum documentation
      mjEnergy = (m_eAP + (m_eR + (maxElementsPerRegion * m_aquaboltArithmeticEnergy * numberOfOperationPerElement))) * numPass * numCore;
      // reduction for all regions
      double aggregateMs = static_cast<double>(numCore) / (3200000 * 16);
      msRuntime += aggregateMs;
      mjEnergy += aggregateMs * cpuTDP;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      totalOp = obj.getNumElements();
      break;
    }
    case PimCmdEnum::REDMIN:
    case PimCmdEnum::REDMIN_RANGE:
    case PimCmdEnum::REDMAX:
    case PimCmdEnum::REDMAX_RANGE:
    default:
      std::cout << "PIM-Warning: Unsupported for Aquabolt: " 
                << pimCmd::getName(cmdType, "") << std::endl;
      break;
  }
  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute, totalOp);
}

//! @brief  Perf energy model of aquabolt PIM for broadcast
pimeval::perfEnergy
pimPerfEnergyAquabolt::getPerfEnergyForBroadcast(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement(PimBitWidth::ACTUAL);
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned numCore = obj.getNumCoresUsed();
  uint64_t totalOp = 0;

  unsigned minElementPerRegion = std::ceil(obj.getNumElements() * 1.0 / obj.getNumCoreAvailable()) - (maxElementsPerRegion * (numPass - 1));
  unsigned maxGDLItr = std::ceil(maxElementsPerRegion * bitsPerElement * 1.0 / m_GDLWidth);
  unsigned minGDLItr = std::ceil(minElementPerRegion * bitsPerElement * 1.0 / m_GDLWidth);
  msWrite = (m_tW + maxGDLItr * m_tGDL) * (numPass - 1);
  msWrite += (m_tW + minGDLItr * m_tGDL);
  msRuntime = msRead + msWrite + msCompute;
  mjEnergy = (m_eAP + m_eR * maxGDLItr) * numPass * numCore;
  mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;

  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute, totalOp);
}

//! @brief  Perf energy model of aquabolt PIM for rotate
pimeval::perfEnergy
pimPerfEnergyAquabolt::getPerfEnergyForRotate(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  uint64_t totalOp = 0;
  std::cout << "PIM-Warning: Unsupported for Aquabolt: " << pimCmd::getName(cmdType, "") << std::endl;

  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute, totalOp);
}

