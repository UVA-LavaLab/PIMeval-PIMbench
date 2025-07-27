// File: pimPerfEnergyAim.cc
// PIMeval Simulator - Performance Energy Models
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimPerfEnergyAim.h"
#include "pimCmd.h"
#include <iostream>
#include <cmath>

// AiM adds a SIMD Multiplier and a Reduction Tree in each bank.
// The supported instructions are: MAC.
// For simplicity, the SIMD lane width is assumed to be determined by the GDL width of the HBM/DDR memory.
// NOTE: The energy model is approximated. 

//! @brief  Perf energy model of aim PIM for func1
pimeval::perfEnergy
pimPerfEnergyAim::getPerfEnergyForFunc1(PimCmdEnum cmdType, const pimObjInfo& obj, const pimObjInfo& objDest) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  uint64_t totalOp = 0;
  switch (cmdType)
  {
    // Refer to AiM Paper (Table 2, Figure 5). OP Format: GRF = BANK +/* SRF
    case PimCmdEnum::ADD_SCALAR:
    case PimCmdEnum::MUL_SCALAR:
    case PimCmdEnum::AES_SBOX:
    case PimCmdEnum::AES_INVERSE_SBOX:
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

//! @brief  Perf energy model of aim for func2
pimeval::perfEnergy
pimPerfEnergyAim::getPerfEnergyForFunc2(PimCmdEnum cmdType, const pimObjInfo& obj, const pimObjInfo& objSrc2, const pimObjInfo& objDest) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  uint64_t totalOp = 0;
  switch (cmdType)
  {
    // Refer to Aquabolt Paper (Table 2, Figure 5). OP Format: GRF = BANK +/* GRF
    case PimCmdEnum::ADD:
    case PimCmdEnum::MUL:
    case PimCmdEnum::SCALED_ADD:
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
      std::cout << "PIM-Warning: Unsupported for AiM: " << pimCmd::getName(cmdType, "") << std::endl;
      break;
  }

  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute, totalOp);
}

//! @brief  Perf energy model of aim PIM for reduction sum
pimeval::perfEnergy
pimPerfEnergyAim::getPerfEnergyForReduction(PimCmdEnum cmdType, const pimObjInfo& obj, unsigned numPass) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  uint64_t totalOp = 0;

  switch (cmdType) {
    case PimCmdEnum::REDSUM:
    case PimCmdEnum::REDSUM_RANGE:
    case PimCmdEnum::REDMIN:
    case PimCmdEnum::REDMIN_RANGE:
    case PimCmdEnum::REDMAX:
    case PimCmdEnum::REDMAX_RANGE:
    default:
      std::cout << "PIM-Warning: Unsupported for AiM: " << pimCmd::getName(cmdType, "") << std::endl;
      break;
  }
  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute, totalOp);
}

//! @brief  Perf energy model of aim for broadcast
pimeval::perfEnergy
pimPerfEnergyAim::getPerfEnergyForBroadcast(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  uint64_t totalOp = 0;

  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute, totalOp);
}

//! @brief  Perf energy model of aim for rotate
pimeval::perfEnergy
pimPerfEnergyAim::getPerfEnergyForRotate(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  uint64_t totalOp = 0;
  std::cout << "PIM-Warning: Unsupported for AiM: " << pimCmd::getName(cmdType, "") << std::endl;

  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute, totalOp);
}

pimeval::perfEnergy pimPerfEnergyAim::getPerfEnergyForMac(PimCmdEnum cmdType, const pimObjInfo &obj) const
{
  // NumPass is always 1 for MAC operation in AiM. User really needs to make sure that this holds true.
  // Buffer read time is `tCAS - m_tGDL` based on following reasoning:
  // 1. tCAS = cycles required to data available at the I/O interface after a read command.
  // 2. m_tGDL = cycles required for two consecutive read commands to the same bank.
  // Hence, the time to read data from the global AiM buffer to the bank interface is `tCAS - m_tGDL`.
  // User may wonder why buffer read time is not multiplied by number of banks per chip. This is because according the AiM paper mentions that buffer is n-way fanout to n banks in the same chip.
  // AiM paper mentions accumulation reduction tree requires 4 cycles after the multiplier. Hence, the compute time for accumulation is `4 * tCK`.
  // TODO: Energy model
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  uint64_t totalOp = 0;
  unsigned bitsPerElement = obj.getBitsPerElement(PimBitWidth::ACTUAL);
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned numCore = obj.getNumCoreAvailable();
  unsigned elementsPerCore = std::ceil(obj.getNumElements() * 1.0 / numCore);
  unsigned gdlItr = std::ceil(elementsPerCore * bitsPerElement * 1.0 / m_GDLWidth);

  pimeval::perfEnergy perfEnergyBT = getPerfEnergyForBytesTransfer(PimCmdEnum::COPY_D2H, (bitsPerElement * numCore) / 8);
  
  msRead = m_tACT + m_tPRE + (m_tCAS - m_tGDL) * gdlItr;
  msWrite = perfEnergyBT.m_msRuntime;
  msCompute = (gdlItr * m_tGDL + 4 * m_tCK);
  msRuntime = msRead + msWrite + msCompute;
  mjEnergy = (m_eAP + (m_eR + (maxElementsPerRegion * m_aquaboltArithmeticEnergy))) * numCore;
  mjEnergy += perfEnergyBT.m_mjEnergy;
  mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
  totalOp = obj.getNumElements() * 2;
  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute, totalOp);
}
