// File: pimParamsPerf.cc
// PIMeval Simulator - Performance parameters
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimParamsPerf.h"
#include "pimSim.h"
#include "pimCmd.h"
#include "pimPerfEnergyTables.h"
#include <cstdio>
#include <cmath>

//! @brief  pimParamsPerf ctor
pimParamsPerf::pimParamsPerf(pimParamsDram* paramsDram)
  : m_paramsDram(paramsDram)
{
  m_tR = m_paramsDram->getNsRowRead() / m_nano_to_milli;
  m_tW = m_paramsDram->getNsRowWrite() / m_nano_to_milli;
  m_tL = m_paramsDram->getNsTCCD_S() / m_nano_to_milli;
  m_tGDL = m_paramsDram->getNsTCAS() / m_nano_to_milli;
  m_GDLWidth = m_paramsDram->getBurstLength() * m_paramsDram->getDeviceWidth();
}

//! @brief  Get ms runtime for bytes transferred between host and device
double
pimParamsPerf::getMsRuntimeForBytesTransfer(uint64_t numBytes) const
{
  int numRanks = static_cast<int>(pimSim::get()->getNumRanks());
  int numActiveRank = numRanks;
  double typicalRankBW = m_paramsDram->getTypicalRankBW(); // GB/s
  double totalMsRuntime = static_cast<double>(numBytes) / (typicalRankBW * numActiveRank * 1024 * 1024 * 1024 / 1000);
  return totalMsRuntime;
}

//! @brief  Get ms runtime for bit-serial PIM devices
//!         BitSIMD and SIMDRAM need different fields
double
pimParamsPerf::getMsRuntimeBitSerial(PimDeviceEnum deviceType, PimCmdEnum cmdType, PimDataType dataType, unsigned bitsPerElement, unsigned numPass) const
{
  bool ok = false;
  double msRuntime = 0.0;

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
  return msRuntime;
}

//! @brief  Get ms runtime for func1
double
pimParamsPerf::getMsRuntimeForFunc1(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  PimDeviceEnum simTarget = pimSim::get()->getSimTarget();
  double msRuntime = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  PimDataType dataType = obj.getDataType();
  switch (simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_AP:
  case PIM_DEVICE_BITSIMD_H:
  case PIM_DEVICE_SIMDRAM:
    msRuntime += getMsRuntimeBitSerial(simTarget, cmdType, dataType, bitsPerElement, numPass);
    break;
  case PIM_DEVICE_FULCRUM:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double numberOfALUOperationPerElement = ((double)bitsPerElement / m_flucrumAluBitWidth); 
    switch (cmdType)
    {
    case PimCmdEnum::POPCOUNT: numberOfALUOperationPerElement *= 12; break; // 4 shifts, 4 ands, 3 add/sub, 1 mul
    case PimCmdEnum::ADD_SCALAR:
    case PimCmdEnum::SUB_SCALAR:
    case PimCmdEnum::MUL_SCALAR:
    case PimCmdEnum::DIV_SCALAR:
    case PimCmdEnum::AND_SCALAR:
    case PimCmdEnum::OR_SCALAR:
    case PimCmdEnum::XOR_SCALAR:
    case PimCmdEnum::XNOR_SCALAR:
    case PimCmdEnum::GT_SCALAR:
    case PimCmdEnum::LT_SCALAR:
    case PimCmdEnum::EQ_SCALAR:
    case PimCmdEnum::MIN_SCALAR:
    case PimCmdEnum::MAX_SCALAR:
    case PimCmdEnum::ABS:
    case PimCmdEnum::SHIFT_BITS_L:
    case PimCmdEnum::SHIFT_BITS_R: break;
    default:
       std::printf("PIM-Warning: Unsupported PIM command.\n");
       break;
    }

    // Fulcrum utilizes three walkers: two for input operands and one for the output operand.
    // For instructions that operate on a single operand, the next operand is fetched by the walker.
    // Consequently, only one row read operation is required in this case.
    // Additionally, using the walker-renaming technique (refer to the Fulcrum paper for details),
    // the write operation is also pipelined. Thus, only one row write operation is needed.
    msRuntime = m_tR + m_tW + (maxElementsPerRegion * m_fulcrumAluLatency * numberOfALUOperationPerElement * numPass);
    break;
  }
  case PIM_DEVICE_BANK_LEVEL:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double numberOfOperationPerElement = ((double)bitsPerElement / m_blimpCoreBitWidth);

    // How many iteration require to read / write max elements per region
    unsigned numGDLItr = maxElementsPerRegion * bitsPerElement / m_GDLWidth;
    double totalGDLOverhead = m_tGDL * numGDLItr; // read can be pipelined and write cannot be pipelined
    // Refer to fulcrum documentation 
    msRuntime = m_tR + m_tW + totalGDLOverhead + (maxElementsPerRegion * m_blimpCoreLatency * numberOfOperationPerElement * numPass);
    switch (cmdType)
    {
    case PimCmdEnum::POPCOUNT:
    case PimCmdEnum::ABS:
    case PimCmdEnum::SHIFT_BITS_L:
    case PimCmdEnum::SHIFT_BITS_R:
    case PimCmdEnum::ADD_SCALAR:
    case PimCmdEnum::SUB_SCALAR:
    case PimCmdEnum::MUL_SCALAR:
    case PimCmdEnum::DIV_SCALAR:
    case PimCmdEnum::AND_SCALAR:
    case PimCmdEnum::OR_SCALAR:
    case PimCmdEnum::XOR_SCALAR:
    case PimCmdEnum::XNOR_SCALAR:
    case PimCmdEnum::GT_SCALAR:
    case PimCmdEnum::LT_SCALAR:
    case PimCmdEnum::EQ_SCALAR:
    case PimCmdEnum::MIN_SCALAR:
    case PimCmdEnum::MAX_SCALAR: break;
    default:
       std::printf("PIM-Warning: Unsupported PIM command.\n");
       break;
    }
    break;
  }
  default:
    msRuntime = 1000000;
  }
  return msRuntime;
}

//! @brief  Get ms runtime for func2
double
pimParamsPerf::getMsRuntimeForFunc2(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  PimDeviceEnum simTarget = pimSim::get()->getSimTarget();
  double msRuntime = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  PimDataType dataType = obj.getDataType();

  switch (simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_AP:
  case PIM_DEVICE_BITSIMD_H:
  case PIM_DEVICE_SIMDRAM:
    msRuntime = getMsRuntimeBitSerial(simTarget, cmdType, dataType, bitsPerElement, numPass);
    break;
  case PIM_DEVICE_FULCRUM:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double numberOfALUOperationPerElement = ((double)bitsPerElement / m_flucrumAluBitWidth);
    msRuntime = 2 * m_tR + m_tW + maxElementsPerRegion * numberOfALUOperationPerElement * m_fulcrumAluLatency;
    msRuntime *= numPass;
    break;
  }
  case PIM_DEVICE_BANK_LEVEL:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double numberOfOperationPerElement = ((double)bitsPerElement / m_blimpCoreBitWidth);

    // How many iteration require to read / write max elements per region
    unsigned numGDLItr = maxElementsPerRegion * bitsPerElement / m_GDLWidth;
    double totalGDLOverhead = m_tGDL * numGDLItr * 2; // one read can be pipelined

    msRuntime = 2 * m_tR + m_tW + totalGDLOverhead + maxElementsPerRegion * m_blimpCoreLatency * numberOfOperationPerElement;
    
    switch (cmdType)
    {
    case PimCmdEnum::SCALED_ADD:
    {
      msRuntime += maxElementsPerRegion * numberOfOperationPerElement * m_blimpCoreLatency;
      break;
    }
    case PimCmdEnum::ADD:
    case PimCmdEnum::SUB:
    case PimCmdEnum::MUL:
    case PimCmdEnum::DIV:
    case PimCmdEnum::AND:
    case PimCmdEnum::OR:
    case PimCmdEnum::XOR:
    case PimCmdEnum::XNOR:
    case PimCmdEnum::GT:
    case PimCmdEnum::LT:
    case PimCmdEnum::EQ:
    case PimCmdEnum::MIN:
    case PimCmdEnum::MAX:  break;
    default:
       std::printf("PIM-Warning: Unsupported PIM command.\n");
       break;
    }
    msRuntime *= numPass;
    break;
  }
  default:
    msRuntime = 1e10;
  }
  return msRuntime;
}

//! @brief  Get ms runtime for reduction sum
double
pimParamsPerf::getMsRuntimeForRedSum(PimCmdEnum cmdType, const pimObjInfo& obj, unsigned numPass) const
{
  PimDeviceEnum simTarget = pimSim::get()->getSimTarget();
  double msRuntime = 0.0;
  PimDataType dataType = obj.getDataType();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned numRegions = obj.getRegions().size();
  uint64_t numElements = obj.getNumElements();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned numCore = obj.getNumCoresUsed();

  switch (simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_AP:
    if (dataType == PIM_INT8 || dataType == PIM_INT16 || dataType == PIM_INT64 || dataType == PIM_INT32 || dataType == PIM_UINT8 || dataType == PIM_UINT16 || dataType == PIM_UINT32 || dataType == PIM_UINT64) {
      // Assume pop count reduction circut in tR runtime
      msRuntime = ((m_tR + m_tR) * bitsPerElement);
      msRuntime *= numPass;
      // reduction for all regions
      msRuntime += static_cast<double>(numRegions) / 3200000;
    } else {
      assert(0);
    }
    break;
  case PIM_DEVICE_SIMDRAM:
    // todo
    std::printf("PIM-Warning: SIMDRAM performance stats not implemented yet.\n");
    break;
  case PIM_DEVICE_BITSIMD_H:
    // Sequentially process all elements per CPU cycle
    msRuntime = static_cast<double>(numElements) / 3200000; // typical 3.2 GHz CPU
    // consider PCL
    break;
  case PIM_DEVICE_FULCRUM:
  {
    // read a row to walker, then reduce in serial
    double numberOfOperationPerElement = ((double)bitsPerElement / m_flucrumAluBitWidth);
    msRuntime = m_tR + (maxElementsPerRegion * m_fulcrumAluLatency * numberOfOperationPerElement * numPass);
    // reduction for all regions
    msRuntime += static_cast<double>(numCore) / 3200000;
    break;
  }
  case PIM_DEVICE_BANK_LEVEL:
  {
    double numberOfOperationPerElement = ((double)bitsPerElement / m_blimpCoreBitWidth);
    msRuntime = m_tR + m_tGDL + (maxElementsPerRegion * m_blimpCoreLatency * numberOfOperationPerElement * numPass);
    // reduction for all regions
    msRuntime += static_cast<double>(numCore) / 3200000;
    break;
  }
  default:
    msRuntime = 1e10;
  }

  return msRuntime;
}

//! @brief  Get ms runtime for broadcast
double
pimParamsPerf::getMsRuntimeForBroadcast(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  PimDeviceEnum simTarget = pimSim::get()->getSimTarget();
  double msRuntime = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();

  switch (simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_AP:
  {
    // For one pass: For every bit: Set SA to bit value; Write SA to row;
    msRuntime = (m_tL + m_tW) * bitsPerElement;
    msRuntime *= numPass;
    break;
  }
  case PIM_DEVICE_SIMDRAM:
  {
    // todo
    msRuntime *= numPass;
    break;
  }
  case PIM_DEVICE_BITSIMD_H:
  {
    // For one pass: For every element: 1 tCCD per byte
    uint64_t maxBytesPerRegion = (uint64_t)maxElementsPerRegion * (bitsPerElement / 8);
    msRuntime = m_tW + m_tL * maxBytesPerRegion; // for one pass
    msRuntime *= numPass;
    break;
  }
  case PIM_DEVICE_FULCRUM:
  {
    // assume taking 1 ALU latency to write an element
    double numberOfOperationPerElement = ((double)bitsPerElement / m_flucrumAluBitWidth);
    msRuntime = m_tW + m_fulcrumAluLatency * maxElementsPerRegion * numberOfOperationPerElement;
    msRuntime *= numPass;
    break;
  }
  case PIM_DEVICE_BANK_LEVEL:
  {
    // assume taking 1 ALU latency to write an element
    double numberOfOperationPerElement = ((double)bitsPerElement / m_blimpCoreBitWidth);
    msRuntime = m_tW + m_tGDL + (m_blimpCoreLatency * maxElementsPerRegion * numberOfOperationPerElement);
    msRuntime *= numPass;
    break;
  }
  default:
    msRuntime = 1e10;
  }

  return msRuntime;
}

//! @brief  Get ms runtime for rotate
double
//pimParamsPerf::getMsRuntimeForRotate(PimCmdEnum cmdType, unsigned bitsPerElement, unsigned numRegions) const
pimParamsPerf::getMsRuntimeForRotate(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  PimDeviceEnum simTarget = pimSim::get()->getSimTarget();
  double msRuntime = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned numRegions = obj.getRegions().size();

  switch (simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_AP:
    // rotate within subarray:
    // For every bit: Read row to SA; move SA to R1; Shift R1; Move R1 to SA; Write SA to row
    msRuntime = (m_tR + 3 * m_tL + m_tW) * bitsPerElement; // for one pass
    msRuntime *= numPass;
    // boundary handling
    msRuntime += 2 * getMsRuntimeForBytesTransfer(numRegions * bitsPerElement / 8);
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
    msRuntime = (m_tR + (bitsPerElement + 2) * m_tL + m_tW); // for one pass
    msRuntime *= numPass;
    // boundary handling
    msRuntime += 2 * getMsRuntimeForBytesTransfer(numRegions * bitsPerElement / 8);
    break;
  default:
    msRuntime = 1e10;
  }

  return msRuntime;
}

