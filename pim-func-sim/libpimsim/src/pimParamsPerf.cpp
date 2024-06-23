// File: pimParamsPerf.cc
// PIM Functional Simulator - Performance parameters
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "pimParamsPerf.h"
#include "pimSim.h"
#include "pimCmd.h"
#include <cstdio>
#include <unordered_map>
#include <tuple>


//! @brief  BitSIMD performance table (Tuple: #R, #W, #L)
const std::unordered_map<PimDeviceEnum, std::unordered_map<PimDataType,
    std::unordered_map<PimCmdEnum, std::tuple<unsigned, unsigned, unsigned>>>>
pimParamsPerf::s_bitsimdPerfTable = {
  { PIM_DEVICE_BITSIMD_V, {
    { PIM_INT32, {
      { PimCmdEnum::ABS,          {   98,   66,  192 } },
      { PimCmdEnum::POPCOUNT,     {  161,  105,  286 } },
      { PimCmdEnum::SHIFT_BITS_R, {   31,   32,    1 } },
      { PimCmdEnum::SHIFT_BITS_L, {   31,   32,    1 } },
      { PimCmdEnum::ADD,          {   64,   33,  161 } },
      { PimCmdEnum::SUB,          {   64,   33,  161 } },
      { PimCmdEnum::MUL,          { 1940, 1095, 3606 } },
      { PimCmdEnum::DIV,          { 3168, 1727, 4257 } },
      { PimCmdEnum::AND,          {   64,   32,   64 } },
      { PimCmdEnum::OR,           {   64,   32,   64 } },
      { PimCmdEnum::XOR,          {   64,   32,   64 } },
      { PimCmdEnum::XNOR,         {   64,   32,   64 } },
      { PimCmdEnum::GT,           {   64,   32,   66 } },
      { PimCmdEnum::LT,           {   64,   32,   66 } },
      { PimCmdEnum::EQ,           {   64,   32,   66 } },
      { PimCmdEnum::MIN,          {  164,   67,  258 } },
      { PimCmdEnum::MAX,          {  164,   67,  258 } },
    }},
    { PIM_FP32, {
      { PimCmdEnum::ADD,          { 1331,  685, 1687 } },
      { PimCmdEnum::SUB,          { 1331,  685, 1687 } },
      { PimCmdEnum::MUL,          { 1852, 1000, 3054 } },
      { PimCmdEnum::DIV,          { 2744, 1458, 4187 } },
    }}
  }},
  { PIM_DEVICE_BITSIMD_V_AP, {
    { PIM_INT32, {
      { PimCmdEnum::ABS,          {   98,   66,  320 } },
      { PimCmdEnum::POPCOUNT,     {  161,  105,  318 } },
      { PimCmdEnum::SHIFT_BITS_R, {   31,   32,    1 } },
      { PimCmdEnum::SHIFT_BITS_L, {   31,   32,    1 } },
      { PimCmdEnum::ADD,          {   64,   33,  161 } },
      { PimCmdEnum::SUB,          {   64,   33,  161 } },
      { PimCmdEnum::MUL,          { 4291, 1799, 7039 } },
      { PimCmdEnum::DIV,          { 3728, 1744, 6800 } },
      { PimCmdEnum::AND,          {   64,   32,   64 } },
      { PimCmdEnum::OR,           {   64,   32,  128 } },
      { PimCmdEnum::XOR,          {   64,   32,  128 } },
      { PimCmdEnum::XNOR,         {   64,   32,   64 } },
      { PimCmdEnum::GT,           {   64,   32,   66 } },
      { PimCmdEnum::LT,           {   64,   32,   66 } },
      { PimCmdEnum::EQ,           {   64,   32,   66 } },
      { PimCmdEnum::MIN,          {  164,   67,  261 } },
      { PimCmdEnum::MAX,          {  164,   67,  261 } },
    }},
    { PIM_FP32, {
      { PimCmdEnum::ADD,          { 1597,  822, 2024 } },
      { PimCmdEnum::SUB,          { 1597,  822, 2024 } },
      { PimCmdEnum::MUL,          { 2222, 1200, 3664 } },
      { PimCmdEnum::DIV,          { 3292, 1749, 5024 } },
    }}
  }},
};

//! @brief  pimParamsPerf ctor
pimParamsPerf::pimParamsPerf(pimParamsDram* paramsDram)
  : m_paramsDram(paramsDram)
{
  m_tR = m_paramsDram->getNsRowRead() / 1000000.0;
  m_tW = m_paramsDram->getNsRowWrite() / 1000000.0;
  m_tL = m_paramsDram->getNsTCCD() / 1000000.0;
}

//! @brief  Set PIM device and simulation target
void
pimParamsPerf::setDevice(PimDeviceEnum deviceType)
{
  m_curDevice = deviceType;
  m_simTarget = deviceType;

  // determine simulation target for functional device
  if (deviceType == PIM_FUNCTIONAL) {
    PimDeviceEnum simTarget = PIM_DEVICE_NONE;
    // from 'make PIM_SIM_TARGET=...'
    #if defined(PIM_SIM_TARGET)
    simTarget = PIM_SIM_TARGET;
    #endif
    // default sim target
    if (simTarget == PIM_DEVICE_NONE || simTarget == PIM_FUNCTIONAL) {
      simTarget = PIM_DEVICE_BITSIMD_V;
    }
    m_simTarget = simTarget;
  }
}

//! @brief  If a PIM device uses vertical data layout
bool
pimParamsPerf::isVLayoutDevice() const
{
  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V: return true;
  case PIM_DEVICE_BITSIMD_V_AP: return true;
  case PIM_DEVICE_SIMDRAM: return true;
  case PIM_DEVICE_BITSIMD_H: return false;
  case PIM_DEVICE_FULCRUM: return false;
  case PIM_DEVICE_BANK_LEVEL: return false;
  case PIM_DEVICE_NONE:
  case PIM_FUNCTIONAL:
  default:
    assert(0);
  }
  return false;
}

//! @brief  If a PIM device uses horizontal data layout
bool
pimParamsPerf::isHLayoutDevice() const
{
  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V: return false;
  case PIM_DEVICE_BITSIMD_V_AP: return false;
  case PIM_DEVICE_SIMDRAM: return false;
  case PIM_DEVICE_BITSIMD_H: return true;
  case PIM_DEVICE_FULCRUM: return true;
  case PIM_DEVICE_BANK_LEVEL: return true;
  case PIM_DEVICE_NONE:
  case PIM_FUNCTIONAL:
  default:
    assert(0);
  }
  return false;
}

//! @brief  If a PIM device uses hybrid data layout
bool
pimParamsPerf::isHybridLayoutDevice() const
{
  return false;
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

//! @brief  Get ms runtime for BitSIMD bit-serial operations
double
pimParamsPerf::getMsRuntimeBitsimd(PimDeviceEnum deviceType, PimCmdEnum cmdType, PimDataType dataType, unsigned numPass) const
{
  double msRuntime = 0.0;
  auto it1 = s_bitsimdPerfTable.find(deviceType);
  if (it1 != s_bitsimdPerfTable.end()) {
    auto it2 = it1->second.find(dataType);
    if (it2 != it1->second.end()) {
      auto it3 = it2->second.find(cmdType);
      if (it3 != it2->second.end()) {
        unsigned numR = std::get<0>(it3->second);
        unsigned numW = std::get<1>(it3->second);
        unsigned numL = std::get<2>(it3->second);
        msRuntime = m_tR * numR + m_tW * numW + m_tL * numL;
      }
    }
  }
  msRuntime *= numPass;
  return msRuntime;
}

//! @brief  Get ms runtime for SIMDRAM bit-serial operations
double
pimParamsPerf::getMsRuntimeSimdram(PimDeviceEnum deviceType, PimCmdEnum cmdType, PimDataType dataType, unsigned numPass) const
{
  std::printf("PIM-Warning: SIMDRAM performance stats not implemented yet.\n");
  return 0.0;
}

//! @brief  Get ms runtime for bit-serial PIM devices
double
pimParamsPerf::getMsRuntimeBitSerial(PimDeviceEnum deviceType, PimCmdEnum cmdType, PimDataType dataType, unsigned numPass) const
{
  double msRuntime = 0.0;
  switch (deviceType) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_AP:
    return getMsRuntimeBitsimd(deviceType, cmdType, dataType, numPass);
  case PIM_DEVICE_BITSIMD_H:
    return getMsRuntimeBitsimd(PIM_DEVICE_BITSIMD_V, cmdType, dataType, numPass);
  case PIM_DEVICE_SIMDRAM:
    return getMsRuntimeSimdram(deviceType, cmdType, dataType, numPass);
  default:
    assert(0);
  }
  return msRuntime;
}

//! @brief  Get ms runtime for func1
double
pimParamsPerf::getMsRuntimeForFunc1(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  PimDataType dataType = obj.getDataType();

  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_AP:
  case PIM_DEVICE_BITSIMD_H:
  case PIM_DEVICE_SIMDRAM:
    msRuntime = getMsRuntimeBitSerial(m_simTarget, cmdType, dataType, numPass);
    break;
  case PIM_DEVICE_FULCRUM:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double aluLatency = 0.000005; // 5ns
    unsigned bitsPerElement = obj.getBitsPerElement();
    unsigned aluBits = 32; // 32-bit ALU
    double numberOfALUOperationPerCycle = (bitsPerElement/aluBits);
    if (cmdType == PimCmdEnum::POPCOUNT) {
      numberOfALUOperationPerCycle *= 12; // 4 shifts, 4 ands, 3 add/sub, 1 mul
    }
    msRuntime = m_tR + m_tW + maxElementsPerRegion * aluLatency * numberOfALUOperationPerCycle;
    msRuntime *= numPass;
    break;
  }
  case PIM_DEVICE_BANK_LEVEL:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double aluLatency = 0.000005; // 5ns
    unsigned numALU = 2;
    unsigned bitsPerElement = obj.getBitsPerElement();
    unsigned aluBits = 32; // 32-bit ALU
    double numberOfALUOperationPerCycle = (bitsPerElement/aluBits);
    msRuntime = m_tR + m_tW + maxElementsPerRegion * aluLatency * numberOfALUOperationPerCycle / numALU;
    msRuntime *= numPass;
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
  double msRuntime = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  PimDataType dataType = obj.getDataType();

  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_AP:
  case PIM_DEVICE_BITSIMD_H:
  case PIM_DEVICE_SIMDRAM:
    msRuntime = getMsRuntimeBitSerial(m_simTarget, cmdType, dataType, numPass);
    break;
  case PIM_DEVICE_FULCRUM:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double aluLatency = 0.000005; // 5ns
    unsigned bitsPerElement = obj.getBitsPerElement();
    unsigned aluBits = 32; // 32-bit ALU
    double numberOfALUOperationPerCycle = (bitsPerElement/aluBits);
    msRuntime = 2 * m_tR + m_tW + maxElementsPerRegion * numberOfALUOperationPerCycle * aluLatency;
    msRuntime *= numPass;
    break;
  }
  case PIM_DEVICE_BANK_LEVEL:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double aluLatency = 0.000005; // 5ns
    unsigned numALU = 2;
    unsigned bitsPerElement = obj.getBitsPerElement();
    unsigned aluBits = 32; // 32-bit ALU
    double numberOfALUOperationPerCycle = (bitsPerElement/aluBits);
    msRuntime = 2 * m_tR + m_tW + maxElementsPerRegion * aluLatency * numberOfALUOperationPerCycle / numALU;
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
pimParamsPerf::getMsRuntimeForRedSum(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  PimDataType dataType = obj.getDataType();
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned numRegions = obj.getRegions().size();
  uint64_t numElements = obj.getNumElements();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();

  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_AP:
    if (dataType == PIM_INT32) {
      // Assume pop count reduction circut in tR runtime
      msRuntime = ((m_tR + m_tR) * bitsPerElement);
      msRuntime *= numPass;
      // reduction for all regions
      msRuntime += static_cast<double>(numRegions) / 3200000;
    } else if (dataType == PIM_INT8 || dataType == PIM_INT16 || dataType == PIM_INT64) {
      // todo
      std::printf("PIM-Warning: BitSIMD int8/16/64 performance stats not implemented yet.\n");
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
  case PIM_DEVICE_BANK_LEVEL:
  {
    // read a row to walker, then reduce in serial
    double aluLatency = 0.000005; // 5ns
    msRuntime = (m_tR + maxElementsPerRegion * aluLatency);
    msRuntime *= numPass;
    // reduction for all regions
    msRuntime += static_cast<double>(numRegions) / 3200000;
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
  double msRuntime = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();

  switch (m_simTarget) {
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
  case PIM_DEVICE_BANK_LEVEL:
  {
    // assume taking 1 ALU latency to write an element
    double aluLatency = 0.000005; // 5ns
    msRuntime = aluLatency * maxElementsPerRegion;
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
  double msRuntime = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned numRegions = obj.getRegions().size();

  switch (m_simTarget) {
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

