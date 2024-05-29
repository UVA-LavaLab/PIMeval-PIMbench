// File: pimParamsPerf.cc
// PIM Functional Simulator - Performance parameters
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "pimParamsPerf.h"
#include "pimSim.h"


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
  int numCores = static_cast<int>(pimSim::get()->getNumCores());
  int numActiveRank = std::min(numCores, 16); // Up to 16 ranks
  double typicalRankBW = m_paramsDram->getTypicalRankBW(); // GB/s
  double totalMsRuntime = static_cast<double>(numBytes) / (typicalRankBW * numActiveRank * 1024 * 1024 * 1024 / 1000);
  return totalMsRuntime;
}

//! @brief  Get ms runtime for func1
double
pimParamsPerf::getMsRuntimeForFunc1(PimCmdEnum cmdType) const
{
  double msRuntime = 0.0;

  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  {
    switch (cmdType) {
    case PimCmdEnum::ABS: msRuntime = 98 * m_tR + 66 * m_tW + 192 * m_tL; break;
    case PimCmdEnum::POPCOUNT: msRuntime = 161 * m_tR + 105 * m_tW + 286 * m_tL; break;
    default:
      assert(0);
    }
    break;
  }
  case PIM_DEVICE_FULCRUM:
  case PIM_DEVICE_BANK_LEVEL:
  default:
    msRuntime = 1000000;
  }
  return msRuntime;
}

//! @brief  Get ms runtime for func2
double
pimParamsPerf::getMsRuntimeForFunc2(PimCmdEnum cmdType) const
{
  double msRuntime = 0.0;

  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  {
    // todo: support other data types
    switch (cmdType) {
    case PimCmdEnum::ADD: msRuntime = 64 * m_tR + 33 * m_tW + 161 * m_tL; break;
    case PimCmdEnum::SUB: msRuntime = 64 * m_tR + 33 * m_tW + 161 * m_tL; break;
    case PimCmdEnum::MUL: msRuntime = 2035 * m_tR + 1047 * m_tW + 4031 * m_tL; break;
    case PimCmdEnum::DIV: msRuntime = 3728 * m_tR + 1744 * m_tW + 6800 * m_tL; break;
    case PimCmdEnum::AND: msRuntime = 64 * m_tR + 32 * m_tW + 64 * m_tL; break;
    case PimCmdEnum::OR: msRuntime = 64 * m_tR + 32 * m_tW + 64 * m_tL; break;
    case PimCmdEnum::XOR: msRuntime = 64 * m_tR + 32 * m_tW + 64 * m_tL; break;
    case PimCmdEnum::XNOR: msRuntime = 64 * m_tR + 32 * m_tW + 64 * m_tL; break;
    case PimCmdEnum::GT: msRuntime = 64 * m_tR + 32 * m_tW + 66 * m_tL; break;
    case PimCmdEnum::LT: msRuntime = 64 * m_tR + 32 * m_tW + 66 * m_tL; break;
    case PimCmdEnum::EQ: msRuntime = 64 * m_tR + 32 * m_tW + 66 * m_tL; break;
    case PimCmdEnum::MIN: msRuntime = 164 * m_tR + 67 * m_tW + 258 * m_tL; break;
    case PimCmdEnum::MAX: msRuntime = 164 * m_tR + 67 * m_tW + 258 * m_tL; break;
    default:
      assert(0);
    }
    break;
  }
  case PIM_DEVICE_FULCRUM:
  case PIM_DEVICE_BANK_LEVEL:
  default:
    msRuntime = 1e10;
  }
  return msRuntime;
}

//! @brief  Get ms runtime for reduction sum
double
pimParamsPerf::getMsRuntimeForRedSum(PimCmdEnum cmdType, unsigned numElements) const
{
  double msRuntime = 0.0;

  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V:
    // Sequentially process all elements per CPU cycle
    msRuntime = static_cast<double>(numElements) / 3200000; // typical 3.2 GHz CPU
    // consider PCL
    break;
  case PIM_DEVICE_FULCRUM:
  case PIM_DEVICE_BANK_LEVEL:
  default:
    msRuntime = 1e10;
  }

  return msRuntime;
}

//! @brief  Get ms runtime for broadcast
double
pimParamsPerf::getMsRuntimeForBroadcast(PimCmdEnum cmdType, bool isVLayout, unsigned bitsPerElement, unsigned maxElementsPerRegion) const
{
  double msRuntime = 0.0;

  if (cmdType == PimCmdEnum::BROADCAST && isVLayout) {
    switch (m_simTarget) {
    case PIM_DEVICE_BITSIMD_V:
    {
      // For one pass: For every bit: Set SA to bit value; Write SA to row;
      msRuntime = (m_tW + m_tL) * bitsPerElement;
      break;
    }
    case PIM_DEVICE_FULCRUM:
    case PIM_DEVICE_BANK_LEVEL:
    default:
      msRuntime = 1e10;
    }

  } else if (cmdType == PimCmdEnum::BROADCAST && !isVLayout) {
    switch (m_simTarget) {
    case PIM_DEVICE_BITSIMD_V:
    {
      // For one pass: For every element: 1 tCCD per byte
      unsigned maxBytesPerRegion = maxElementsPerRegion * (bitsPerElement / 8);
      msRuntime = m_tW + m_tL * maxBytesPerRegion; // for one pass
      break;
      break;
    }
    case PIM_DEVICE_FULCRUM:
    case PIM_DEVICE_BANK_LEVEL:
    default:
      msRuntime = 1e10;
    }
  } else {
    assert(0);
  }

  return msRuntime;
}

//! @brief  Get ms runtime for rotate
double
pimParamsPerf::getMsRuntimeForRotate(PimCmdEnum cmdType, unsigned bitsPerElement, unsigned numRegions) const
{
  double msRuntime = 0.0;

  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V:
    // rotate within subarray:
    // For every bit: Read row to SA; move SA to R1; Shift R1; Move R1 to SA; Write SA to row
    msRuntime = (m_tR + m_tW + 3 * m_tL) * bitsPerElement; // for one pass
    // boundary handling
    msRuntime += 2 * getMsRuntimeForBytesTransfer(numRegions);
    break;
  case PIM_DEVICE_FULCRUM:
  case PIM_DEVICE_BANK_LEVEL:
  default:
    msRuntime = 1e10;
  }

  return msRuntime;
}

