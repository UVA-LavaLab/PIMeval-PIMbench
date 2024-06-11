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
  case PIM_DEVICE_BITSIMD_V_AP: return true;
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

//! @brief  Get ms runtime for func1
double
pimParamsPerf::getMsRuntimeForFunc1(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  PimDataType dataType = obj.getDataType();

  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_H:
  {
    if (dataType == PIM_INT32) {
      switch (cmdType) {
      case PimCmdEnum::ABS: msRuntime = 98 * m_tR + 66 * m_tW + 192 * m_tL; break;
      case PimCmdEnum::POPCOUNT: msRuntime = 161 * m_tR + 105 * m_tW + 286 * m_tL; break;
      default:
        assert(0);
      }
    } else {
      assert(0);
    }
    msRuntime *= numPass;
    break;
  }
  case PIM_DEVICE_BITSIMD_V_AP:
  {
    if (dataType == PIM_INT32) {
      switch (cmdType) {
      case PimCmdEnum::ABS: msRuntime = 98 * m_tR + 66 * m_tW + 320 * m_tL; break;
      case PimCmdEnum::POPCOUNT: msRuntime = 161 * m_tR + 105 * m_tW + 318 * m_tL; break;
      default:
        assert(0);
      }
    } else {
      assert(0);
    }
    msRuntime *= numPass;
    break;
  }
  case PIM_DEVICE_FULCRUM:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double aluLatency = 0.000005; // 5ns
    double numberOfALUOperation = 1;
    if (cmdType == PimCmdEnum::POPCOUNT) {
      numberOfALUOperation = 12; // 4 shifts, 4 ands, 3 add/sub, 1 mul
    }
    msRuntime = m_tR + m_tW + maxElementsPerRegion * aluLatency * numberOfALUOperation;
    msRuntime *= numPass;
    break;
  }
  case PIM_DEVICE_BANK_LEVEL:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double aluLatency = 0.000005; // 5ns
    unsigned numALU = 2;
    msRuntime = m_tR + m_tW + maxElementsPerRegion * aluLatency / numALU;
    msRuntime *= numPass;
    break;
  }
  default:
    msRuntime = 1000000;
  }
  return msRuntime;
}

//! @brief  Get ms runtime for func1 with immediate value
double
pimParamsPerf::getMsRuntimeForFunc1Imm(PimCmdEnum cmdType, const pimObjInfo& obj, unsigned immValue) const
{
  double msRuntime = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  PimDataType dataType = obj.getDataType();

  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_H:
  case PIM_DEVICE_BITSIMD_V_AP:
  {
    if (dataType == PIM_INT32) {
      switch (cmdType)
      {
        case PimCmdEnum::SHIFT_BITS_RIGHT:
        case PimCmdEnum::SHIFT_BITS_LEFT:
        {
          // Assuming fulcrum ALU has shift circuit, so it can perform one 32-bit shift in one cycle.
          unsigned bitsPerElement = obj.getBitsPerElement();
          msRuntime = (m_tR + m_tW) * bitsPerElement * immValue;
          msRuntime *= numPass;
        }
        break;
        default:
          assert(0);
      }
    }
  }
  case PIM_DEVICE_FULCRUM:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double aluLatency = 0.000005; // 5ns
    double numberOfALUOperation = 1;
    switch (cmdType)
    {
      case PimCmdEnum::SHIFT_BITS_RIGHT:
      case PimCmdEnum::SHIFT_BITS_LEFT:
      {
        // Assuming fulcrum ALU has shift circuit, so it can perform one 32-bit shift in one cycle.
        msRuntime = m_tR + m_tW + maxElementsPerRegion * aluLatency * numberOfALUOperation;
        msRuntime *= numPass;
      }
      break;
      default:
        assert(0);
    }
  }
  case PIM_DEVICE_BANK_LEVEL:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double aluLatency = 0.000005; // 5ns
    unsigned numALU = 2;
    double numberOfALUOperation = 1;
    switch (cmdType)
    {
      case PimCmdEnum::SHIFT_BITS_RIGHT:
      case PimCmdEnum::SHIFT_BITS_LEFT:
      {
        // Assuming bank level ALU has shift circuit, so it can perform one 32-bit shift in one cycle.
        msRuntime = m_tR + m_tW + maxElementsPerRegion * aluLatency * numberOfALUOperation / numALU;
        msRuntime *= numPass;
      }
      break;
      default:
        assert(0);
    }
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
  case PIM_DEVICE_BITSIMD_H:
  {
    if (dataType == PIM_INT32) {
      switch (cmdType) {
      case PimCmdEnum::ADD: msRuntime = 64 * m_tR + 33 * m_tW + 161 * m_tL; break;
      case PimCmdEnum::SUB: msRuntime = 64 * m_tR + 33 * m_tW + 161 * m_tL; break;
      case PimCmdEnum::MUL: msRuntime = 1940 * m_tR + 1095 * m_tW + 3606 * m_tL; break;
      case PimCmdEnum::DIV: msRuntime = 3168 * m_tR + 1727 * m_tW + 4257 * m_tL; break;
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
    } else if (dataType == PIM_FP32) {
      switch (cmdType) {
      case PimCmdEnum::ADD: msRuntime = 1331 * m_tR + 685 * m_tW + 1687 * m_tL; break;
      case PimCmdEnum::SUB: msRuntime = 1331 * m_tR + 685 * m_tW + 1687 * m_tL; break;
      case PimCmdEnum::MUL: msRuntime = 1852 * m_tR + 1000 * m_tW + 3054 * m_tL; break;
      case PimCmdEnum::DIV: msRuntime = 2744 * m_tR + 1458 * m_tW + 4187 * m_tL; break;
      default:
        assert(0);
      }
    } else {
      assert(0);
    }
    msRuntime *= numPass;
    break;
  }
  case PIM_DEVICE_BITSIMD_V_AP:
  {
    if (dataType == PIM_INT32) {
      switch (cmdType) {
      case PimCmdEnum::ADD: msRuntime = 64 * m_tR + 33 * m_tW + 161 * m_tL; break;
      case PimCmdEnum::SUB: msRuntime = 64 * m_tR + 33 * m_tW + 161 * m_tL; break;
      case PimCmdEnum::MUL: msRuntime = 4291 * m_tR + 1799 * m_tW + 7039 * m_tL; break;
      case PimCmdEnum::DIV: msRuntime = 3728 * m_tR + 1744 * m_tW + 6800 * m_tL; break;
      case PimCmdEnum::AND: msRuntime = 64 * m_tR + 32 * m_tW + 64 * m_tL; break;
      case PimCmdEnum::OR: msRuntime = 64 * m_tR + 32 * m_tW + 128 * m_tL; break;
      case PimCmdEnum::XOR: msRuntime = 64 * m_tR + 32 * m_tW + 128 * m_tL; break;
      case PimCmdEnum::XNOR: msRuntime = 64 * m_tR + 32 * m_tW + 64 * m_tL; break;
      case PimCmdEnum::GT: msRuntime = 64 * m_tR + 32 * m_tW + 66 * m_tL; break;
      case PimCmdEnum::LT: msRuntime = 64 * m_tR + 32 * m_tW + 66 * m_tL; break;
      case PimCmdEnum::EQ: msRuntime = 64 * m_tR + 32 * m_tW + 66 * m_tL; break;
      case PimCmdEnum::MIN: msRuntime = 164 * m_tR + 67 * m_tW + 261 * m_tL; break;
      case PimCmdEnum::MAX: msRuntime = 164 * m_tR + 67 * m_tW + 261 * m_tL; break;
      default:
        assert(0);
      }
    } else if (dataType == PIM_FP32) {
      switch (cmdType) {
      case PimCmdEnum::ADD: msRuntime = 1597 * m_tR + 822 * m_tW + 2024 * m_tL; break;
      case PimCmdEnum::SUB: msRuntime = 1597 * m_tR + 822 * m_tW + 2024 * m_tL; break;
      case PimCmdEnum::MUL: msRuntime = 2222 * m_tR + 1200 * m_tW + 3664 * m_tL; break;
      case PimCmdEnum::DIV: msRuntime = 3292 * m_tR + 1749 * m_tW + 5024 * m_tL; break;
      default:
        assert(0);
      }
    } else {
      assert(0);
    }
    msRuntime *= numPass;
    break;
  }
  case PIM_DEVICE_FULCRUM:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double aluLatency = 0.000005; // 5ns
    msRuntime = 2 * m_tR + m_tW + maxElementsPerRegion * aluLatency;
    msRuntime *= numPass;
    break;
  }
  case PIM_DEVICE_BANK_LEVEL:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double aluLatency = 0.000005; // 5ns
    unsigned numALU = 2;
    msRuntime = 2 * m_tR + m_tW + maxElementsPerRegion * aluLatency / numALU;
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
  unsigned numElements = obj.getNumElements();
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
    } else {
      assert(0);
    }
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
  case PIM_DEVICE_BITSIMD_H:
  {
    // For one pass: For every element: 1 tCCD per byte
    unsigned maxBytesPerRegion = maxElementsPerRegion * (bitsPerElement / 8);
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

