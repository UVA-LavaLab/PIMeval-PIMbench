// File: pimCmd.cpp
// PIM Functional Simulator - PIM Commands
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "pimCmd.h"
#include "pimSim.h"
#include "pimDevice.h"
#include "pimCore.h"
#include "pimResMgr.h"
#include <cstdio>
#include <cmath>
#include <unordered_map>


//! @brief  Get PIM command name from command type enum
std::string
pimCmd::getName(PimCmdEnum cmdType)
{
  static const std::unordered_map<PimCmdEnum, std::string> cmdNames = {
    { PimCmdEnum::NOOP, "noop" },
    { PimCmdEnum::ABS_V, "abs.v" },
    { PimCmdEnum::POPCOUNT_V, "popcount.v" },
    { PimCmdEnum::BROADCAST_V, "broadcast.v" },
    { PimCmdEnum::BROADCAST_H, "broadcast.H" },
    { PimCmdEnum::ADD_V, "add.v" },
    { PimCmdEnum::SUB_V, "sub.v" },
    { PimCmdEnum::MUL_V, "mul.v" },
    { PimCmdEnum::DIV_V, "div.v" },
    { PimCmdEnum::AND_V, "and.v" },
    { PimCmdEnum::OR_V, "or.v" },
    { PimCmdEnum::XOR_V, "xor.v" },
    { PimCmdEnum::XNOR_V, "xnor.v" },
    { PimCmdEnum::GT_V, "gt.v" },
    { PimCmdEnum::LT_V, "lt.v" },
    { PimCmdEnum::EQ_V, "eq.v" },
    { PimCmdEnum::MIN_V, "min.v" },
    { PimCmdEnum::MAX_V, "max.v" },
    { PimCmdEnum::REDSUM_V, "redsum.v" },
    { PimCmdEnum::REDSUM_RANGE_V, "redsum_range.v" },
    { PimCmdEnum::ROTATE_R_V, "rotate_r.v" },
    { PimCmdEnum::ROTATE_L_V, "rotate_l.v" },
    { PimCmdEnum::ROW_R, "row_r" },
    { PimCmdEnum::ROW_W, "row_w" },
    { PimCmdEnum::RREG_MOV, "rreg.mov" },
    { PimCmdEnum::RREG_SET, "rreg.set" },
    { PimCmdEnum::RREG_NOT, "rreg.not" },
    { PimCmdEnum::RREG_AND, "rreg.and" },
    { PimCmdEnum::RREG_OR, "rreg.or" },
    { PimCmdEnum::RREG_NAND, "rreg.nand" },
    { PimCmdEnum::RREG_NOR, "rreg.nor" },
    { PimCmdEnum::RREG_XOR, "rreg.xor" },
    { PimCmdEnum::RREG_XNOR, "rreg.xnor" },
    { PimCmdEnum::RREG_MAJ, "rreg.maj" },
    { PimCmdEnum::RREG_SEL, "rreg.sel" },
    { PimCmdEnum::RREG_ROTATE_R, "rreg.rotate_r" },
    { PimCmdEnum::RREG_ROTATE_L, "rreg.rotate_l" },
  };
  auto it = cmdNames.find(cmdType);
  return it != cmdNames.end() ? it->second : "unknown";
}

//! @brief  Check if two PIM objects are core aligned
bool
pimCmd::isCoreAligned(PimObjId objId1, PimObjId objId2, pimResMgr* resMgr)
{
  if (!resMgr->isValidObjId(objId1)) {
    std::printf("PIM-Error: Invalid object id %d\n", objId1);
    return false;
  }
  if (!resMgr->isValidObjId(objId2)) {
    std::printf("PIM-Error: Invalid object id %d\n", objId2);
    return false;
  }
  const pimObjInfo& obj1 = resMgr->getObjInfo(objId1);
  const pimObjInfo& obj2 = resMgr->getObjInfo(objId2);

  if (obj1.getRegions().size() != obj2.getRegions().size()) {
    std::printf("PIM-Error: Operands %d and %d have differet number of regions\n", objId1, objId2);
    return false;
  }

  for (unsigned i = 0; i < obj1.getRegions().size(); ++i) {
    const pimRegion& reg1 = obj1.getRegions()[i];
    const pimRegion& reg2 = obj2.getRegions()[i];
    if (reg1.getCoreId() != reg2.getCoreId()) {
      std::printf("PIM-Error: Operands %d and %d are not aligned on same cores\n", objId1, objId2);
      return false;
    }
  }

  return true;
}

//! @brief  Check if two PIM objects are vertically aligned
bool
pimCmd::isVAligned(PimObjId objId1, PimObjId objId2, pimResMgr* resMgr)
{
  if (!isCoreAligned(objId1, objId2, resMgr)) {
    return false;
  }

  const pimObjInfo& obj1 = resMgr->getObjInfo(objId1);
  const pimObjInfo& obj2 = resMgr->getObjInfo(objId2);

  for (unsigned i = 0; i < obj1.getRegions().size(); ++i) {
    const pimRegion& reg1 = obj1.getRegions()[i];
    const pimRegion& reg2 = obj2.getRegions()[i];

    if (reg1.getColIdx() != reg2.getColIdx() || reg1.getNumAllocCols() != reg2.getNumAllocCols()) {
      std::printf("PIM-Error: Operands %d and %d are not vertically aligned\n", objId1, objId2);
      return false;
    }
  }

  return true;
}

//! @brief  Check if two PIM objects are horizontally aligned
bool
pimCmd::isHAligned(PimObjId objId1, PimObjId objId2, pimResMgr* resMgr)
{
  if (!isCoreAligned(objId1, objId2, resMgr)) {
    return false;
  }

  const pimObjInfo& obj1 = resMgr->getObjInfo(objId1);
  const pimObjInfo& obj2 = resMgr->getObjInfo(objId2);

  for (unsigned i = 0; i < obj1.getRegions().size(); ++i) {
    const pimRegion& reg1 = obj1.getRegions()[i];
    const pimRegion& reg2 = obj2.getRegions()[i];

    if (reg1.getRowIdx() != reg2.getRowIdx() || reg1.getNumAllocRows() != reg2.getNumAllocRows()) {
      std::printf("PIM-Error: Operands %d and %d are not horizontally aligned\n", objId1, objId2);
      return false;
    }
  }

  return true;
}

//! @brief  Base class update stats
void
pimCmd::updateStats(int numPass)
{
  pimSim::get()->getStatsMgr()->recordCmd(getName(), 0.0);
}


//! @brief  PIM CMD: Functional 1-operand v-layout
bool
pimCmdFunc1V::execute(pimDevice* device)
{
  #if defined(DEBUG)
  std::printf("PIM-Info: %s (obj id %d -> %d)\n", getName().c_str(), m_src, m_dest);
  #endif

  pimResMgr* resMgr = device->getResMgr();
  if (!isVAligned(m_src, m_dest, resMgr)) {
    return false;
  }

  const pimObjInfo& objSrc = resMgr->getObjInfo(m_src);
  const pimObjInfo& objDest = resMgr->getObjInfo(m_dest);

  // todo: any data type checks

  std::unordered_map<int, int> coreIdCnt;
  int numPass = 0;
  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];
    const pimRegion& destRegion = objDest.getRegions()[i];

    if (srcRegion.getNumAllocRows() != destRegion.getNumAllocRows()) {
      std::printf("PIM-Error: Operands %d and %d do not have equal bit length for v-layout\n", m_src, m_dest);
      return false;
    }

    PimCoreId coreId = srcRegion.getCoreId();
    coreIdCnt[coreId]++;
    if (numPass < coreIdCnt[coreId]) {
      numPass = coreIdCnt[coreId];
    }

    // perform the computation
    unsigned colIdx = srcRegion.getColIdx();
    unsigned numAllocCols = srcRegion.getNumAllocCols();
    for (unsigned j = 0; j < numAllocCols; ++j) {
      switch (m_cmdType) {
      case PimCmdEnum::ABS_V:
      {
        auto operandVal = device->getCore(coreId).getB32V(srcRegion.getRowIdx(), colIdx + j);
        int operand = *reinterpret_cast<int*>(&operandVal);
        int result = std::abs(operand);
        device->getCore(coreId).setB32V(destRegion.getRowIdx(), colIdx + j,
                                        *reinterpret_cast<unsigned *>(&result));
      }
      break;
      case PimCmdEnum::POPCOUNT_V:
      {
        auto operandVal = device->getCore(coreId).getB32V(srcRegion.getRowIdx(), colIdx + j);
        int operand = *reinterpret_cast<unsigned *>(&operandVal);
        int result = 0;
        while (operand) {
          operand &= (operand - 1);
          result++;
        }
        device->getCore(coreId).setB32V(destRegion.getRowIdx(), colIdx + j,
                                        *reinterpret_cast<unsigned *>(&result));
      }
      break;
      default:
        std::printf("PIM-Error: Unexpected cmd type %d\n", m_cmdType);
        assert(0);
      }
    }
  }

  updateStats(numPass);
  return true;
}

//! @brief  Update stats for Func1V
void
pimCmdFunc1V::updateStats(int numPass)
{
  double msRuntime = 0.0;
  double tR = pimSim::get()->getParamsDram()->getNsRowRead() / 1000000.0;
  double tW = pimSim::get()->getParamsDram()->getNsRowWrite() / 1000000.0;
  double tL = pimSim::get()->getParamsDram()->getNsTCCD() / 1000000.0;

  PimDeviceEnum device = pimSim::get()->getDeviceType();
  switch (device) {
  case PIM_FUNCTIONAL:
  case PIM_DEVICE_BITSIMD_V:
  {
    switch (m_cmdType) {
    case PimCmdEnum::ABS_V: msRuntime = 98 * tR + 66 * tW + 192 * tL; break;
    case PimCmdEnum::POPCOUNT_V: msRuntime = 161 * tR + 105 * tW + 286 * tL; break;
    default:
      std::printf("PIM-Error: Unexpected cmd type %d\n", m_cmdType);
      assert(0);
    }
    break;
  }
  default:
    ;
  }
  msRuntime *= numPass;
  pimSim::get()->getStatsMgr()->recordCmd(getName(), msRuntime);
}

//! @brief  PIM CMD: Functional 2-operand v-layout
bool
pimCmdFunc2V::execute(pimDevice* device)
{ 

  #if defined(DEBUG)
  std::printf("PIM-Info: %s (obj id %d - %d -> %d)\n", getName().c_str(), m_src1, m_src2, m_dest);
  #endif

  pimResMgr* resMgr = device->getResMgr();
  if (!isVAligned(m_src1, m_src2, resMgr) || !isVAligned(m_src1, m_dest, resMgr)) {
    return false;
  }

  const pimObjInfo& objSrc1 = resMgr->getObjInfo(m_src1);
  const pimObjInfo& objSrc2 = resMgr->getObjInfo(m_src2);
  const pimObjInfo& objDest = resMgr->getObjInfo(m_dest);
  
  if (objSrc1.getDataType() != objSrc2.getDataType()) {
    std::printf("PIM-Error: Type mismatch between object %d and %d\n", m_src1, m_src2);
    return false;
  }

  if (objSrc1.getDataType() != objDest.getDataType()) {
    std::printf("PIM-Error: Cannot convert from %s to %s\n", objSrc1.getDataTypeName().c_str(), objDest.getDataTypeName().c_str());
    return false;
  }

  PimDataType dataType = objSrc1.getDataType();
  std::unordered_map<int, int> coreIdCnt;
  int numPass = 0;
  for (unsigned i = 0; i < objSrc1.getRegions().size(); ++i) {
    const pimRegion& src1Region = objSrc1.getRegions()[i];
    const pimRegion& src2Region = objSrc2.getRegions()[i];
    const pimRegion& destRegion = objDest.getRegions()[i];

    if (src1Region.getNumAllocRows() != src2Region.getNumAllocRows() || src1Region.getNumAllocRows() != destRegion.getNumAllocRows()) {
      std::printf("PIM-Error: Operands %d, %d and %d do not have equal bit length for v-layout \n", m_src1, m_src2, m_dest);
      return false;
    }

    PimCoreId coreId = src1Region.getCoreId();
    coreIdCnt[coreId]++;
    if (numPass < coreIdCnt[coreId]) {
      numPass = coreIdCnt[coreId];
    }

    // perform the computation
    unsigned colIdx = src1Region.getColIdx();
    unsigned numAllocCols = src1Region.getNumAllocCols();
    for (unsigned j = 0; j < numAllocCols; ++j) {
      if (dataType == PIM_INT32) {
        auto operandVal1 =
            device->getCore(coreId).getB32V(src1Region.getRowIdx(), colIdx + j);
        auto operandVal2 =
            device->getCore(coreId).getB32V(src2Region.getRowIdx(), colIdx + j);
        int operand1 = *reinterpret_cast<unsigned *>(&operandVal1);
        int operand2 = *reinterpret_cast<unsigned *>(&operandVal2);
        int result = 0;
        switch (m_cmdType) {
        case PimCmdEnum::ADD_V: result = operand1 + operand2; break;
        case PimCmdEnum::SUB_V: result = operand1 - operand2; break;
        case PimCmdEnum::MUL_V: result = operand1 * operand2; break;
        case PimCmdEnum::DIV_V:
          if (operand2 == 0) {
            std::printf("PIM-Error: Division by zero\n");
            return false;
          }
          result = operand1 / operand2;
          break;
        case PimCmdEnum::AND_V: result = operand1 & operand2; break;
        case PimCmdEnum::OR_V: result = operand1 | operand2; break;
        case PimCmdEnum::XOR_V: result = operand1 ^ operand2; break;
        case PimCmdEnum::XNOR_V: result = ~(operand1 ^ operand2); break;
        case PimCmdEnum::GT_V: result = operand1 > operand2 ? 1 : 0; break;
        case PimCmdEnum::LT_V: result = operand1 < operand2 ? 1 : 0; break;
        case PimCmdEnum::EQ_V: result = operand1 == operand2 ? 1 : 0; break;
        case PimCmdEnum::MIN_V: result = (operand1 < operand2) ? operand1 : operand2; break;
        case PimCmdEnum::MAX_V: result = (operand1 > operand2) ? operand1 : operand2; break;
        default:
          std::printf("PIM-Error: Unexpected cmd type %d\n", m_cmdType);
          assert(0);
        }
        device->getCore(coreId).setB32V(destRegion.getRowIdx(), colIdx + j,
                                        *reinterpret_cast<unsigned *>(&result));
      } else {
        assert(0); // todo
      }
    }
  }

  updateStats(numPass);
  return true;
}

//! @brief  Update stats for Func2V
void
pimCmdFunc2V::updateStats(int numPass)
{
  double msRuntime = 0.0;
  double tR = pimSim::get()->getParamsDram()->getNsRowRead() / 1000000.0;
  double tW = pimSim::get()->getParamsDram()->getNsRowWrite() / 1000000.0;
  double tL = pimSim::get()->getParamsDram()->getNsTCCD() / 1000000.0;

  PimDeviceEnum device = pimSim::get()->getDeviceType();
  switch (device) {
  case PIM_FUNCTIONAL:
  case PIM_DEVICE_BITSIMD_V:
  {
    // todo: support other data types
    switch (m_cmdType) {
    case PimCmdEnum::ADD_V: msRuntime = 64 * tR + 33 * tW + 161 * tL; break;
    case PimCmdEnum::SUB_V: msRuntime = 64 * tR + 33 * tW + 161 * tL; break;
    case PimCmdEnum::MUL_V: msRuntime = 2035 * tR + 1047 * tW + 4031 * tL; break;
    case PimCmdEnum::DIV_V: msRuntime = 3728 * tR + 1744 * tW + 6800 * tL; break;
    case PimCmdEnum::AND_V: msRuntime = 64 * tR + 32 * tW + 64 * tL; break;
    case PimCmdEnum::OR_V: msRuntime = 64 * tR + 32 * tW + 64 * tL; break;
    case PimCmdEnum::XOR_V: msRuntime = 64 * tR + 32 * tW + 64 * tL; break;
    case PimCmdEnum::XNOR_V: msRuntime = 64 * tR + 32 * tW + 64 * tL; break;
    case PimCmdEnum::GT_V: msRuntime = 64 * tR + 32 * tW + 66 * tL; break;
    case PimCmdEnum::LT_V: msRuntime = 64 * tR + 32 * tW + 66 * tL; break;
    case PimCmdEnum::EQ_V: msRuntime = 64 * tR + 32 * tW + 66 * tL; break;
    case PimCmdEnum::MIN_V: msRuntime = 164 * tR + 67 * tW + 258 * tL; break;
    case PimCmdEnum::MAX_V: msRuntime = 164 * tR + 67 * tW + 258 * tL; break;
    default:
      std::printf("PIM-Error: Unexpected cmd type %d\n", m_cmdType);
      assert(0);
    }
    break;
  }
  default:
    ;
  }
  msRuntime *= numPass;
  pimSim::get()->getStatsMgr()->recordCmd(getName(), msRuntime);
}


//! @brief  PIM CMD: redsum non-ranged/ranged v-layout
bool
pimCmdRedSumV::execute(pimDevice* device)
{

  #if defined(DEBUG)
  std::printf("PIM-Info: %s (obj id %d)\n", getName().c_str(), m_src);
  #endif

  pimResMgr* resMgr = device->getResMgr();

  const pimObjInfo& objSrc = resMgr->getObjInfo(m_src);
  unsigned currIdx = 0;
  for (unsigned i = 0; i < objSrc.getRegions().size() && currIdx < m_idxEnd; ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];

    PimCoreId coreId = srcRegion.getCoreId();

    unsigned colIdx = srcRegion.getColIdx();
    unsigned numAllocCols = srcRegion.getNumAllocCols();
    for (unsigned j = 0; j < numAllocCols && currIdx < m_idxEnd; ++j) {
      if (currIdx >= m_idxBegin) {
        int operand = static_cast<int>(device->getCore(coreId).getB32V(srcRegion.getRowIdx(), colIdx + j));
        *m_result += operand;
      }
      currIdx += 1;
    }
  }

  m_numElements = objSrc.getNumElements();
  unsigned bitsPerElement = objSrc.getBitsPerElement();
  m_totalBytes = m_numElements * bitsPerElement / 8;
  updateStats(1);
  return true;
}

//! @brief  Update stats for redsum
void
pimCmdRedSumV::updateStats(int numPass)
{
  double msRuntime = 0.0;
  PimDeviceEnum device = pimSim::get()->getDeviceType();
  switch (device) {
  case PIM_FUNCTIONAL:
  case PIM_DEVICE_BITSIMD_V:
    // Sequentially process all elements per CPU cycle
    msRuntime = static_cast<double>(m_numElements) / 3200000; // typical 3.2 GHz CPU
    // consider PCL
    break;
  default:
    ;
  }
  msRuntime *= numPass;
  pimSim::get()->getStatsMgr()->recordCmd(getName(), msRuntime);
}


//! @brief  PIM CMD: broadcast a value to all elements
bool
pimCmdBroadcast::execute(pimDevice* device)
{
  #if defined(DEBUG)
  std::printf("PIM-Info: %s (obj id %d value %u)\n", getName().c_str(), m_dest, m_val);
  #endif

  pimResMgr* resMgr = device->getResMgr();

  const pimObjInfo& objDest = resMgr->getObjInfo(m_dest);
  m_bitsPerElement = objDest.getBitsPerElement();
  m_numElements = objDest.getNumElements();
  m_numRegions = objDest.getRegions().size();

  assert(m_bitsPerElement == 32); // todo: support other types

  std::unordered_map<int, int> coreIdCnt;
  int numPass = 0;
  for (const auto &region : objDest.getRegions()) {
    PimCoreId coreId = region.getCoreId();
    coreIdCnt[coreId]++;
    if (numPass < coreIdCnt[coreId]) {
      numPass = coreIdCnt[coreId];
    }

    pimCore &core = device->getCore(coreId);
    unsigned colIdx = region.getColIdx();
    unsigned numAllocCols = region.getNumAllocCols();
    unsigned rowIdx = region.getRowIdx();
    m_maxElementsPerRegion = std::max(m_maxElementsPerRegion, numAllocCols / m_bitsPerElement);

    if (m_cmdType == PimCmdEnum::BROADCAST_V) {
      for (unsigned i = 0; i < numAllocCols; ++i) {
        core.setB32V(rowIdx, colIdx + i, m_val);
      }
    } else if (m_cmdType == PimCmdEnum::BROADCAST_H) {
      for (unsigned i = 0; i < numAllocCols; i += m_bitsPerElement) {
        core.setB32H(rowIdx, colIdx + i, m_val);
      }
    } else {
      assert(0);
    }
  }

  updateStats(numPass);
  return true;
}

//! @brief  Update stats for broadcast
void
pimCmdBroadcast::updateStats(int numPass)
{
  double msRuntime = 0.0;
  //double tR = pimSim::get()->getParamsDram()->getNsRowRead() / 1000000.0;
  double tW = pimSim::get()->getParamsDram()->getNsRowWrite() / 1000000.0;
  double tL = pimSim::get()->getParamsDram()->getNsTCCD() / 1000000.0;

  PimDeviceEnum device = pimSim::get()->getDeviceType();
  switch (device) {
  case PIM_FUNCTIONAL:
  case PIM_DEVICE_BITSIMD_V:
  {
    switch (m_cmdType) {
    case PimCmdEnum::BROADCAST_V:
      // For one pass: For every bit: Set SA to bit value; Write SA to row;
      msRuntime = (tW + tL) * m_bitsPerElement;
      break;
    case PimCmdEnum::BROADCAST_H:
    {
      // For one pass: For every element: 1 tCCD per byte
      unsigned maxBytesPerRegion = m_maxElementsPerRegion * (m_bitsPerElement / 8);
      msRuntime = tW + tL * maxBytesPerRegion; // for one pass
      break;
    }
    default: assert(0);
    }
    break;
  }
  default:
    ;
  }
  msRuntime *= numPass;
  pimSim::get()->getStatsMgr()->recordCmd(getName(), msRuntime);
}


//! @brief  PIM CMD: rotate right/left v-layout
bool
pimCmdRotateV::execute(pimDevice* device)
{ 

  #if defined(DEBUG)
  std::printf("PIM-Info: %s (obj id %d)\n", getName().c_str(), m_src);
  #endif

  pimResMgr* resMgr = device->getResMgr();

  const pimObjInfo& objSrc = resMgr->getObjInfo(m_src);
  m_bitsPerElement = objSrc.getBitsPerElement();
  m_numElements = objSrc.getNumElements();

  std::unordered_map<int, int> coreIdCnt;
  int numPass = 0;
  if (m_cmdType == PimCmdEnum::ROTATE_R_V) {
    unsigned carry = 0;
    for (const auto &srcRegion : objSrc.getRegions()) {
      unsigned coreId = srcRegion.getCoreId();
      coreIdCnt[coreId]++;
      numPass = std::max(numPass, coreIdCnt[coreId]);
      pimCore &core = device->getCore(coreId);

      // retrieve the values
      unsigned colIdx = srcRegion.getColIdx();
      unsigned numAllocCols = srcRegion.getNumAllocCols();
      unsigned rowIdx = srcRegion.getRowIdx();
      std::vector<unsigned> regionVector(numAllocCols);
      for (unsigned j = 0; j < numAllocCols; ++j) {
        regionVector[j] = core.getB32V(rowIdx, colIdx + j);
      }
      // Perform the rotation
      for (unsigned j = 0; j < numAllocCols; ++j) {
        int temp = regionVector[j];
        regionVector[j] = carry;
        carry = temp;
      }
      for (unsigned j = 0; j < numAllocCols; ++j) {
        core.setB32V(srcRegion.getRowIdx(), colIdx + j, regionVector[j]);
      }
    }
    if (!objSrc.getRegions().empty()) {
      const pimRegion &srcRegion = objSrc.getRegions().front();
      device->getCore(srcRegion.getCoreId())
          .setB32V(srcRegion.getRowIdx(), srcRegion.getColIdx(),
                   *reinterpret_cast<unsigned *>(&carry));
    }
  } else if (m_cmdType == PimCmdEnum::ROTATE_L_V) {
    unsigned carry = 0;
    for (unsigned i = objSrc.getRegions().size(); i > 0; --i) {
      const pimRegion &srcRegion = objSrc.getRegions()[i - 1];
      unsigned coreId = srcRegion.getCoreId();
      coreIdCnt[coreId]++;
      numPass = std::max(numPass, coreIdCnt[coreId]);
      pimCore &core = device->getCore(coreId);

      // retrieve the values
      unsigned colIdx = srcRegion.getColIdx();
      unsigned numAllocCols = srcRegion.getNumAllocCols();
      unsigned rowIdx = srcRegion.getRowIdx();
      std::vector<unsigned> regionVector(numAllocCols);
      for (unsigned j = 0; j < numAllocCols; ++j) {
        regionVector[j] = core.getB32V(rowIdx, colIdx + j);
      }
      // Perform the rotation
      for (int j = numAllocCols - 1; j >= 0; --j) {
        unsigned temp = regionVector[j];
        regionVector[j] = carry;
        carry = temp;
      }
      for (unsigned j = 0; j < numAllocCols; ++j) {
        core.setB32V(srcRegion.getRowIdx(), colIdx + j, regionVector[j]);
      }
    }
    if (!objSrc.getRegions().empty()) {
      const pimRegion &srcRegion = objSrc.getRegions().back();
      device->getCore(srcRegion.getCoreId())
          .setB32V(srcRegion.getRowIdx(),
                   srcRegion.getColIdx() + srcRegion.getNumAllocCols() - 1,
                   *reinterpret_cast<unsigned *>(&carry));
    }
  }

  m_numRegions = objSrc.getRegions().size();
  updateStats(numPass);
  return true;
}

//! @brief  Update stats for rotate
void
pimCmdRotateV::updateStats(int numPass)
{
  double msRuntime = 0.0;
  double tR = pimSim::get()->getParamsDram()->getNsRowRead() / 1000000.0;
  double tW = pimSim::get()->getParamsDram()->getNsRowWrite() / 1000000.0;
  double tL = pimSim::get()->getParamsDram()->getNsTCCD() / 1000000.0;

  PimDeviceEnum device = pimSim::get()->getDeviceType();
  switch (device) {
  case PIM_FUNCTIONAL:
  case PIM_DEVICE_BITSIMD_V:
    // rotate within subarray:
    // For every bit: Read row to SA; move SA to R1; Shift R1; Move R1 to SA; Write SA to row
    msRuntime = (tR + tW + 3 * tL) * m_bitsPerElement; // for one pass
    // boundary handling
    msRuntime += 2 * pimSim::get()->getStatsMgr()->getMsRuntimeForBytesTransfer(m_numRegions);
    break;
  default:
    ;
  }
  msRuntime *= numPass;
  pimSim::get()->getStatsMgr()->recordCmd(getName(), msRuntime);
}


//! @brief  Pim CMD: BitSIMD-V: Read a row to SA
bool
pimCmdReadRowToSa::execute(pimDevice* device)
{

  #if defined(DEBUG)
  std::printf("PIM-Info: BitSIMD-V ReadRowToSa (obj id %d ofst %u)\n", m_objId, m_ofst);
  #endif

  pimResMgr* resMgr = device->getResMgr();
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_objId);
  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];
    if (m_ofst >= srcRegion.getNumAllocRows()) {
      std::printf("PIM-Error: Row offset %u out of range [0, %u)\n", m_ofst, srcRegion.getNumAllocRows());
      return false;
    }
    PimCoreId coreId = srcRegion.getCoreId();
    device->getCore(coreId).readRow(srcRegion.getRowIdx() + m_ofst);
  }
  updateStats(1);
  return true;
}

//! @brief  Pim CMD: BitSIMD-V: Write SA to a row
bool
pimCmdWriteSaToRow::execute(pimDevice* device)
{
  
  #if defined(DEBUG)
  std::printf("PIM-Info: BitSIMD-V WriteSaToRow (obj id %d ofst %u)\n", m_objId, m_ofst);
  #endif

  pimResMgr* resMgr = device->getResMgr();
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_objId);
  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];
    if (m_ofst >= srcRegion.getNumAllocRows()) {
      std::printf("PIM-Error: Row offset %u out of range [0, %u)\n", m_ofst, srcRegion.getNumAllocRows());
      return false;
    }
    PimCoreId coreId = srcRegion.getCoreId();
    device->getCore(coreId).writeRow(srcRegion.getRowIdx() + m_ofst);
  }
  updateStats(1);
  return true;
}

//! @brief  Pim CMD: BitSIMD-V: Row reg operations
bool
pimCmdRRegOp::execute(pimDevice* device)
{

  #if defined(DEBUG)
  std::printf("PIM-Info: BitSIMD-V %s (obj-id %d dest-reg %d src-reg %d %d %d val %d)\n",
              getName().c_str(), m_objId, m_dest, m_src1, m_src2, m_src3, m_val);
  #endif

  pimResMgr* resMgr = device->getResMgr();
  const pimObjInfo& refObj = resMgr->getObjInfo(m_objId);
  for (unsigned i = 0; i < refObj.getRegions().size(); ++i) {
    const pimRegion& refRegion = refObj.getRegions()[i];
    PimCoreId coreId = refRegion.getCoreId();
    for (unsigned j = 0; j < device->getNumCols(); j++) {
      switch (m_cmdType) {
      case PimCmdEnum::RREG_MOV: 
      {
        device->getCore(coreId).getRowReg(m_dest)[j] = device->getCore(coreId).getRowReg(m_src1)[j];
        break;
      }
      case PimCmdEnum::RREG_SET: 
      {
        device->getCore(coreId).getRowReg(m_dest)[j] = m_val;
        break;
      }
      case PimCmdEnum::RREG_NOT: 
      {
        bool src = device->getCore(coreId).getRowReg(m_src1)[j];
        device->getCore(coreId).getRowReg(m_dest)[j] = !src;
        break;
      }
      case PimCmdEnum::RREG_AND: 
      {
        bool src1 = device->getCore(coreId).getRowReg(m_src1)[j];
        bool src2 = device->getCore(coreId).getRowReg(m_src2)[j];
        device->getCore(coreId).getRowReg(m_dest)[j] = (src1 & src2);
        break;
      }
      case PimCmdEnum::RREG_OR: 
      {
        bool src1 = device->getCore(coreId).getRowReg(m_src1)[j];
        bool src2 = device->getCore(coreId).getRowReg(m_src2)[j];
        device->getCore(coreId).getRowReg(m_dest)[j] = src1 | src2;
        break;
      }
      case PimCmdEnum::RREG_NAND: 
      {
        bool src1 = device->getCore(coreId).getRowReg(m_src1)[j];
        bool src2 = device->getCore(coreId).getRowReg(m_src2)[j];
        device->getCore(coreId).getRowReg(m_dest)[j] = !(src1 & src2);
        break;
      }
      case PimCmdEnum::RREG_NOR: 
      {
        bool src1 = device->getCore(coreId).getRowReg(m_src1)[j];
        bool src2 = device->getCore(coreId).getRowReg(m_src2)[j];
        device->getCore(coreId).getRowReg(m_dest)[j] = !(src1 | src2);
        break;
      }
      case PimCmdEnum::RREG_XOR: 
      {
        bool src1 = device->getCore(coreId).getRowReg(m_src1)[j];
        bool src2 = device->getCore(coreId).getRowReg(m_src2)[j];
        device->getCore(coreId).getRowReg(m_dest)[j] = src1 ^ src2;
        break;
      }
      case PimCmdEnum::RREG_XNOR:
      {
        bool src1 = device->getCore(coreId).getRowReg(m_src1)[j];
        bool src2 = device->getCore(coreId).getRowReg(m_src2)[j];
        device->getCore(coreId).getRowReg(m_dest)[j] = !(src1 ^ src2);
        break;
      }
      case PimCmdEnum::RREG_MAJ:
      {
        bool src1 = device->getCore(coreId).getRowReg(m_src1)[j];
        bool src2 = device->getCore(coreId).getRowReg(m_src2)[j];
        bool src3 = device->getCore(coreId).getRowReg(m_src3)[j];
        device->getCore(coreId).getRowReg(m_dest)[j] =
            ((src1 & src2) || (src1 & src3) || (src2 & src3));
        break;
      }
      case PimCmdEnum::RREG_SEL: 
      {
        bool cond = device->getCore(coreId).getRowReg(m_src1)[j];
        bool src2 = device->getCore(coreId).getRowReg(m_src2)[j];
        bool src3 = device->getCore(coreId).getRowReg(m_src3)[j];
        device->getCore(coreId).getRowReg(m_dest)[j] = (cond ? src2 : src3);
        break;
      }
      default:
        std::printf("PIM-Error: Unexpected cmd type %d\n", m_cmdType);
        assert(0);
      }
    }
  }
  updateStats(1);
  return true;
}


//! @brief  Pim CMD: BitSIMD-V: row reg rotate right/left by one step
bool
pimCmdRRegRotate::execute(pimDevice* device)
{

  #if defined(DEBUG)
  std::printf("PIM-Info: BitSIMD-V %s (obj-id %d src-reg %d)\n", getName().c_str(), m_objId, m_dest);
  #endif
  
  pimResMgr* resMgr = device->getResMgr();
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_objId);
  if (m_cmdType == PimCmdEnum::RREG_ROTATE_R) {  // Right Rotate
    bool prevVal = 0;
    for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
      const pimRegion &srcRegion = objSrc.getRegions()[i];
      PimCoreId coreId = srcRegion.getCoreId();
      for (unsigned j = 0; j < srcRegion.getNumAllocCols(); ++j) {
        unsigned colIdx = srcRegion.getColIdx() + j;
        bool tmp = device->getCore(coreId).getRowReg(m_dest)[colIdx];
        device->getCore(coreId).getRowReg(m_dest)[colIdx] = prevVal;
        prevVal = tmp;
      }
    }
    // write the last val to the first place
    const pimRegion &firstRegion = objSrc.getRegions().front();
    PimCoreId firstCoreId = firstRegion.getCoreId();
    unsigned firstColIdx = firstRegion.getColIdx();
    device->getCore(firstCoreId).getRowReg(m_dest)[firstColIdx] = prevVal;
  } else if (m_cmdType == PimCmdEnum::RREG_ROTATE_L) {  // Left Rotate
    bool prevVal = 0;
    for (unsigned i = objSrc.getRegions().size(); i > 0; --i) {
      const pimRegion &srcRegion = objSrc.getRegions()[i - 1];
      PimCoreId coreId = srcRegion.getCoreId();
      for (unsigned j = srcRegion.getNumAllocCols(); j > 0; --j) {
        unsigned colIdx = srcRegion.getColIdx() + j - 1;
        bool tmp = device->getCore(coreId).getRowReg(m_dest)[colIdx];
        device->getCore(coreId).getRowReg(m_dest)[colIdx] = prevVal;
        prevVal = tmp;
      }
    }
    // write the first val to the last place
    const pimRegion &lastRegion = objSrc.getRegions().back();
    PimCoreId lastCoreId = lastRegion.getCoreId();
    unsigned lastColIdx = lastRegion.getColIdx() + lastRegion.getNumAllocCols() - 1;
    device->getCore(lastCoreId).getRowReg(m_dest)[lastColIdx] = prevVal;
  }

  updateStats(1);
  return true;
}

