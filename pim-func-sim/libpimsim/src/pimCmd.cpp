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
#include <bitset>
#include <unordered_map>


//! @brief  Get PIM command name from command type enum
std::string
pimCmd::getName(PimCmdEnum cmdType, const std::string& suffix)
{
  static const std::unordered_map<PimCmdEnum, std::string> cmdNames = {
    { PimCmdEnum::NOOP, "noop" },
    { PimCmdEnum::ABS, "abs" },
    { PimCmdEnum::POPCOUNT, "popcount" },
    { PimCmdEnum::BROADCAST, "broadcast" },
    { PimCmdEnum::ADD, "add" },
    { PimCmdEnum::SUB, "sub" },
    { PimCmdEnum::MUL, "mul" },
    { PimCmdEnum::DIV, "div" },
    { PimCmdEnum::AND, "and" },
    { PimCmdEnum::OR, "or" },
    { PimCmdEnum::XOR, "xor" },
    { PimCmdEnum::XNOR, "xnor" },
    { PimCmdEnum::GT, "gt" },
    { PimCmdEnum::LT, "lt" },
    { PimCmdEnum::EQ, "eq" },
    { PimCmdEnum::MIN, "min" },
    { PimCmdEnum::MAX, "max" },
    { PimCmdEnum::REDSUM, "redsum" },
    { PimCmdEnum::REDSUM_RANGE, "redsum_range" },
    { PimCmdEnum::ROTATE_R, "rotate_r" },
    { PimCmdEnum::ROTATE_L, "rotate_l" },
    { PimCmdEnum::SHIFT_R, "shift_r" },
    { PimCmdEnum::SHIFT_L, "shift_l" },
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
  return it != cmdNames.end() ? it->second + suffix : "unknown";
}

//! @brief  Check if an obj ID is valid
bool
pimCmd::isValidObjId(pimResMgr* resMgr, PimObjId objId) const
{
  if (!resMgr->isValidObjId(objId)) {
    std::printf("PIM-Error: Invalid object id %d\n", objId);
    return false;
  }
  return true;
}

//! @brief  Check if two objects are associated
bool
pimCmd::isAssociated(const pimObjInfo& obj1, const pimObjInfo& obj2) const
{
  if (obj1.getRefObjId() != obj2.getRefObjId()) {
    std::printf("PIM-Error: Object id %d and %d are not associated\n", obj1.getObjId(), obj2.getObjId());
    return false;
  }
  return true;
}

//! @brief  Utility: Get number of elements in a region
unsigned
pimCmd::getNumElementsInRegion(const pimRegion& region, unsigned bitsPerElement) const
{
  unsigned numAllocRows = region.getNumAllocRows();
  unsigned numAllocCols = region.getNumAllocCols();
  return (uint64_t)numAllocRows * numAllocCols / bitsPerElement;
}

//! @brief  PIM CMD: Functional 1-operand
bool
pimCmdFunc1::execute(pimDevice* device)
{
  #if defined(DEBUG)
  std::printf("PIM-Info: %s (obj id %d -> %d)\n", getName().c_str(), m_src, m_dest);
  #endif

  pimResMgr* resMgr = device->getResMgr();
  if (!isValidObjId(resMgr, m_src) || !isValidObjId(resMgr, m_dest)) {
    return false;
  }
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_src);
  const pimObjInfo& objDest = resMgr->getObjInfo(m_dest);
  if (!isAssociated(objSrc, objDest)) {
    return false;
  }
  PimDataType dataType = objSrc.getDataType();
  bool isVLayout = objSrc.isVLayout();

  // todo: any data type checks
  unsigned bitsPerElement = objSrc.getBitsPerElement();

  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];
    const pimRegion& destRegion = objDest.getRegions()[i];

    PimCoreId coreId = srcRegion.getCoreId();
    pimCore& core = device->getCore(coreId);

    // perform the computation
    unsigned numElementsInRegion = getNumElementsInRegion(srcRegion, bitsPerElement);
    for (unsigned j = 0; j < numElementsInRegion; ++j) {
      if (dataType == PIM_INT32) {
        auto locSrc = locateNthB32(srcRegion, isVLayout, j);
        auto locDest = locateNthB32(destRegion, isVLayout, j);

        switch (m_cmdType) {
        case PimCmdEnum::ABS:
        {
          auto operandBits = getB32(core, isVLayout, locSrc.first, locSrc.second);
          int operand = *reinterpret_cast<int*>(&operandBits);
          int result = std::abs(operand);
          setB32(core, isVLayout, locDest.first, locDest.second,
                 *reinterpret_cast<unsigned *>(&result));
        }
        break;
        case PimCmdEnum::POPCOUNT:
        {
          auto operandBits = getB32(core, isVLayout, locSrc.first, locSrc.second);
          int result = std::bitset<32>(operandBits).count();
          setB32(core, isVLayout, locDest.first, locDest.second,
                 *reinterpret_cast<unsigned *>(&result));
        }
        break;
        default:
          std::printf("PIM-Error: Unexpected cmd type %d\n", m_cmdType);
          assert(0);
        }
      } else {
        assert(0); // todo: data type
      }
    }
  }

  // Update stats
  double msRuntime = pimSim::get()->getParamsPerf()->getMsRuntimeForFunc1(m_cmdType, objSrc);
  pimSim::get()->getStatsMgr()->recordCmd(getName(dataType, isVLayout), msRuntime);
  return true;
}

//! @brief  PIM CMD: Functional 2-operand
bool
pimCmdFunc2::execute(pimDevice* device)
{ 
  #if defined(DEBUG)
  std::printf("PIM-Info: %s (obj id %d - %d -> %d)\n", getName().c_str(), m_src1, m_src2, m_dest);
  #endif

  pimResMgr* resMgr = device->getResMgr();
  if (!isValidObjId(resMgr, m_src1) || !isValidObjId(resMgr, m_src2) || !isValidObjId(resMgr, m_dest)) {
    return false;
  }
  const pimObjInfo& objSrc1 = resMgr->getObjInfo(m_src1);
  const pimObjInfo& objSrc2 = resMgr->getObjInfo(m_src2);
  const pimObjInfo& objDest = resMgr->getObjInfo(m_dest);
  if (!isAssociated(objSrc1, objSrc2) || !isAssociated(objSrc1, objDest)) {
    return false;
  }
  PimDataType dataType = objSrc1.getDataType();
  bool isVLayout = objSrc1.isVLayout();
  
  if (objSrc1.getDataType() != objSrc2.getDataType()) {
    std::printf("PIM-Error: Type mismatch between object %d and %d\n", m_src1, m_src2);
    return false;
  }

  if (objSrc1.getDataType() != objDest.getDataType()) {
    std::printf("PIM-Error: Cannot convert from %s to %s\n", objSrc1.getDataTypeName().c_str(), objDest.getDataTypeName().c_str());
    return false;
  }
  // todo: other data types
  unsigned bitsPerElement = objSrc1.getBitsPerElement();

  for (unsigned i = 0; i < objSrc1.getRegions().size(); ++i) {
    const pimRegion& src1Region = objSrc1.getRegions()[i];
    const pimRegion& src2Region = objSrc2.getRegions()[i];
    const pimRegion& destRegion = objDest.getRegions()[i];

    PimCoreId coreId = src1Region.getCoreId();
    pimCore& core = device->getCore(coreId);

    // perform the computation
    unsigned numElementsInRegion = getNumElementsInRegion(src1Region, bitsPerElement);
    for (unsigned j = 0; j < numElementsInRegion; ++j) {
      auto locSrc1 = locateNthB32(src1Region, isVLayout, j);
      auto locSrc2 = locateNthB32(src2Region, isVLayout, j);
      auto locDest = locateNthB32(destRegion, isVLayout, j);

      if (dataType == PIM_INT32) {
        auto operandBits1 = getB32(core, isVLayout, locSrc1.first, locSrc1.second);
        auto operandBits2 = getB32(core, isVLayout, locSrc2.first, locSrc2.second);
        int operand1 = *reinterpret_cast<unsigned *>(&operandBits1);
        int operand2 = *reinterpret_cast<unsigned *>(&operandBits2);
        int result = 0;
        switch (m_cmdType) {
        case PimCmdEnum::ADD: result = operand1 + operand2; break;
        case PimCmdEnum::SUB: result = operand1 - operand2; break;
        case PimCmdEnum::MUL: result = operand1 * operand2; break;
        case PimCmdEnum::DIV:
          if (operand2 == 0) {
            std::printf("PIM-Error: Division by zero\n");
            return false;
          }
          result = operand1 / operand2;
          break;
        case PimCmdEnum::AND: result = operand1 & operand2; break;
        case PimCmdEnum::OR: result = operand1 | operand2; break;
        case PimCmdEnum::XOR: result = operand1 ^ operand2; break;
        case PimCmdEnum::XNOR: result = ~(operand1 ^ operand2); break;
        case PimCmdEnum::GT: result = operand1 > operand2 ? 1 : 0; break;
        case PimCmdEnum::LT: result = operand1 < operand2 ? 1 : 0; break;
        case PimCmdEnum::EQ: result = operand1 == operand2 ? 1 : 0; break;
        case PimCmdEnum::MIN: result = (operand1 < operand2) ? operand1 : operand2; break;
        case PimCmdEnum::MAX: result = (operand1 > operand2) ? operand1 : operand2; break;
        default:
          std::printf("PIM-Error: Unexpected cmd type %d\n", m_cmdType);
          assert(0);
        }
        setB32(core, isVLayout, locDest.first, locDest.second,
               *reinterpret_cast<unsigned *>(&result));
      } else if (dataType == PIM_FP32) {
        auto operandBits1 = getB32(core, isVLayout, locSrc1.first, locSrc1.second);
        auto operandBits2 = getB32(core, isVLayout, locSrc2.first, locSrc2.second);
        float operand1 = *reinterpret_cast<float *>(&operandBits1);
        float operand2 = *reinterpret_cast<float *>(&operandBits2);
        float result = 0;
        switch (m_cmdType) {
        case PimCmdEnum::ADD: result = operand1 + operand2; break;
        case PimCmdEnum::SUB: result = operand1 - operand2; break;
        case PimCmdEnum::MUL: result = operand1 * operand2; break;
        case PimCmdEnum::DIV:
          if (operand2 == 0) {
            std::printf("PIM-Error: Division by zero\n");
            return false;
          }
          result = operand1 / operand2;
          break;
        default:
          std::printf("PIM-Error: Unsupported FP32 cmd type %d\n", m_cmdType);
          assert(0);
        }
        setB32(core, isVLayout, locDest.first, locDest.second,
               *reinterpret_cast<unsigned *>(&result));
      } else {
        assert(0); // todo: data type
      }
    }
  }

  // Update stats
  double msRuntime = pimSim::get()->getParamsPerf()->getMsRuntimeForFunc2(m_cmdType, objSrc1);
  pimSim::get()->getStatsMgr()->recordCmd(getName(dataType, isVLayout), msRuntime);
  return true;
}

//! @brief  PIM CMD: redsum non-ranged/ranged
bool
pimCmdRedSum::execute(pimDevice* device)
{
  #if defined(DEBUG)
  std::printf("PIM-Info: %s (obj id %d)\n", getName().c_str(), m_src);
  #endif

  pimResMgr* resMgr = device->getResMgr();
  if (!isValidObjId(resMgr, m_src) || !m_result) {
    return false;
  }
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_src);
  PimDataType dataType = objSrc.getDataType();
  bool isVLayout = objSrc.isVLayout();

  unsigned bitsPerElement = objSrc.getBitsPerElement();

  unsigned currIdx = 0;
  for (unsigned i = 0; i < objSrc.getRegions().size() && currIdx < m_idxEnd; ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];

    PimCoreId coreId = srcRegion.getCoreId();
    pimCore& core = device->getCore(coreId);

    unsigned numElementsInRegion = getNumElementsInRegion(srcRegion, bitsPerElement);
    for (unsigned j = 0; j < numElementsInRegion && currIdx < m_idxEnd; ++j) {
      if (currIdx >= m_idxBegin) {
        auto locSrc = locateNthB32(srcRegion, isVLayout, j);
        auto operandBits = getB32(core, isVLayout, locSrc.first, locSrc.second);
        int operand = *reinterpret_cast<int*>(&operandBits);
        *m_result += operand;
      }
      currIdx += 1;
    }
  }

  // Update stats
  double msRuntime = pimSim::get()->getParamsPerf()->getMsRuntimeForRedSum(m_cmdType, objSrc);
  pimSim::get()->getStatsMgr()->recordCmd(getName(dataType, isVLayout), msRuntime);
  return true;
}

//! @brief  PIM CMD: broadcast a value to all elements
bool
pimCmdBroadcast::execute(pimDevice* device)
{
  #if defined(DEBUG)
  std::printf("PIM-Info: %s (obj id %d value %u)\n", getName().c_str(), m_dest, m_val);
  #endif

  pimResMgr* resMgr = device->getResMgr();
  if (!isValidObjId(resMgr, m_dest)) {
    return false;
  }
  const pimObjInfo& objDest = resMgr->getObjInfo(m_dest);
  PimDataType dataType = objDest.getDataType();
  bool isVLayout = objDest.isVLayout();

  unsigned bitsPerElement = objDest.getBitsPerElement();

  assert(bitsPerElement == 32); // todo: support other types

  for (const auto &region : objDest.getRegions()) {
    PimCoreId coreId = region.getCoreId();
    pimCore &core = device->getCore(coreId);

    unsigned numElementsInRegion = getNumElementsInRegion(region, bitsPerElement);

    for (unsigned j = 0; j < numElementsInRegion; ++j) {
      auto locDest = locateNthB32(region, isVLayout, j);
      setB32(core, isVLayout, locDest.first, locDest.second,
             *reinterpret_cast<unsigned *>(&m_val));
    }
  }

  // Update stats
  double msRuntime = pimSim::get()->getParamsPerf()->getMsRuntimeForBroadcast(m_cmdType, objDest);
  pimSim::get()->getStatsMgr()->recordCmd(getName(dataType, isVLayout), msRuntime);
  return true;
}

//! @brief  PIM CMD: rotate right/left
bool
pimCmdRotate::execute(pimDevice* device)
{ 
  #if defined(DEBUG)
  std::printf("PIM-Info: %s (obj id %d)\n", getName().c_str(), m_src);
  #endif

  pimResMgr* resMgr = device->getResMgr();
  if (!isValidObjId(resMgr, m_src)) {
    return false;
  }
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_src);
  PimDataType dataType = objSrc.getDataType();
  bool isVLayout = objSrc.isVLayout();
  unsigned bitsPerElement = objSrc.getBitsPerElement();

  if (m_cmdType == PimCmdEnum::ROTATE_R || m_cmdType == PimCmdEnum::SHIFT_R) {
    unsigned carry = 0;
    for (const auto &srcRegion : objSrc.getRegions()) {
      unsigned coreId = srcRegion.getCoreId();
      pimCore &core = device->getCore(coreId);

      // retrieve the values
      unsigned numElementsInRegion = getNumElementsInRegion(srcRegion, bitsPerElement);
      std::vector<unsigned> regionVector(numElementsInRegion);
      for (unsigned j = 0; j < numElementsInRegion; ++j) {
        auto locSrc = locateNthB32(srcRegion, isVLayout, j);
        regionVector[j] = getB32(core, isVLayout, locSrc.first, locSrc.second);
      }
      // Perform the rotation
      for (unsigned j = 0; j < numElementsInRegion; ++j) {
        int temp = regionVector[j];
        regionVector[j] = carry;
        carry = temp;
      }
      for (unsigned j = 0; j < numElementsInRegion; ++j) {
        auto locSrc = locateNthB32(srcRegion, isVLayout, j);
        setB32(core, isVLayout, locSrc.first, locSrc.second, regionVector[j]);
      }
    }
    if (m_cmdType == PimCmdEnum::SHIFT_R) {
      carry = 0; // fill with zero
    }
    if (!objSrc.getRegions().empty()) {
      const pimRegion &srcRegion = objSrc.getRegions().front();
      unsigned coreId = srcRegion.getCoreId();
      pimCore &core = device->getCore(coreId);
      auto locSrc = locateNthB32(srcRegion, isVLayout, 0);
      setB32(core, isVLayout, locSrc.first, locSrc.second,
             *reinterpret_cast<unsigned *>(&carry));
    }
  } else if (m_cmdType == PimCmdEnum::ROTATE_L || m_cmdType == PimCmdEnum::SHIFT_L) {
    unsigned carry = 0;
    for (unsigned i = objSrc.getRegions().size(); i > 0; --i) {
      const pimRegion &srcRegion = objSrc.getRegions()[i - 1];
      unsigned coreId = srcRegion.getCoreId();
      pimCore &core = device->getCore(coreId);

      // retrieve the values
      unsigned numElementsInRegion = getNumElementsInRegion(srcRegion, bitsPerElement);
      std::vector<unsigned> regionVector(numElementsInRegion);
      for (unsigned j = 0; j < numElementsInRegion; ++j) {
        auto locSrc = locateNthB32(srcRegion, isVLayout, j);
        regionVector[j] = getB32(core, isVLayout, locSrc.first, locSrc.second);
      }
      // Perform the rotation
      for (int j = numElementsInRegion - 1; j >= 0; --j) {
        unsigned temp = regionVector[j];
        regionVector[j] = carry;
        carry = temp;
      }
      for (unsigned j = 0; j < numElementsInRegion; ++j) {
        auto locSrc = locateNthB32(srcRegion, isVLayout, j);
        setB32(core, isVLayout, locSrc.first, locSrc.second, regionVector[j]);
      }
    }
    if (m_cmdType == PimCmdEnum::SHIFT_L) {
      carry = 0; // fill with zero
    }
    if (!objSrc.getRegions().empty()) {
      const pimRegion &srcRegion = objSrc.getRegions().back();
      unsigned coreId = srcRegion.getCoreId();
      pimCore &core = device->getCore(coreId);
      unsigned numElementsInRegion = getNumElementsInRegion(srcRegion, bitsPerElement);
      auto locSrc = locateNthB32(srcRegion, isVLayout, numElementsInRegion - 1);
      setB32(core, isVLayout, locSrc.first, locSrc.second,
             *reinterpret_cast<unsigned *>(&carry));
    }
  } else {
    assert(0);
  }

  // Update stats
  double msRuntime = pimSim::get()->getParamsPerf()->getMsRuntimeForRotate(m_cmdType, objSrc);
  pimSim::get()->getStatsMgr()->recordCmd(getName(dataType, isVLayout), msRuntime);
  return true;
}

//! @brief  PIM CMD: shift right/left by n bits
bool
pimCmdShift::execute(pimDevice* device)
{ 
  
  #if defined(DEBUG)
  std::printf("PIM-Info: %s (obj id %d - %u -> %d)\n", getName().c_str(), m_src, m_val, m_dest);
  #endif

  pimResMgr* resMgr = device->getResMgr();
  if (!isValidObjId(resMgr, m_src)) {
    return false;
  }
  
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_src);
  PimDataType srcDataType = objSrc.getDataType();

  const pimObjInfo& objDest = resMgr->getObjInfo(m_dest);
  PimDataType destDataType = objDest.getDataType();

  if (srcDataType != destDataType) {
    std::printf("PIM-Error: Cannot convert from %s to %s\n", objSrc.getDataTypeName().c_str(), objDest.getDataTypeName().c_str());
    return false;
  }

  // bit shift should be called only for int types
  if (srcDataType != PIM_INT32 && srcDataType != PIM_INT64) {
    std::printf("PIM-Error: Bit shift does not work for %u type.\n", srcDataType);
    return false;
  }

  bool isVLayout = objSrc.isVLayout();
  unsigned bitsPerElement = objSrc.getBitsPerElement();

  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& src1Region = objSrc.getRegions()[i];
    const pimRegion& destRegion = objDest.getRegions()[i];

    PimCoreId coreId = src1Region.getCoreId();
    pimCore& core = device->getCore(coreId);

    // perform the computation
    unsigned numElementsInRegion = getNumElementsInRegion(src1Region, bitsPerElement);
    for (unsigned j = 0; j < numElementsInRegion; ++j) {
      auto locSrc1 = locateNthB32(src1Region, isVLayout, j);
      auto locDest = locateNthB32(destRegion, isVLayout, j);

      if (bitsPerElement == 32) {
        auto operandBits = getB32(core, isVLayout, locSrc1.first, locSrc1.second);
        int operand = *reinterpret_cast<unsigned *>(&operandBits);
        int result = 0;
        switch (m_cmdType) {
        case PimCmdEnum::SHIFT_R: result = operand >> m_shiftAmount; break;
        case PimCmdEnum::SHIFT_L: result = operand << m_shiftAmount; break;
        default:
          std::printf("PIM-Error: Unexpected cmd type %d\n", m_cmdType);
          assert(0);
        }
        setB32(core, isVLayout, locDest.first, locDest.second,
               *reinterpret_cast<unsigned *>(&result));
      } else {
        assert(0); // todo: data type
      }
    }
  }

  // Update stats
  double msRuntime = pimSim::get()->getParamsPerf()->getMsRuntimeForRotate(m_cmdType, objSrc);
  pimSim::get()->getStatsMgr()->recordCmd(getName(srcDataType, isVLayout), msRuntime);
  return true;
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

  // Update stats
  pimSim::get()->getStatsMgr()->recordCmd(getName(), 0.0);
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

  // Update stats
  pimSim::get()->getStatsMgr()->recordCmd(getName(), 0.0);
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

  // Update stats
  pimSim::get()->getStatsMgr()->recordCmd(getName(), 0.0);
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

  // Update stats
  pimSim::get()->getStatsMgr()->recordCmd(getName(), 0.0);
  return true;
}

