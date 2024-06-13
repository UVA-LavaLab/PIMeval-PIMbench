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

//! @brief  Process all regions in MT used by derived classes
bool
pimCmd::computeAllRegions(unsigned numRegions)
{
  if (pimSim::get()->getNumThreads() > 1) { // MT
    std::vector<pimUtils::threadWorker*> workers;
    for (unsigned i = 0; i < numRegions; ++i) {
      workers.push_back(new regionWorker(this, i));
    }
    pimSim::get()->getThreadPool()->doWork(workers);
    for (unsigned i = 0; i < numRegions; ++i) {
      delete workers[i];
    }
  } else { // single thread
    for (unsigned i = 0; i < numRegions; ++i) {
      computeRegion(i);
    }
  }
  return true;
}


//! @brief  PIM Data Copy
bool
pimCmdCopy::execute()
{
  if (!sanityCheck()) {
    return false;
  }

  if (m_cmdType == PimCmdEnum::COPY_H2D) {
    const pimObjInfo &objDest = m_device->getResMgr()->getObjInfo(m_dest);
    unsigned numRegions = objDest.getRegions().size();
    computeAllRegions(numRegions);
  } else if (m_cmdType == PimCmdEnum::COPY_D2H) {
    const pimObjInfo &objSrc = m_device->getResMgr()->getObjInfo(m_src);
    unsigned numRegions = objSrc.getRegions().size();
    computeAllRegions(numRegions);
  } else if (m_cmdType == PimCmdEnum::COPY_D2D) {
    const pimObjInfo &objSrc = m_device->getResMgr()->getObjInfo(m_src);
    unsigned numRegions = objSrc.getRegions().size();
    computeAllRegions(numRegions);
  } else {
    assert(0);
  }

  updateStats();
  return true;
}

//! @brief  PIM Data Copy - sanity check
bool
pimCmdCopy::sanityCheck() const
{
  pimResMgr* resMgr = m_device->getResMgr();
  switch (m_cmdType) {
  case PimCmdEnum::COPY_H2D:
  {
    if (!m_ptr) {
      std::printf("PIM-Error: Invalid null pointer as copy source\n");
      return false;
    }
    if (!resMgr->isValidObjId(m_dest)) {
      std::printf("PIM-Error: Invalid PIM object ID %d as copy destination\n", m_dest);
      return false;
    }
    break;
  }
  case PimCmdEnum::COPY_D2H:
  {
    if (!resMgr->isValidObjId(m_src)) {
      std::printf("PIM-Error: Invalid PIM object ID %d as copy source\n", m_src);
      return false;
    }
    if (!m_ptr) {
      std::printf("PIM-Error: Invalid null pointer as copy destination\n");
      return false;
    }
    break;
  }
  case PimCmdEnum::COPY_D2D:
  {
    if (!resMgr->isValidObjId(m_src)) {
      std::printf("PIM-Error: Invalid PIM object ID %d as copy source\n", m_src);
      return false;
    }
    if (!resMgr->isValidObjId(m_dest)) {
      std::printf("PIM-Error: Invalid PIM object ID %d as copy destination\n", m_dest);
      return false;
    }
    const pimObjInfo &objSrc = resMgr->getObjInfo(m_src);
    const pimObjInfo &objDest = resMgr->getObjInfo(m_dest);
    if (!isAssociated(objSrc, objDest)) {
      std::printf("PIM-Error: PIM object IDs %d and %d are not associated for device-to-device copying\n", m_src, m_dest);
      return false;
    }
    break;
  }
  default:
    assert(0);
  }
  return true;
}

//! @brief  PIM Data Copy - compute region
bool
pimCmdCopy::computeRegion(unsigned index)
{
  // read in bits from src (host or PIM)
  std::vector<bool> bits;
  if (m_cmdType == PimCmdEnum::COPY_H2D) {
    const pimObjInfo &objDest = m_device->getResMgr()->getObjInfo(m_dest);
    const pimRegion& region = objDest.getRegions()[index];
    unsigned numAllocRows = region.getNumAllocRows();
    unsigned numAllocCols = region.getNumAllocCols();
    unsigned bitsPerElement = objDest.getBitsPerElement();
    unsigned numElementInCurRegion = numAllocRows * numAllocCols / bitsPerElement;
    unsigned byteOfst = index * objDest.getMaxElementsPerRegion() * bitsPerElement / 8;
    void* ptr = (void*)((char*)m_ptr + byteOfst);
    bits = pimUtils::readBitsFromHost(ptr, numElementInCurRegion, bitsPerElement);
  } else if (m_cmdType == PimCmdEnum::COPY_D2H || m_cmdType == PimCmdEnum::COPY_D2D) {
    const pimObjInfo &objSrc = m_device->getResMgr()->getObjInfo(m_src);
    const pimRegion& region = objSrc.getRegions()[index];
    unsigned rowIdx = region.getRowIdx();
    unsigned colIdx = region.getColIdx();
    unsigned numAllocRows = region.getNumAllocRows();
    unsigned numAllocCols = region.getNumAllocCols();
    PimCoreId coreId = region.getCoreId();
    pimCore& core = m_device->getCore(coreId);
    if (m_copyType == PIM_COPY_V) {
      for (unsigned c = 0; c < numAllocCols; ++c) {
        for (unsigned r = 0; r < numAllocRows; ++r) {
          unsigned row = rowIdx + r;
          unsigned col = colIdx + c;
          bool val = core.getBit(row, col);
          bits.push_back(val);
        }
      }
    } else if (m_copyType == PIM_COPY_H) {
      for (unsigned r = 0; r < numAllocRows; ++r) {
        for (unsigned c = 0; c < numAllocCols; ++c) {
          unsigned row = rowIdx + r;
          unsigned col = colIdx + c;
          bool val = core.getBit(row, col);
          bits.push_back(val);
        }
      }
    } else {
      assert(0);
    }
  } else {
    assert(0);
  }

  // write bits to dest (host or PIM)
  if (m_cmdType == PimCmdEnum::COPY_H2D || m_cmdType == PimCmdEnum::COPY_D2D) {
    const pimObjInfo &objDest = m_device->getResMgr()->getObjInfo(m_dest);
    const pimRegion& region = objDest.getRegions()[index];
    unsigned rowIdx = region.getRowIdx();
    unsigned colIdx = region.getColIdx();
    unsigned numAllocRows = region.getNumAllocRows();
    unsigned numAllocCols = region.getNumAllocCols();
    PimCoreId coreId = region.getCoreId();
    pimCore& core = m_device->getCore(coreId);
    if (m_copyType == PIM_COPY_V) {
      size_t bitIdx = 0;
      for (size_t i = 0; i < (size_t)numAllocRows * numAllocCols; ++i) {
        bool val = bits[bitIdx++];
        unsigned row = rowIdx + i % numAllocRows;
        unsigned col = colIdx + i / numAllocRows;
        core.setBit(row, col, val);
      }
    } else if (m_copyType == PIM_COPY_H) {
      size_t bitIdx = 0;
      for (size_t i = 0; i < (size_t)numAllocRows * numAllocCols; ++i) {
        bool val = bits[bitIdx++];
        unsigned row = rowIdx + i / numAllocCols;
        unsigned col = colIdx + i % numAllocCols;
        core.setBit(row, col, val);
      }
    } else {
      assert(0);
    }
  } else if (m_cmdType == PimCmdEnum::COPY_D2H) {
    const pimObjInfo &objSrc = m_device->getResMgr()->getObjInfo(m_src);
    unsigned bitsPerElement = objSrc.getBitsPerElement();
    unsigned byteOfst = index * objSrc.getMaxElementsPerRegion() * bitsPerElement / 8;
    void* ptr = (void*)((char*)m_ptr + byteOfst);
    pimUtils::writeBitsToHost(ptr, bits);
  } else {
    assert(0);
  }

  return true;
}

//! @brief  PIM Data Copy - update stats
bool
pimCmdCopy::updateStats() const
{
   if (m_cmdType == PimCmdEnum::COPY_H2D) {
    const pimObjInfo &objDest = m_device->getResMgr()->getObjInfo(m_dest);
    unsigned numElements = objDest.getNumElements();
    unsigned bitsPerElement = objDest.getBitsPerElement();
    pimSim::get()->getStatsMgr()->recordCopyMainToDevice((uint64_t)numElements * bitsPerElement);

    #if defined(DEBUG)
    std::printf("PIM-Info: Copied %u elements of %u bits from host to PIM obj %d\n",
                numElements, bitsPerElement, m_dest);
    #endif

  } else if (m_cmdType == PimCmdEnum::COPY_D2H) {
    const pimObjInfo &objSrc = m_device->getResMgr()->getObjInfo(m_src);
    unsigned numElements = objSrc.getNumElements();
    unsigned bitsPerElement = objSrc.getBitsPerElement();
    pimSim::get()->getStatsMgr()->recordCopyDeviceToMain((uint64_t)numElements * bitsPerElement);

    #if defined(DEBUG)
    std::printf("PIM-Info: Copied %u elements of %u bits from PIM obj %d to host\n",
                numElements, bitsPerElement, m_src);
    #endif

  } else if (m_cmdType == PimCmdEnum::COPY_D2D) {
    const pimObjInfo &objSrc = m_device->getResMgr()->getObjInfo(m_src);
    unsigned numElements = objSrc.getNumElements();
    unsigned bitsPerElement = objSrc.getBitsPerElement();
    pimSim::get()->getStatsMgr()->recordCopyDeviceToDevice((uint64_t)numElements * bitsPerElement);

    #if defined(DEBUG)
    std::printf("PIM-Info: Copied %u elements of %u bits from PIM obj %d to PIM obj %d\n",
                numElements, bitsPerElement, m_src, m_dest);
    #endif

  } else {
    assert(0);
  }
  return true;
}


//! @brief  PIM CMD: Functional 1-operand
bool
pimCmdFunc1::execute()
{
  #if defined(DEBUG)
  std::printf("PIM-Info: %s (obj id %d -> %d)\n", getName().c_str(), m_src, m_dest);
  #endif

  if (!sanityCheck()) {
    return false;
  }

  const pimObjInfo& objSrc = m_device->getResMgr()->getObjInfo(m_src);
  unsigned numRegions = objSrc.getRegions().size();
  computeAllRegions(numRegions);

  updateStats();
  return true;
}

//! @brief  PIM CMD: Functional 1-operand - sanity check
bool
pimCmdFunc1::sanityCheck() const
{
  pimResMgr* resMgr = m_device->getResMgr();
  if (!isValidObjId(resMgr, m_src) || !isValidObjId(resMgr, m_dest)) {
    return false;
  }
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_src);
  const pimObjInfo& objDest = resMgr->getObjInfo(m_dest);
  if (!isAssociated(objSrc, objDest)) {
    return false;
  }
  return true;
}

//! @brief  PIM CMD: Functional 1-operand - compute region
bool
pimCmdFunc1::computeRegion(unsigned index)
{
  const pimObjInfo& objSrc = m_device->getResMgr()->getObjInfo(m_src);
  const pimObjInfo& objDest = m_device->getResMgr()->getObjInfo(m_dest);
  PimDataType dataType = objSrc.getDataType();
  bool isVLayout = objSrc.isVLayout();
  unsigned bitsPerElement = objSrc.getBitsPerElement();

  const pimRegion& srcRegion = objSrc.getRegions()[index];
  const pimRegion& destRegion = objDest.getRegions()[index];

  PimCoreId coreId = srcRegion.getCoreId();
  pimCore& core = m_device->getCore(coreId);

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
  return true;
}

//! @brief  PIM CMD: Functional 1-operand - update stats
bool
pimCmdFunc1::updateStats() const
{
  const pimObjInfo& objSrc = m_device->getResMgr()->getObjInfo(m_src);
  PimDataType dataType = objSrc.getDataType();
  bool isVLayout = objSrc.isVLayout();

  double msRuntime = pimSim::get()->getParamsPerf()->getMsRuntimeForFunc1(m_cmdType, objSrc);
  pimSim::get()->getStatsMgr()->recordCmd(getName(dataType, isVLayout), msRuntime);
  return true;
}


//! @brief  PIM CMD: Functional 2-operand
bool
pimCmdFunc2::execute()
{
  #if defined(DEBUG)
  std::printf("PIM-Info: %s (obj id %d - %d -> %d)\n", getName().c_str(), m_src1, m_src2, m_dest);
  #endif

  if (!sanityCheck()) {
    return false;
  }

  const pimObjInfo& objSrc1 = m_device->getResMgr()->getObjInfo(m_src1);
  unsigned numRegions = objSrc1.getRegions().size();
  computeAllRegions(numRegions);

  updateStats();
  return true;
}

//! @brief  PIM CMD: Functional 2-operand - sanity check
bool
pimCmdFunc2::sanityCheck() const
{
  pimResMgr* resMgr = m_device->getResMgr();
  if (!isValidObjId(resMgr, m_src1) || !isValidObjId(resMgr, m_src2) || !isValidObjId(resMgr, m_dest)) {
    return false;
  }
  const pimObjInfo& objSrc1 = resMgr->getObjInfo(m_src1);
  const pimObjInfo& objSrc2 = resMgr->getObjInfo(m_src2);
  const pimObjInfo& objDest = resMgr->getObjInfo(m_dest);
  if (!isAssociated(objSrc1, objSrc2) || !isAssociated(objSrc1, objDest)) {
    return false;
  }

  if (objSrc1.getDataType() != objSrc2.getDataType()) {
    std::printf("PIM-Error: Type mismatch between object %d and %d\n", m_src1, m_src2);
    return false;
  }

  if (objSrc1.getDataType() != objDest.getDataType()) {
    std::printf("PIM-Error: Cannot convert from %s to %s\n", objSrc1.getDataTypeName().c_str(), objDest.getDataTypeName().c_str());
    return false;
  }
  return true;
}

//! @brief  PIM CMD: Functional 2-operand - compute region
bool
pimCmdFunc2::computeRegion(unsigned index)
{
  const pimObjInfo& objSrc1 = m_device->getResMgr()->getObjInfo(m_src1);
  const pimObjInfo& objSrc2 = m_device->getResMgr()->getObjInfo(m_src2);
  const pimObjInfo& objDest = m_device->getResMgr()->getObjInfo(m_dest);

  PimDataType dataType = objSrc1.getDataType();
  bool isVLayout = objSrc1.isVLayout();
  unsigned bitsPerElement = objSrc1.getBitsPerElement();

  const pimRegion& src1Region = objSrc1.getRegions()[index];
  const pimRegion& src2Region = objSrc2.getRegions()[index];
  const pimRegion& destRegion = objDest.getRegions()[index];

  PimCoreId coreId = src1Region.getCoreId();
  pimCore& core = m_device->getCore(coreId);

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
  return true;
}

//! @brief  PIM CMD: Functional 2-operand - update stats
bool
pimCmdFunc2::updateStats() const
{
  const pimObjInfo& objSrc1 = m_device->getResMgr()->getObjInfo(m_src1);
  PimDataType dataType = objSrc1.getDataType();
  bool isVLayout = objSrc1.isVLayout();

  double msRuntime = pimSim::get()->getParamsPerf()->getMsRuntimeForFunc2(m_cmdType, objSrc1);
  pimSim::get()->getStatsMgr()->recordCmd(getName(dataType, isVLayout), msRuntime);
  return true;
}


//! @brief  PIM CMD: redsum non-ranged/ranged
bool
pimCmdRedSum::execute()
{
  #if defined(DEBUG)
  std::printf("PIM-Info: %s (obj id %d)\n", getName().c_str(), m_src);
  #endif

  if (!sanityCheck()) {
    return false;
  }

  const pimObjInfo& objSrc = m_device->getResMgr()->getObjInfo(m_src);
  unsigned numRegions = objSrc.getRegions().size();

  // prepare per-region storage
  m_regionSum.resize(numRegions, 0);

  computeAllRegions(numRegions);

  // reduction
  for (unsigned i = 0; i < numRegions; ++i) {
    *m_result += m_regionSum[i];
  }

  updateStats();
  return true;
}

//! @brief  PIM CMD: redsum non-ranged/ranged - sanity check
bool
pimCmdRedSum::sanityCheck() const
{
  pimResMgr* resMgr = m_device->getResMgr();
  if (!isValidObjId(resMgr, m_src) || !m_result) {
    return false;
  }
  return true;
}

//! @brief  PIM CMD: redsum non-ranged/ranged - compute region
bool
pimCmdRedSum::computeRegion(unsigned index)
{
  const pimObjInfo& objSrc = m_device->getResMgr()->getObjInfo(m_src);
  bool isVLayout = objSrc.isVLayout();
  unsigned bitsPerElement = objSrc.getBitsPerElement();

  const pimRegion& srcRegion = objSrc.getRegions()[index];
  PimCoreId coreId = srcRegion.getCoreId();
  pimCore& core = m_device->getCore(coreId);

  unsigned numElementsInRegion = getNumElementsInRegion(srcRegion, bitsPerElement);
  unsigned currIdx = numElementsInRegion * index;
  for (unsigned j = 0; j < numElementsInRegion && currIdx < m_idxEnd; ++j) {
    if (currIdx >= m_idxBegin) {
      auto locSrc = locateNthB32(srcRegion, isVLayout, j);
      auto operandBits = getB32(core, isVLayout, locSrc.first, locSrc.second);
      int operand = *reinterpret_cast<int*>(&operandBits);
      m_regionSum[index] += operand;
    }
    currIdx += 1;
  }
  return true;
}

//! @brief  PIM CMD: redsum non-ranged/ranged - update stats
bool
pimCmdRedSum::updateStats() const
{
  const pimObjInfo& objSrc = m_device->getResMgr()->getObjInfo(m_src);
  PimDataType dataType = objSrc.getDataType();
  bool isVLayout = objSrc.isVLayout();

  double msRuntime = pimSim::get()->getParamsPerf()->getMsRuntimeForRedSum(m_cmdType, objSrc);
  pimSim::get()->getStatsMgr()->recordCmd(getName(dataType, isVLayout), msRuntime);
  return true;
}

//! @brief  PIM CMD: broadcast a value to all elements
bool
pimCmdBroadcast::execute()
{
  #if defined(DEBUG)
  std::printf("PIM-Info: %s (obj id %d value %u)\n", getName().c_str(), m_dest, m_val);
  #endif

  if (!sanityCheck()) {
    return false;
  }

  const pimObjInfo& objDest = m_device->getResMgr()->getObjInfo(m_dest);
  unsigned numRegions = objDest.getRegions().size();
  computeAllRegions(numRegions);

  updateStats();
  return true;
}

//! @brief  PIM CMD: broadcast a value to all elements - sanity check
bool
pimCmdBroadcast::sanityCheck() const
{
  pimResMgr* resMgr = m_device->getResMgr();
  if (!isValidObjId(resMgr, m_dest)) {
    return false;
  }
  return true;
}

//! @brief  PIM CMD: broadcast a value to all elements - compute region
bool
pimCmdBroadcast::computeRegion(unsigned index)
{
  const pimObjInfo& objDest = m_device->getResMgr()->getObjInfo(m_dest);
  bool isVLayout = objDest.isVLayout();

  unsigned bitsPerElement = objDest.getBitsPerElement();
  assert(bitsPerElement == 32); // todo: support other types

  const pimRegion& destRegion = objDest.getRegions()[index];
  PimCoreId coreId = destRegion.getCoreId();
  pimCore &core = m_device->getCore(coreId);

  unsigned numElementsInRegion = getNumElementsInRegion(destRegion, bitsPerElement);

  unsigned val = *reinterpret_cast<unsigned *>(&m_val);
  for (unsigned j = 0; j < numElementsInRegion; ++j) {
    auto locDest = locateNthB32(destRegion, isVLayout, j);
    setB32(core, isVLayout, locDest.first, locDest.second, val);
  }
  return true;
}

//! @brief  PIM CMD: broadcast a value to all elements - update stats
bool
pimCmdBroadcast::updateStats() const
{
  const pimObjInfo& objDest = m_device->getResMgr()->getObjInfo(m_dest);
  PimDataType dataType = objDest.getDataType();
  bool isVLayout = objDest.isVLayout();

  double msRuntime = pimSim::get()->getParamsPerf()->getMsRuntimeForBroadcast(m_cmdType, objDest);
  pimSim::get()->getStatsMgr()->recordCmd(getName(dataType, isVLayout), msRuntime);
  return true;
}


//! @brief  PIM CMD: rotate right/left
bool
pimCmdRotate::execute()
{
  #if defined(DEBUG)
  std::printf("PIM-Info: %s (obj id %d)\n", getName().c_str(), m_src);
  #endif

  if (!sanityCheck()) {
    return false;
  }

  const pimObjInfo& objSrc = m_device->getResMgr()->getObjInfo(m_src);
  unsigned numRegions = objSrc.getRegions().size();
  m_regionBoundary.resize(numRegions, 0);

  computeAllRegions(numRegions);

  // handle region boundaries
  bool isVLayout = objSrc.isVLayout();
  unsigned bitsPerElement = objSrc.getBitsPerElement();
  if (m_cmdType == PimCmdEnum::ROTATE_R || m_cmdType == PimCmdEnum::SHIFT_R) {
    for (unsigned i = 0; i < numRegions; ++i) {
      const pimRegion &srcRegion = objSrc.getRegions()[i];
      unsigned coreId = srcRegion.getCoreId();
      pimCore &core = m_device->getCore(coreId);
      auto locSrc = locateNthB32(srcRegion, isVLayout, 0);
      unsigned val = 0;
      if (i == 0 && m_cmdType == PimCmdEnum::ROTATE_R) {
        val = m_regionBoundary[numRegions - 1];
      } else if (i > 0) {
        val = m_regionBoundary[i - 1];
      }
      setB32(core, isVLayout, locSrc.first, locSrc.second, val);
    }
  } else if (m_cmdType == PimCmdEnum::ROTATE_L || m_cmdType == PimCmdEnum::SHIFT_L) {
    for (unsigned i = 0; i < numRegions; ++i) {
      const pimRegion &srcRegion = objSrc.getRegions()[i];
      unsigned coreId = srcRegion.getCoreId();
      pimCore &core = m_device->getCore(coreId);
      unsigned numElementsInRegion = getNumElementsInRegion(srcRegion, bitsPerElement);
      auto locSrc = locateNthB32(srcRegion, isVLayout, numElementsInRegion - 1);
      unsigned val = 0;
      if (i == numRegions - 1 && m_cmdType == PimCmdEnum::ROTATE_R) {
        val = m_regionBoundary[0];
      } else if (i < numRegions - 1) {
        val = m_regionBoundary[i + 1];
      }
      setB32(core, isVLayout, locSrc.first, locSrc.second, val);
    }
  } else {
    assert(0);
  }

  updateStats();
  return true;
}

//! @brief  PIM CMD: rotate right/left - sanity check
bool
pimCmdRotate::sanityCheck() const
{
  pimResMgr* resMgr = m_device->getResMgr();
  if (!isValidObjId(resMgr, m_src)) {
    return false;
  }
  return true;
}

//! @brief  PIM CMD: rotate right/left - compute region
bool
pimCmdRotate::computeRegion(unsigned index)
{
  const pimObjInfo& objSrc = m_device->getResMgr()->getObjInfo(m_src);
  bool isVLayout = objSrc.isVLayout();
  unsigned bitsPerElement = objSrc.getBitsPerElement();

  const pimRegion& srcRegion = objSrc.getRegions()[index];
  unsigned coreId = srcRegion.getCoreId();
  pimCore &core = m_device->getCore(coreId);

  // read out values
  unsigned numElementsInRegion = getNumElementsInRegion(srcRegion, bitsPerElement);
  std::vector<unsigned> regionVector(numElementsInRegion);
  for (unsigned j = 0; j < numElementsInRegion; ++j) {
    auto locSrc = locateNthB32(srcRegion, isVLayout, j);
    regionVector[j] = getB32(core, isVLayout, locSrc.first, locSrc.second);
  }

  // perform rotation
  if (m_cmdType == PimCmdEnum::ROTATE_R || m_cmdType == PimCmdEnum::SHIFT_R) {
    m_regionBoundary[index] = regionVector[numElementsInRegion - 1];
    unsigned carry = 0;
    for (unsigned j = 0; j < numElementsInRegion; ++j) {
      int temp = regionVector[j];
      regionVector[j] = carry;
      carry = temp;
    }
  } else if (m_cmdType == PimCmdEnum::ROTATE_L || m_cmdType == PimCmdEnum::SHIFT_L) {
    m_regionBoundary[index] = regionVector[0];
    unsigned carry = 0;
    for (int j = numElementsInRegion - 1; j >= 0; --j) {
      unsigned temp = regionVector[j];
      regionVector[j] = carry;
      carry = temp;
    }
  } else {
    assert(0);
  }

  // write back values
  for (unsigned j = 0; j < numElementsInRegion; ++j) {
    auto locSrc = locateNthB32(srcRegion, isVLayout, j);
    setB32(core, isVLayout, locSrc.first, locSrc.second, regionVector[j]);
  }
  return true;
}

//! @brief  PIM CMD: rotate right/left - update stats
bool
pimCmdRotate::updateStats() const
{
  const pimObjInfo& objSrc = m_device->getResMgr()->getObjInfo(m_src);
  PimDataType dataType = objSrc.getDataType();
  bool isVLayout = objSrc.isVLayout();

  double msRuntime = pimSim::get()->getParamsPerf()->getMsRuntimeForRotate(m_cmdType, objSrc);
  pimSim::get()->getStatsMgr()->recordCmd(getName(dataType, isVLayout), msRuntime);
  return true;
}


//! @brief  Pim CMD: BitSIMD-V: Read a row to SA
bool
pimCmdReadRowToSa::execute()
{
  #if defined(DEBUG)
  std::printf("PIM-Info: BitSIMD-V ReadRowToSa (obj id %d ofst %u)\n", m_objId, m_ofst);
  #endif

  pimResMgr* resMgr = m_device->getResMgr();
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_objId);
  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];
    if (m_ofst >= srcRegion.getNumAllocRows()) {
      std::printf("PIM-Error: Row offset %u out of range [0, %u)\n", m_ofst, srcRegion.getNumAllocRows());
      return false;
    }
    PimCoreId coreId = srcRegion.getCoreId();
    m_device->getCore(coreId).readRow(srcRegion.getRowIdx() + m_ofst);
  }

  // Update stats
  pimSim::get()->getStatsMgr()->recordCmd(getName(), 0.0);
  return true;
}

//! @brief  Pim CMD: BitSIMD-V: Write SA to a row
bool
pimCmdWriteSaToRow::execute()
{
  #if defined(DEBUG)
  std::printf("PIM-Info: BitSIMD-V WriteSaToRow (obj id %d ofst %u)\n", m_objId, m_ofst);
  #endif

  pimResMgr* resMgr = m_device->getResMgr();
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_objId);
  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];
    if (m_ofst >= srcRegion.getNumAllocRows()) {
      std::printf("PIM-Error: Row offset %u out of range [0, %u)\n", m_ofst, srcRegion.getNumAllocRows());
      return false;
    }
    PimCoreId coreId = srcRegion.getCoreId();
    m_device->getCore(coreId).writeRow(srcRegion.getRowIdx() + m_ofst);
  }

  // Update stats
  pimSim::get()->getStatsMgr()->recordCmd(getName(), 0.0);
  return true;
}

//! @brief  Pim CMD: BitSIMD-V: Row reg operations
bool
pimCmdRRegOp::execute()
{
  #if defined(DEBUG)
  std::printf("PIM-Info: BitSIMD-V %s (obj-id %d dest-reg %d src-reg %d %d %d val %d)\n",
              getName().c_str(), m_objId, m_dest, m_src1, m_src2, m_src3, m_val);
  #endif

  pimResMgr* resMgr = m_device->getResMgr();
  const pimObjInfo& refObj = resMgr->getObjInfo(m_objId);
  for (unsigned i = 0; i < refObj.getRegions().size(); ++i) {
    const pimRegion& refRegion = refObj.getRegions()[i];
    PimCoreId coreId = refRegion.getCoreId();
    for (unsigned j = 0; j < m_device->getNumCols(); j++) {
      switch (m_cmdType) {
      case PimCmdEnum::RREG_MOV:
      {
        m_device->getCore(coreId).getRowReg(m_dest)[j] = m_device->getCore(coreId).getRowReg(m_src1)[j];
        break;
      }
      case PimCmdEnum::RREG_SET:
      {
        m_device->getCore(coreId).getRowReg(m_dest)[j] = m_val;
        break;
      }
      case PimCmdEnum::RREG_NOT:
      {
        bool src = m_device->getCore(coreId).getRowReg(m_src1)[j];
        m_device->getCore(coreId).getRowReg(m_dest)[j] = !src;
        break;
      }
      case PimCmdEnum::RREG_AND:
      {
        bool src1 = m_device->getCore(coreId).getRowReg(m_src1)[j];
        bool src2 = m_device->getCore(coreId).getRowReg(m_src2)[j];
        m_device->getCore(coreId).getRowReg(m_dest)[j] = (src1 & src2);
        break;
      }
      case PimCmdEnum::RREG_OR:
      {
        bool src1 = m_device->getCore(coreId).getRowReg(m_src1)[j];
        bool src2 = m_device->getCore(coreId).getRowReg(m_src2)[j];
        m_device->getCore(coreId).getRowReg(m_dest)[j] = src1 | src2;
        break;
      }
      case PimCmdEnum::RREG_NAND:
      {
        bool src1 = m_device->getCore(coreId).getRowReg(m_src1)[j];
        bool src2 = m_device->getCore(coreId).getRowReg(m_src2)[j];
        m_device->getCore(coreId).getRowReg(m_dest)[j] = !(src1 & src2);
        break;
      }
      case PimCmdEnum::RREG_NOR:
      {
        bool src1 = m_device->getCore(coreId).getRowReg(m_src1)[j];
        bool src2 = m_device->getCore(coreId).getRowReg(m_src2)[j];
        m_device->getCore(coreId).getRowReg(m_dest)[j] = !(src1 | src2);
        break;
      }
      case PimCmdEnum::RREG_XOR:
      {
        bool src1 = m_device->getCore(coreId).getRowReg(m_src1)[j];
        bool src2 = m_device->getCore(coreId).getRowReg(m_src2)[j];
        m_device->getCore(coreId).getRowReg(m_dest)[j] = src1 ^ src2;
        break;
      }
      case PimCmdEnum::RREG_XNOR:
      {
        bool src1 = m_device->getCore(coreId).getRowReg(m_src1)[j];
        bool src2 = m_device->getCore(coreId).getRowReg(m_src2)[j];
        m_device->getCore(coreId).getRowReg(m_dest)[j] = !(src1 ^ src2);
        break;
      }
      case PimCmdEnum::RREG_MAJ:
      {
        bool src1 = m_device->getCore(coreId).getRowReg(m_src1)[j];
        bool src2 = m_device->getCore(coreId).getRowReg(m_src2)[j];
        bool src3 = m_device->getCore(coreId).getRowReg(m_src3)[j];
        m_device->getCore(coreId).getRowReg(m_dest)[j] =
            ((src1 & src2) || (src1 & src3) || (src2 & src3));
        break;
      }
      case PimCmdEnum::RREG_SEL:
      {
        bool cond = m_device->getCore(coreId).getRowReg(m_src1)[j];
        bool src2 = m_device->getCore(coreId).getRowReg(m_src2)[j];
        bool src3 = m_device->getCore(coreId).getRowReg(m_src3)[j];
        m_device->getCore(coreId).getRowReg(m_dest)[j] = (cond ? src2 : src3);
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
pimCmdRRegRotate::execute()
{
  #if defined(DEBUG)
  std::printf("PIM-Info: BitSIMD-V %s (obj-id %d src-reg %d)\n", getName().c_str(), m_objId, m_dest);
  #endif

  pimResMgr* resMgr = m_device->getResMgr();
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_objId);
  if (m_cmdType == PimCmdEnum::RREG_ROTATE_R) {  // Right Rotate
    bool prevVal = 0;
    for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
      const pimRegion &srcRegion = objSrc.getRegions()[i];
      PimCoreId coreId = srcRegion.getCoreId();
      for (unsigned j = 0; j < srcRegion.getNumAllocCols(); ++j) {
        unsigned colIdx = srcRegion.getColIdx() + j;
        bool tmp = m_device->getCore(coreId).getRowReg(m_dest)[colIdx];
        m_device->getCore(coreId).getRowReg(m_dest)[colIdx] = prevVal;
        prevVal = tmp;
      }
    }
    // write the last val to the first place
    const pimRegion &firstRegion = objSrc.getRegions().front();
    PimCoreId firstCoreId = firstRegion.getCoreId();
    unsigned firstColIdx = firstRegion.getColIdx();
    m_device->getCore(firstCoreId).getRowReg(m_dest)[firstColIdx] = prevVal;
  } else if (m_cmdType == PimCmdEnum::RREG_ROTATE_L) {  // Left Rotate
    bool prevVal = 0;
    for (unsigned i = objSrc.getRegions().size(); i > 0; --i) {
      const pimRegion &srcRegion = objSrc.getRegions()[i - 1];
      PimCoreId coreId = srcRegion.getCoreId();
      for (unsigned j = srcRegion.getNumAllocCols(); j > 0; --j) {
        unsigned colIdx = srcRegion.getColIdx() + j - 1;
        bool tmp = m_device->getCore(coreId).getRowReg(m_dest)[colIdx];
        m_device->getCore(coreId).getRowReg(m_dest)[colIdx] = prevVal;
        prevVal = tmp;
      }
    }
    // write the first val to the last place
    const pimRegion &lastRegion = objSrc.getRegions().back();
    PimCoreId lastCoreId = lastRegion.getCoreId();
    unsigned lastColIdx = lastRegion.getColIdx() + lastRegion.getNumAllocCols() - 1;
    m_device->getCore(lastCoreId).getRowReg(m_dest)[lastColIdx] = prevVal;
  }

  // Update stats
  pimSim::get()->getStatsMgr()->recordCmd(getName(), 0.0);
  return true;
}

