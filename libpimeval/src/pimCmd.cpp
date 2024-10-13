// File: pimCmd.cpp
// PIMeval Simulator - PIM Commands
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimCmd.h"          // for pimCmd
#include "pimSim.h"          // for pimSim
#include "pimDevice.h"       // for pimDevice
#include "pimCore.h"         // for pimCore
#include "pimResMgr.h"       // for pimResMgr
#include "libpimeval.h"      // for PimObjId
#include <cstdio>            // for printf
#include <unordered_map>     // for unordered_map
#include <unordered_set>     // for unordered_set


//! @brief  Get PIM command name from command type enum
std::string
pimCmd::getName(PimCmdEnum cmdType, const std::string& suffix)
{
  static const std::unordered_map<PimCmdEnum, std::string> cmdNames = {
    { PimCmdEnum::NOOP, "noop" },
    { PimCmdEnum::COPY_H2D, "copy_h2d" },
    { PimCmdEnum::COPY_D2H, "copy_d2h" },
    { PimCmdEnum::COPY_D2D, "copy_d2d" },
    { PimCmdEnum::ABS, "abs" },
    { PimCmdEnum::POPCOUNT, "popcount" },
    { PimCmdEnum::SHIFT_BITS_R, "shift_bits_r" },
    { PimCmdEnum::SHIFT_BITS_L, "shift_bits_l" },
    { PimCmdEnum::BROADCAST, "broadcast" },
    { PimCmdEnum::ADD, "add" },
    { PimCmdEnum::SUB, "sub" },
    { PimCmdEnum::MUL, "mul" },
    { PimCmdEnum::SCALED_ADD, "scaled_add" },
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
    { PimCmdEnum::ADD_SCALAR, "add_scalar" },
    { PimCmdEnum::SUB_SCALAR, "sub_scalar" },
    { PimCmdEnum::MUL_SCALAR, "mul_scalar" },
    { PimCmdEnum::DIV_SCALAR, "div_scalar" },
    { PimCmdEnum::AND_SCALAR, "and_scalar" },
    { PimCmdEnum::OR_SCALAR, "or_scalar" },
    { PimCmdEnum::XOR_SCALAR, "xor_scalar" },
    { PimCmdEnum::XNOR_SCALAR, "xnor_scalar" },
    { PimCmdEnum::GT_SCALAR, "gt_scalar" },
    { PimCmdEnum::LT_SCALAR, "lt_scalar" },
    { PimCmdEnum::EQ_SCALAR, "eq_scalar" },
    { PimCmdEnum::MIN_SCALAR, "min_scalar" },
    { PimCmdEnum::MAX_SCALAR, "max_scalar" },
    { PimCmdEnum::REDSUM, "redsum" },
    { PimCmdEnum::REDSUM_RANGE, "redsum_range" },
    { PimCmdEnum::ROTATE_ELEM_R, "rotate_elem_r" },
    { PimCmdEnum::ROTATE_ELEM_L, "rotate_elem_l" },
    { PimCmdEnum::SHIFT_ELEM_R, "shift_elem_r" },
    { PimCmdEnum::SHIFT_ELEM_L, "shift_elem_l" },
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
    { PimCmdEnum::ROW_AP, "row_ap" },
    { PimCmdEnum::ROW_AAP, "row_aap" },
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
  if (obj1.getAssocObjId() != obj2.getAssocObjId()) {
    std::printf("PIM-Error: Object id %d and %d are not associated\n", obj1.getObjId(), obj2.getObjId());
    return false;
  }
  return true;
}

//! @brief  Check if two objects have compatible type.
bool
pimCmd::isCompatibleType(const pimObjInfo& obj1, const pimObjInfo& obj2) const
{
  // TODO: Type conversion eg. 32-bit and 64-bit should be compatible
  if (obj1.getDataType() != obj2.getDataType()) {
    std::printf("PIM-Error: Type mismatch between object %d and %d\n", obj1.getObjId(), obj2.getObjId());
    return false;
  }
  return true;
}

//! @brief  Check if src type can be converted to dest type.
bool
pimCmd::isConvertibleType(const pimObjInfo& src, const pimObjInfo& dest) const
{
  // TODO: Type conversion
  if (src.getDataType() != dest.getDataType()) {
    std::printf("PIM-Error: Cannot convert from %s to %s\n", src.getDataTypeName().c_str(), dest.getDataTypeName().c_str());
    return false;
  }
  return true;
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

  // for non-functional simulation, sync src data from simulated memory
  if (pimSim::get()->getDeviceType() != PIM_FUNCTIONAL) {
    if (m_cmdType == PimCmdEnum::COPY_D2H || m_cmdType == PimCmdEnum::COPY_D2D) {
      pimObjInfo &objSrc = m_device->getResMgr()->getObjInfo(m_src);
      objSrc.syncFromSimulatedMem();
    }
  }

  if (m_cmdType == PimCmdEnum::COPY_H2D) {
    pimObjInfo &objDest = m_device->getResMgr()->getObjInfo(m_dest);
    objDest.copyFromHost(m_ptr, m_idxBegin, m_idxEnd);
  } else if (m_cmdType == PimCmdEnum::COPY_D2H) {
    const pimObjInfo &objSrc = m_device->getResMgr()->getObjInfo(m_src);
    objSrc.copyToHost(m_ptr, m_idxBegin, m_idxEnd);
  } else if (m_cmdType == PimCmdEnum::COPY_D2D) {
    const pimObjInfo &objSrc = m_device->getResMgr()->getObjInfo(m_src);
    pimObjInfo &objDest = m_device->getResMgr()->getObjInfo(m_dest);
    objSrc.copyToObj(objDest, m_idxBegin, m_idxEnd);
  } else {
    assert(0);
  }

  // for non-functional simulation, sync dest data to simulated memory
  if (pimSim::get()->getDeviceType() != PIM_FUNCTIONAL) {
    if (m_cmdType == PimCmdEnum::COPY_H2D || m_cmdType == PimCmdEnum::COPY_D2D) {
      const pimObjInfo &objDest = m_device->getResMgr()->getObjInfo(m_dest);
      objDest.syncToSimulatedMem();
    }
  }

  updateStats();
  return true;
}

//! @brief  PIM Data Copy - sanity check
bool
pimCmdCopy::sanityCheck() const
{
  pimResMgr* resMgr = m_device->getResMgr();
  uint64_t numElements = 0;
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
    const pimObjInfo &objDest = m_device->getResMgr()->getObjInfo(m_dest);
    numElements = objDest.getNumElements();
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
    const pimObjInfo &objSrc = m_device->getResMgr()->getObjInfo(m_src);
    numElements = objSrc.getNumElements();
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
    numElements = objSrc.getNumElements();
    break;
  }
  default:
    assert(0);
  }
  if (!m_copyFullRange) {
    if (m_idxBegin > numElements) {
      std::printf("PIM-Error: The beggining of the copy range for PIM object ID %d is greater than the number of elements\n", m_dest);
      return false;
    }
    if (m_idxEnd > numElements) {
      std::printf("PIM-Error: The end of the copy range for PIM object ID %d is greater than the number of elements\n", m_dest);
      return false;
    }
    if (m_idxEnd < m_idxBegin) {
      std::printf("PIM-Error: The end of the copy range for PIM object ID %d is less than its beggining\n", m_dest);
      return false;
    }
  }

  return true;
}

//! @brief  PIM Data Copy - update stats
bool
pimCmdCopy::updateStats() const
{
   if (m_cmdType == PimCmdEnum::COPY_H2D) {
    const pimObjInfo &objDest = m_device->getResMgr()->getObjInfo(m_dest);
    uint64_t numElements = objDest.getNumElements();
    if (!m_copyFullRange) {
      numElements = m_idxEnd - m_idxBegin;
    }
    unsigned bitsPerElement = objDest.getBitsPerElement();
    pimeval::perfEnergy mPerfEnergy = pimSim::get()->getPerfEnergyModel()->getPerfEnergyForBytesTransfer(m_cmdType, numElements * bitsPerElement / 8);
    pimSim::get()->getStatsMgr()->recordCopyMainToDevice(numElements * bitsPerElement, mPerfEnergy);

    #if defined(DEBUG)
    std::printf("PIM-Info: Copied %llu elements of %u bits from host to PIM obj %d\n",
                numElements, bitsPerElement, m_dest);
    #endif

  } else if (m_cmdType == PimCmdEnum::COPY_D2H) {
    const pimObjInfo &objSrc = m_device->getResMgr()->getObjInfo(m_src);
    uint64_t numElements = objSrc.getNumElements();
    if (!m_copyFullRange) {
      numElements = m_idxEnd - m_idxBegin;
    }
    unsigned bitsPerElement = objSrc.getBitsPerElement();
    pimeval::perfEnergy mPerfEnergy = pimSim::get()->getPerfEnergyModel()->getPerfEnergyForBytesTransfer(m_cmdType, numElements * bitsPerElement / 8);
    pimSim::get()->getStatsMgr()->recordCopyDeviceToMain(numElements * bitsPerElement, mPerfEnergy);

    #if defined(DEBUG)
    std::printf("PIM-Info: Copied %llu elements of %u bits from PIM obj %d to host\n",
                numElements, bitsPerElement, m_src);
    #endif

  } else if (m_cmdType == PimCmdEnum::COPY_D2D) {
    const pimObjInfo &objSrc = m_device->getResMgr()->getObjInfo(m_src);
    uint64_t numElements = objSrc.getNumElements();
    if (!m_copyFullRange) {
      numElements = m_idxEnd - m_idxBegin;
    }
    unsigned bitsPerElement = objSrc.getBitsPerElement();
    pimeval::perfEnergy mPerfEnergy = pimSim::get()->getPerfEnergyModel()->getPerfEnergyForBytesTransfer(m_cmdType, numElements * bitsPerElement / 8);
    pimSim::get()->getStatsMgr()->recordCopyDeviceToDevice(numElements * bitsPerElement, mPerfEnergy);

    #if defined(DEBUG)
    std::printf("PIM-Info: Copied %llu elements of %u bits from PIM obj %d to PIM obj %d\n",
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

  if (pimSim::get()->getDeviceType() != PIM_FUNCTIONAL) {
    pimObjInfo &objSrc = m_device->getResMgr()->getObjInfo(m_src);
    objSrc.syncFromSimulatedMem();
  }

  const pimObjInfo& objSrc = m_device->getResMgr()->getObjInfo(m_src);
  unsigned numRegions = objSrc.getRegions().size();
  computeAllRegions(numRegions);

  if (pimSim::get()->getDeviceType() != PIM_FUNCTIONAL) {
    const pimObjInfo &objDest = m_device->getResMgr()->getObjInfo(m_dest);
    objDest.syncToSimulatedMem();
  }

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
  if (!isAssociated(objSrc, objDest) || !isConvertibleType(objSrc, objDest)) {
    return false;
  }
  return true;
}

//! @brief  PIM CMD: Functional 1-operand - compute region
bool
pimCmdFunc1::computeRegion(unsigned index)
{
  const pimObjInfo& objSrc = m_device->getResMgr()->getObjInfo(m_src);
  pimObjInfo& objDest = m_device->getResMgr()->getObjInfo(m_dest);

  PimDataType dataType = objSrc.getDataType();
  unsigned bitsPerElementSrc = objSrc.getBitsPerElement();
  const pimRegion& srcRegion = objSrc.getRegions()[index];

  // perform the computation
  uint64_t elemIdxBegin = srcRegion.getElemIdxBegin();
  unsigned numElementsInRegion = srcRegion.getNumElemInRegion();
  for (unsigned j = 0; j < numElementsInRegion; ++j) {
    uint64_t elemIdx = elemIdxBegin + j;
    if (dataType == PIM_INT8 || dataType == PIM_INT16 || dataType == PIM_INT32 || dataType == PIM_INT64 || dataType == PIM_UINT8 || dataType == PIM_UINT16 || dataType == PIM_UINT32 || dataType == PIM_UINT64 || dataType == PIM_FP32) {
      bool isSigned = (dataType == PIM_INT8 || dataType == PIM_INT16 || dataType == PIM_INT32 || dataType == PIM_INT64);
      if (isSigned) {
        int64_t signedOperand = objSrc.getElementBits(elemIdx);
        int64_t result = 0;
        if(!computeResult(signedOperand, m_cmdType, (int64_t)m_scalarValue, result, bitsPerElementSrc)) return false;
        objDest.setElement(elemIdx, result);
      } else if (dataType != PIM_FP32) { // signed int.
        uint64_t unsignedOperand = objSrc.getElementBits(elemIdx);
        uint64_t result = 0;
        if(!computeResult(unsignedOperand, m_cmdType, m_scalarValue, result, bitsPerElementSrc)) return false;
        objDest.setElement(elemIdx, result);
      } else if (dataType == PIM_FP32){
        uint64_t bits = objSrc.getElementBits(elemIdx);
        float floatOperand = pimUtils::castBitsToType<float>(bits);
        float result = 0.0;
        if(!computeResultFP(floatOperand, m_cmdType, pimUtils::castBitsToType<float>(m_scalarValue), result, bitsPerElementSrc)) return false;
        objDest.setElement(elemIdx, result);
      } else {
      assert(0); // todo: data type
      }
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

  
  pimeval::perfEnergy mPerfEnergy = pimSim::get()->getPerfEnergyModel()->getPerfEnergyForFunc1(m_cmdType, objSrc);
  pimSim::get()->getStatsMgr()->recordCmd(getName(dataType, isVLayout), mPerfEnergy);
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

  if (pimSim::get()->getDeviceType() != PIM_FUNCTIONAL) {
    pimObjInfo &objSrc1 = m_device->getResMgr()->getObjInfo(m_src1);
    pimObjInfo &objSrc2 = m_device->getResMgr()->getObjInfo(m_src2);
    objSrc1.syncFromSimulatedMem();
    objSrc2.syncFromSimulatedMem();
  }

  const pimObjInfo& objSrc1 = m_device->getResMgr()->getObjInfo(m_src1);
  unsigned numRegions = objSrc1.getRegions().size();
  computeAllRegions(numRegions);

  if (pimSim::get()->getDeviceType() != PIM_FUNCTIONAL) {
    const pimObjInfo &objDest = m_device->getResMgr()->getObjInfo(m_dest);
    objDest.syncToSimulatedMem();
  }

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
  if (!isAssociated(objSrc1, objSrc2) || !isAssociated(objSrc1, objDest) || !isCompatibleType(objSrc1, objSrc2) || !isConvertibleType(objSrc1, objDest)) {
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
  pimObjInfo& objDest = m_device->getResMgr()->getObjInfo(m_dest);

  PimDataType dataType = objSrc1.getDataType();

  const pimRegion& src1Region = objSrc1.getRegions()[index];

  // perform the computation
  uint64_t elemIdxBegin = src1Region.getElemIdxBegin();
  unsigned numElementsInRegion = src1Region.getNumElemInRegion();
  for (unsigned j = 0; j < numElementsInRegion; ++j) {
    uint64_t elemIdx = elemIdxBegin + j;

    if (dataType == PIM_INT8 || dataType == PIM_INT16 || dataType == PIM_INT32 || dataType == PIM_INT64 || dataType == PIM_UINT8 || dataType == PIM_UINT16 || dataType == PIM_UINT32 || dataType == PIM_UINT64) {
      uint64_t operandBits1 = objSrc1.getElementBits(elemIdx);
      uint64_t operandBits2 = objSrc2.getElementBits(elemIdx);
      // The following if-else block is the perfect example of where a Template would have been much more cleaner and efficient and less error prone
      if (dataType == PIM_INT8 || dataType == PIM_INT16 || dataType == PIM_INT32 || dataType == PIM_INT64) {
        int64_t operand1 = pimUtils::signExt(operandBits1, dataType);
        int64_t operand2 = pimUtils::signExt(operandBits2, dataType);
        int64_t result = 0;
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
        case PimCmdEnum::SCALED_ADD: result = (operand1 * m_scalarValue) + operand2; break;
        default:
          std::printf("PIM-Error: Unexpected cmd type %d\n", static_cast<int>(m_cmdType));
          assert(0);
        }
        objDest.setElement(elemIdx, result);
      } else {
        uint64_t operand1 = operandBits1;
        uint64_t operand2 = operandBits2;
        uint64_t result = 0;
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
        case PimCmdEnum::SCALED_ADD: result = (operand1 * m_scalarValue) + operand2; break;
        default:
          std::printf("PIM-Error: Unexpected cmd type %d\n", static_cast<int>(m_cmdType));
          assert(0);
        }
        objDest.setElement(elemIdx, result);
      }
    } else if (dataType == PIM_FP32) {
      uint64_t operandBits1 = objSrc1.getElementBits(elemIdx);
      uint64_t operandBits2 = objSrc2.getElementBits(elemIdx);
      float operand1 = pimUtils::castBitsToType<float>(operandBits1);
      float operand2 = pimUtils::castBitsToType<float>(operandBits2);
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
        std::printf("PIM-Error: Unsupported FP32 cmd type %d\n", static_cast<int>(m_cmdType));
        assert(0);
      }
      objDest.setElement(elemIdx, result);
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

  pimeval::perfEnergy mPerfEnergy = pimSim::get()->getPerfEnergyModel()->getPerfEnergyForFunc2(m_cmdType, objSrc1);
  pimSim::get()->getStatsMgr()->recordCmd(getName(dataType, isVLayout), mPerfEnergy);
  return true;
}


//! @brief  PIM CMD: redsum non-ranged/ranged
template <typename T> bool
pimCmdRedSum<T>::execute()
{
  #if defined(DEBUG)
  std::printf("PIM-Info: %s (obj id %d)\n", getName().c_str(), m_src);
  #endif

  if (!sanityCheck()) {
    return false;
  }

  if (pimSim::get()->getDeviceType() != PIM_FUNCTIONAL) {
    pimObjInfo &objSrc = m_device->getResMgr()->getObjInfo(m_src);
    objSrc.syncFromSimulatedMem();
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
template <typename T> bool
pimCmdRedSum<T>::sanityCheck() const
{
  pimResMgr* resMgr = m_device->getResMgr();
  if (!isValidObjId(resMgr, m_src) || !m_result) {
    return false;
  }
  return true;
}

//! @brief  PIM CMD: redsum non-ranged/ranged - compute region
template <typename T> bool
pimCmdRedSum<T>::computeRegion(unsigned index)
{
  const pimObjInfo& objSrc = m_device->getResMgr()->getObjInfo(m_src);
  const pimRegion& srcRegion = objSrc.getRegions()[index];

  unsigned numElementsInRegion = srcRegion.getNumElemInRegion();
  uint64_t currIdx = srcRegion.getElemIdxBegin();
  for (unsigned j = 0; j < numElementsInRegion && currIdx < m_idxEnd; ++j) {
    if (currIdx >= m_idxBegin) {
      uint64_t operandBits = objSrc.getElementBits(currIdx);
      T operand = pimUtils::signExt(operandBits, objSrc.getDataType());
      m_regionSum[index] += operand;
    }
    currIdx += 1;
  }
  return true;
}

//! @brief  PIM CMD: redsum non-ranged/ranged - update stats
template <typename T> bool
pimCmdRedSum<T>::updateStats() const
{
  const pimObjInfo& objSrc = m_device->getResMgr()->getObjInfo(m_src);
  PimDataType dataType = objSrc.getDataType();
  bool isVLayout = objSrc.isVLayout();

  unsigned numPass = 0;
  if (m_cmdType == PimCmdEnum::REDSUM_RANGE) {
    // determine numPass for ranged redSum
    std::unordered_map<PimCoreId, unsigned> activeRegionPerCore;
    uint64_t index = 0;
    for (const auto& region : objSrc.getRegions()) {
      PimCoreId coreId = region.getCoreId();
      unsigned numElementsInRegion = region.getNumElemInRegion();
      bool isActive = index < m_idxEnd && index + numElementsInRegion - 1 >= m_idxBegin;
      if (isActive) {
        activeRegionPerCore[coreId]++;
      }
      index += numElementsInRegion;
    }
    for (const auto& [coreId, count] : activeRegionPerCore) {
      if (numPass < count) {
        numPass = count;
      }
    }
  } else {
    numPass = objSrc.getMaxNumRegionsPerCore();
  }

  pimeval::perfEnergy mPerfEnergy = pimSim::get()->getPerfEnergyModel()->getPerfEnergyForRedSum(m_cmdType, objSrc, numPass);
  pimSim::get()->getStatsMgr()->recordCmd(getName(dataType, isVLayout), mPerfEnergy);
  return true;
}

//! @brief  PIM CMD: broadcast a value to all elements
bool
pimCmdBroadcast::execute()
{
  #if defined(DEBUG)
  std::printf("PIM-Info: %s (obj id %d value %llu)\n", getName().c_str(), m_dest, m_signExtBits);
  #endif

  if (!sanityCheck()) {
    return false;
  }

  const pimObjInfo& objDest = m_device->getResMgr()->getObjInfo(m_dest);
  unsigned numRegions = objDest.getRegions().size();
  computeAllRegions(numRegions);

  if (pimSim::get()->getDeviceType() != PIM_FUNCTIONAL) {
    const pimObjInfo &objDest = m_device->getResMgr()->getObjInfo(m_dest);
    objDest.syncToSimulatedMem();
  }

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
  pimObjInfo& objDest = m_device->getResMgr()->getObjInfo(m_dest);
  const pimRegion& destRegion = objDest.getRegions()[index];

  uint64_t elemIdxBegin = destRegion.getElemIdxBegin();
  unsigned numElementsInRegion = destRegion.getNumElemInRegion();

  for (unsigned j = 0; j < numElementsInRegion; ++j) {
    objDest.setElement(elemIdxBegin + j, m_signExtBits);
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

  pimeval::perfEnergy mPerfEnergy = pimSim::get()->getPerfEnergyModel()->getPerfEnergyForBroadcast(m_cmdType, objDest);
  pimSim::get()->getStatsMgr()->recordCmd(getName(dataType, isVLayout), mPerfEnergy);
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

  if (pimSim::get()->getDeviceType() != PIM_FUNCTIONAL) {
    pimObjInfo &objSrc = m_device->getResMgr()->getObjInfo(m_src);
    objSrc.syncFromSimulatedMem();
  }

  pimObjInfo& objSrc = m_device->getResMgr()->getObjInfo(m_src);
  unsigned numRegions = objSrc.getRegions().size();
  m_regionBoundary.resize(numRegions, 0);

  computeAllRegions(numRegions);

  // handle region boundaries
  if (m_cmdType == PimCmdEnum::ROTATE_ELEM_R || m_cmdType == PimCmdEnum::SHIFT_ELEM_R) {
    for (unsigned i = 0; i < numRegions; ++i) {
      const pimRegion &srcRegion = objSrc.getRegions()[i];
      uint64_t elemIdxBegin = srcRegion.getElemIdxBegin();
      uint64_t val = 0;
      if (i == 0 && m_cmdType == PimCmdEnum::ROTATE_ELEM_R) {
        val = m_regionBoundary[numRegions - 1];
      } else if (i > 0) {
        val = m_regionBoundary[i - 1];
      }
      objSrc.setElement(elemIdxBegin, val);
    }
  } else if (m_cmdType == PimCmdEnum::ROTATE_ELEM_L || m_cmdType == PimCmdEnum::SHIFT_ELEM_L) {
    for (unsigned i = 0; i < numRegions; ++i) {
      const pimRegion &srcRegion = objSrc.getRegions()[i];
      unsigned numElementsInRegion = srcRegion.getNumElemInRegion();
      uint64_t elemIdxBegin = srcRegion.getElemIdxBegin();
      uint64_t val = 0;
      if (i == numRegions - 1 && m_cmdType == PimCmdEnum::ROTATE_ELEM_L) {
        val = m_regionBoundary[0];
      } else if (i < numRegions - 1) {
        val = m_regionBoundary[i + 1];
      }
      objSrc.setElement(elemIdxBegin + numElementsInRegion - 1, val);
    }
  } else {
    assert(0);
  }

  if (pimSim::get()->getDeviceType() != PIM_FUNCTIONAL) {
    const pimObjInfo &objSrc = m_device->getResMgr()->getObjInfo(m_src);
    objSrc.syncToSimulatedMem();
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
  pimObjInfo& objSrc = m_device->getResMgr()->getObjInfo(m_src);

  const pimRegion& srcRegion = objSrc.getRegions()[index];

  // read out values
  uint64_t elemIdxBegin = srcRegion.getElemIdxBegin();
  unsigned numElementsInRegion = srcRegion.getNumElemInRegion();
  std::vector<uint64_t> regionVector(numElementsInRegion);
  for (unsigned j = 0; j < numElementsInRegion; ++j) {
    regionVector[j] = objSrc.getElementBits(elemIdxBegin + j);
  }

  // perform rotation
  if (m_cmdType == PimCmdEnum::ROTATE_ELEM_R || m_cmdType == PimCmdEnum::SHIFT_ELEM_R) {
    m_regionBoundary[index] = regionVector[numElementsInRegion - 1];
    uint64_t carry = 0;
    for (unsigned j = 0; j < numElementsInRegion; ++j) {
      uint64_t temp = regionVector[j];
      regionVector[j] = carry;
      carry = temp;
    }
  } else if (m_cmdType == PimCmdEnum::ROTATE_ELEM_L || m_cmdType == PimCmdEnum::SHIFT_ELEM_L) {
    m_regionBoundary[index] = regionVector[0];
    uint64_t carry = 0;
    for (int j = numElementsInRegion - 1; j >= 0; --j) {
      uint64_t temp = regionVector[j];
      regionVector[j] = carry;
      carry = temp;
    }
  } else {
    assert(0);
  }

  // write back values
  for (unsigned j = 0; j < numElementsInRegion; ++j) {
    objSrc.setElement(elemIdxBegin + j, regionVector[j]);
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

  pimeval::perfEnergy mPerfEnergy = pimSim::get()->getPerfEnergyModel()->getPerfEnergyForRotate(m_cmdType, objSrc);
  pimSim::get()->getStatsMgr()->recordCmd(getName(dataType, isVLayout), mPerfEnergy);
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
  pimeval::perfEnergy prfEnrgy;
  pimSim::get()->getStatsMgr()->recordCmd(getName(), prfEnrgy);
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
  pimeval::perfEnergy prfEnrgy;
  pimSim::get()->getStatsMgr()->recordCmd(getName(), prfEnrgy);
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
        std::printf("PIM-Error: Unexpected cmd type %d\n", static_cast<int>(m_cmdType));
        assert(0);
      }
    }
  }

  // Update stats
  pimeval::perfEnergy prfEnrgy;
  pimSim::get()->getStatsMgr()->recordCmd(getName(), prfEnrgy);
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
  pimeval::perfEnergy prfEnrgy;
  pimSim::get()->getStatsMgr()->recordCmd(getName(), prfEnrgy);
  return true;
}

//! @brief  Pim CMD: SIMDRAM: Analog based multi-row AP and AAP
bool
pimCmdAnalogAAP::execute()
{
  #if defined(DEBUG)
  printDebugInfo();
  #endif

  if (m_srcRows.empty()) {
    return false;
  }

  pimResMgr* resMgr = m_device->getResMgr();
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_srcRows[0].first);

  // 1st activate: compute majority
  std::unordered_set<unsigned> visitedRows;
  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];
    PimCoreId coreId = srcRegion.getCoreId();
    pimCore &core = m_device->getCore(coreId);

    std::vector<std::pair<unsigned, bool>> rowIdxs;
    for (const auto& objOfst : m_srcRows) {
      if (!isValidObjId(resMgr, objOfst.first)) {
        return false;
      }
      const pimObjInfo& obj = resMgr->getObjInfo(objOfst.first);
      if (!isAssociated(objSrc, obj)) {
        return false;
      }
      unsigned ofst = objOfst.second;
      unsigned idx = obj.getRegions()[i].getRowIdx() + ofst;
      bool isDCCN = obj.isDualContactRef();
      rowIdxs.emplace_back(idx, isDCCN);
      if (i == 0) { // sanity check
        if (visitedRows.find(idx) == visitedRows.end()) {
          visitedRows.insert(idx);
        } else {
          std::printf("PIM-Error: Cannot access same src row multiple times during AP/AAP\n");
          return false;
        }
      }
    }
    core.readMultiRows(rowIdxs);
  }

  // 2nd activate: write multiple rows
  if (!m_destRows.empty()) {
    for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
      const pimRegion& srcRegion = objSrc.getRegions()[i];
      PimCoreId coreId = srcRegion.getCoreId();
      pimCore &core = m_device->getCore(coreId);

      std::vector<std::pair<unsigned, bool>> rowIdxs;
      for (const auto& objOfst : m_destRows) {
        if (!isValidObjId(resMgr, objOfst.first)) {
          return false;
        }
        const pimObjInfo& obj = resMgr->getObjInfo(objOfst.first);
        if (!isAssociated(objSrc, obj)) {
          return false;
        }
        unsigned ofst = objOfst.second;
        unsigned idx = obj.getRegions()[i].getRowIdx() + ofst;
        bool isDCCN = obj.isDualContactRef();
        rowIdxs.emplace_back(idx, isDCCN);
        if (i == 0) { // sanity check
          if (visitedRows.find(idx) == visitedRows.end()) {
            visitedRows.insert(idx);
          } else {
            std::printf("PIM-Error: Cannot access same src/dest row multiple times during AP/AAP\n");
            return false;
          }
        }
      }
      core.writeMultiRows(rowIdxs);
    }
  }

  // Update stats
  std::string cmdName = getName();
  cmdName += "@" + std::to_string(m_srcRows.size()) + "," + std::to_string(m_destRows.size());
  pimeval::perfEnergy prfEnrgy;
  pimSim::get()->getStatsMgr()->recordCmd(cmdName, prfEnrgy);
  return true;
}

//! @brief  Pim CMD: SIMDRAM: AP/AAP debug info
void
pimCmdAnalogAAP::printDebugInfo() const
{
  std::string msg;
  for (const auto &kv : m_srcRows) {
    msg += " " + std::to_string(kv.first) + "[" + std::to_string(kv.second) + "]";
  }
  if (!m_destRows.empty()) {
    msg += " ->";
  }
  for (const auto &kv : m_destRows) {
    msg += " " + std::to_string(kv.first) + "[" + std::to_string(kv.second) + "]";
  }
  std::printf("PIM-Info: %s (#src = %lu, #dest = %lu, rows =%s)\n",
              getName().c_str(), m_srcRows.size(), m_destRows.size(), msg.c_str());
}

// Explicit template instantiation
template class pimCmdRedSum<uint64_t>;
template class pimCmdRedSum<int64_t>;
