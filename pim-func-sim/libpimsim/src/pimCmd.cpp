// File: pimCmd.cpp
// PIM Functional Simulator - PIM Commands
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "pimCmd.h"
#include "pimDevice.h"
#include "pimCore.h"
#include "pimResMgr.h"
#include <cstdio>
#include <cmath>


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

//! @brief  PIM CMD: add v-layout
bool
pimCmdAddV::execute(pimDevice* device)
{
  std::printf("PIM-Info: AddV (obj id %d + %d -> %d)\n", m_src1, m_src2, m_dest);

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

  for (unsigned i = 0; i < objSrc1.getRegions().size(); ++i) {
    const pimRegion& src1Region = objSrc1.getRegions()[i];
    const pimRegion& src2Region = objSrc2.getRegions()[i];
    const pimRegion& destRegion = objDest.getRegions()[i];

    if (src1Region.getNumAllocRows() != src2Region.getNumAllocRows() || src1Region.getNumAllocRows() != destRegion.getNumAllocRows()) {
      std::printf("PIM-Error: Operands %d, %d and %d do not have equal bit length for v-layout\n", m_src1, m_src2, m_dest);
      return false;
    }

    PimCoreId coreId = src1Region.getCoreId();

    // perform the computation
    unsigned colIdx = src1Region.getColIdx();
    unsigned numAllocCols = src1Region.getNumAllocCols();
    for (unsigned j = 0; j < numAllocCols; ++j) {
      auto operandVal1 = device->getCore(coreId).getB32V(src1Region.getRowIdx(), colIdx + j);
      auto operandVal2 = device->getCore(coreId).getB32V(src2Region.getRowIdx(), colIdx + j);
      int operand1 = *reinterpret_cast<unsigned*>(&operandVal1);
      int operand2 = *reinterpret_cast<unsigned*>(&operandVal2);
      int result = operand1 + operand2;
      device->getCore(coreId).setB32V(destRegion.getRowIdx(), colIdx + j, *reinterpret_cast<unsigned*>(&result));
    }
  }

  return true;
}

//! @brief  PIM CMD: abs v-layout
bool
pimCmdAbsV::execute(pimDevice* device)
{
  std::printf("PIM-Info: AbsV (obj id %d -> %d)\n", m_src, m_dest);

  pimResMgr* resMgr = device->getResMgr();
  if (!isVAligned(m_src, m_dest, resMgr)) {
    return false;
  }

  const pimObjInfo& objSrc = resMgr->getObjInfo(m_src);
  const pimObjInfo& objDest = resMgr->getObjInfo(m_dest);

  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];
    const pimRegion& destRegion = objDest.getRegions()[i];

    if (srcRegion.getNumAllocRows() != destRegion.getNumAllocRows()) {
      std::printf("PIM-Error: Operands %d and %d do not have equal bit length for v-layout\n", m_src, m_dest);
      return false;
    }

    PimCoreId coreId = srcRegion.getCoreId();

    // perform the computation
    unsigned colIdx = srcRegion.getColIdx();
    unsigned numAllocCols = srcRegion.getNumAllocCols();
    for (unsigned j = 0; j < numAllocCols; ++j) {
      int operand = static_cast<int>(device->getCore(coreId).getB32V(srcRegion.getRowIdx(), colIdx + j));
      int result = std::abs(operand);
      device->getCore(coreId).setB32V(destRegion.getRowIdx(), colIdx + j, static_cast<unsigned>(result));
    }
  }

  return true;
}

//! @brief  PIM CMD: redsum v-layout
bool
pimCmdRedSum::execute(pimDevice* device)
{
  std::printf("PIM-Info: RedSum (obj id %d)\n", m_src);

  pimResMgr* resMgr = device->getResMgr();

  const pimObjInfo& objSrc = resMgr->getObjInfo(m_src);

  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];

    PimCoreId coreId = srcRegion.getCoreId();

    // perform the computation
    unsigned colIdx = srcRegion.getColIdx();
    unsigned numAllocCols = srcRegion.getNumAllocCols();
    for (unsigned j = 0; j < numAllocCols; ++j) {
      int operand = static_cast<int>(device->getCore(coreId).getB32V(srcRegion.getRowIdx(), colIdx + j));
      m_result += operand;
    }
  }

  return true;
}

//! @brief  PIM CMD: redsum range v-layout
bool
pimCmdRedSumRanged::execute(pimDevice* device)
{
  std::printf("PIM-Info: RedSumRanged (obj id %d)\n", m_src);

  pimResMgr* resMgr = device->getResMgr();
  //first get the start region; then get the column number you have to access for that region. Then keep increasing the idx count.
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_src);
  unsigned currIDX = 0;
  for (unsigned i = 0; i < objSrc.getRegions().size() && currIDX < m_idxEnd; ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];

    PimCoreId coreId = srcRegion.getCoreId();

    // perform the computation
    unsigned colIdx = srcRegion.getColIdx();
    unsigned numAllocCols = srcRegion.getNumAllocCols();
    for (unsigned j = 0; j < numAllocCols && currIDX < m_idxEnd; ++j) {
      if (currIDX >= m_idxBegin) {
        int operand = static_cast<int>(device->getCore(coreId).getB32V(srcRegion.getRowIdx(), colIdx + j));
        m_result += operand;
      } 
      currIDX += 1;      
    }
  }

  return true;
}

//! @brief  PIM CMD: sub v-layout
bool
pimCmdSubV::execute(pimDevice* device)
{ 
  std::printf("PIM-Info: SubV (obj id %d - %d -> %d)\n", m_src1, m_src2, m_dest);

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

  for (unsigned i = 0; i < objSrc1.getRegions().size(); ++i) {
    const pimRegion& src1Region = objSrc1.getRegions()[i];
    const pimRegion& src2Region = objSrc2.getRegions()[i];
    const pimRegion& destRegion = objDest.getRegions()[i];

    if (src1Region.getNumAllocRows() != src2Region.getNumAllocRows() || src1Region.getNumAllocRows() != destRegion.getNumAllocRows()) {
      std::printf("PIM-Error: Operands %d, %d and %d do not have equal bit length for v-layout \n", m_src1, m_src2, m_dest);
      return false;
    }

    PimCoreId coreId = src1Region.getCoreId();

    // perform the computation
    unsigned colIdx = src1Region.getColIdx();
    unsigned numAllocCols = src1Region.getNumAllocCols();
    for (unsigned j = 0; j < numAllocCols; ++j) {
      auto operandVal1 = device->getCore(coreId).getB32V(src1Region.getRowIdx(), colIdx + j);
      auto operandVal2 = device->getCore(coreId).getB32V(src2Region.getRowIdx(), colIdx + j);
      int operand1 = *reinterpret_cast<unsigned*>(&operandVal1);
      int operand2 = *reinterpret_cast<unsigned*>(&operandVal2);
      int result = operand1 - operand2;
      device->getCore(coreId).setB32V(destRegion.getRowIdx(), colIdx + j, *reinterpret_cast<unsigned*>(&result));
    }
  }

  return true;
}

//! @brief  PIM CMD: division v-layout
bool
pimCmdDivV::execute(pimDevice* device)
{ 
  std::printf("PIM-Info: DivV (obj id %d / %d -> %d)\n", m_src1, m_src2, m_dest);

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

  for (unsigned i = 0; i < objSrc1.getRegions().size(); ++i) {
    const pimRegion& src1Region = objSrc1.getRegions()[i];
    const pimRegion& src2Region = objSrc2.getRegions()[i];
    const pimRegion& destRegion = objDest.getRegions()[i];

    if (src1Region.getNumAllocRows() != src2Region.getNumAllocRows() || src1Region.getNumAllocRows() != destRegion.getNumAllocRows()) {
      std::printf("PIM-Error: Operands %d, %d and %d do not have equal bit length for v-layout\n", m_src1, m_src2, m_dest);
      return false;
    }

    PimCoreId coreId = src1Region.getCoreId();

    // perform the computation
    unsigned colIdx = src1Region.getColIdx();
    unsigned numAllocCols = src1Region.getNumAllocCols();
    for (unsigned j = 0; j < numAllocCols; ++j) {
      auto operandVal1 = device->getCore(coreId).getB32V(src1Region.getRowIdx(), colIdx + j);
      auto operandVal2 = device->getCore(coreId).getB32V(src2Region.getRowIdx(), colIdx + j);
      int operand1 = *reinterpret_cast<unsigned*>(&operandVal1);
      int operand2 = *reinterpret_cast<unsigned*>(&operandVal2);
      if (operand2 == 0) {
        std::printf("PIM-Error: Division by zero\n");
        return false;
      }
      int result = operand1 / operand2;
      device->getCore(coreId).setB32V(destRegion.getRowIdx(), colIdx + j, *reinterpret_cast<unsigned*>(&result));
    }
  }

  return true;
}

//! @brief  PIM CMD: mul v-layout
bool
pimCmdMulV::execute(pimDevice* device)
{ 
  std::printf("PIM-Info: MulV (obj id %d * %d -> %d)\n", m_src1, m_src2, m_dest);

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

  for (unsigned i = 0; i < objSrc1.getRegions().size(); ++i) {
    const pimRegion& src1Region = objSrc1.getRegions()[i];
    const pimRegion& src2Region = objSrc2.getRegions()[i];
    const pimRegion& destRegion = objDest.getRegions()[i];

    if (src1Region.getNumAllocRows() != src2Region.getNumAllocRows() || src1Region.getNumAllocRows() != destRegion.getNumAllocRows()) {
      std::printf("PIM-Error: Operands %d, %d and %d do not have equal bit length for v-layout\n", m_src1, m_src2, m_dest);
      return false;
    }

    PimCoreId coreId = src1Region.getCoreId();

    // perform the computation
    unsigned colIdx = src1Region.getColIdx();
    unsigned numAllocCols = src1Region.getNumAllocCols();
    for (unsigned j = 0; j < numAllocCols; ++j) {
      auto operandVal1 = device->getCore(coreId).getB32V(src1Region.getRowIdx(), colIdx + j);
      auto operandVal2 = device->getCore(coreId).getB32V(src2Region.getRowIdx(), colIdx + j);
      int operand1 = *reinterpret_cast<unsigned*>(&operandVal1);
      int operand2 = *reinterpret_cast<unsigned*>(&operandVal2);
      int result = operand1 * operand2;
      device->getCore(coreId).setB32V(destRegion.getRowIdx(), colIdx + j, *reinterpret_cast<unsigned*>(&result));
    }
  }

  return true;
}

//! @brief  PIM CMD: or v-layout
bool
pimCmdOrV::execute(pimDevice* device)
{ 
  std::printf("PIM-Info: OrV (obj id %d | %d -> %d)\n", m_src1, m_src2, m_dest);

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

  for (unsigned i = 0; i < objSrc1.getRegions().size(); ++i) {
    const pimRegion& src1Region = objSrc1.getRegions()[i];
    const pimRegion& src2Region = objSrc2.getRegions()[i];
    const pimRegion& destRegion = objDest.getRegions()[i];

    if (src1Region.getNumAllocRows() != src2Region.getNumAllocRows() || src1Region.getNumAllocRows() != destRegion.getNumAllocRows()) {
      std::printf("PIM-Error: Operands %d, %d and %d do not have equal bit length for v-layout\n", m_src1, m_src2, m_dest);
      return false;
    }

    PimCoreId coreId = src1Region.getCoreId();

    // perform the computation
    unsigned colIdx = src1Region.getColIdx();
    unsigned numAllocCols = src1Region.getNumAllocCols();
    for (unsigned j = 0; j < numAllocCols; ++j) {
      auto operandVal1 = device->getCore(coreId).getB32V(src1Region.getRowIdx(), colIdx + j);
      auto operandVal2 = device->getCore(coreId).getB32V(src2Region.getRowIdx(), colIdx + j);
      int operand1 = *reinterpret_cast<unsigned*>(&operandVal1);
      int operand2 = *reinterpret_cast<unsigned*>(&operandVal2);
      int result = operand1 | operand2;
      device->getCore(coreId).setB32V(destRegion.getRowIdx(), colIdx + j, *reinterpret_cast<unsigned*>(&result));
    }
  }

  return true;
}

//! @brief  PIM CMD: and v-layout
bool
pimCmdAndV::execute(pimDevice* device)
{ 
  std::printf("PIM-Info: AndV (obj id %d & %d -> %d)\n", m_src1, m_src2, m_dest);

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

  for (unsigned i = 0; i < objSrc1.getRegions().size(); ++i) {
    const pimRegion& src1Region = objSrc1.getRegions()[i];
    const pimRegion& src2Region = objSrc2.getRegions()[i];
    const pimRegion& destRegion = objDest.getRegions()[i];

    if (src1Region.getNumAllocRows() != src2Region.getNumAllocRows() || src1Region.getNumAllocRows() != destRegion.getNumAllocRows()) {
      std::printf("PIM-Error: Operands %d, %d and %d do not have equal bit length for v-layout\n", m_src1, m_src2, m_dest);
      return false;
    }

    PimCoreId coreId = src1Region.getCoreId();

    // perform the computation
    unsigned colIdx = src1Region.getColIdx();
    unsigned numAllocCols = src1Region.getNumAllocCols();
    for (unsigned j = 0; j < numAllocCols; ++j) {
      auto operandVal1 = device->getCore(coreId).getB32V(src1Region.getRowIdx(), colIdx + j);
      auto operandVal2 = device->getCore(coreId).getB32V(src2Region.getRowIdx(), colIdx + j);
      int operand1 = *reinterpret_cast<unsigned*>(&operandVal1);
      int operand2 = *reinterpret_cast<unsigned*>(&operandVal2);
      int result = operand1 & operand2;
      device->getCore(coreId).setB32V(destRegion.getRowIdx(), colIdx + j, *reinterpret_cast<unsigned*>(&result));
    }
  }

  return true;
}

//! @brief  PIM CMD: xor v-layout
bool
pimCmdXorV::execute(pimDevice* device)
{ 
  std::printf("PIM-Info: XorV (obj id %d ^ %d -> %d)\n", m_src1, m_src2, m_dest);

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

  for (unsigned i = 0; i < objSrc1.getRegions().size(); ++i) {
    const pimRegion& src1Region = objSrc1.getRegions()[i];
    const pimRegion& src2Region = objSrc2.getRegions()[i];
    const pimRegion& destRegion = objDest.getRegions()[i];

    if (src1Region.getNumAllocRows() != src2Region.getNumAllocRows() || src1Region.getNumAllocRows() != destRegion.getNumAllocRows()) {
      std::printf("PIM-Error: Operands %d, %d and %d do not have equal bit length for v-layout\n", m_src1, m_src2, m_dest);
      return false;
    }

    PimCoreId coreId = src1Region.getCoreId();

    // perform the computation
    unsigned colIdx = src1Region.getColIdx();
    unsigned numAllocCols = src1Region.getNumAllocCols();
    for (unsigned j = 0; j < numAllocCols; ++j) {
      auto operandVal1 = device->getCore(coreId).getB32V(src1Region.getRowIdx(), colIdx + j);
      auto operandVal2 = device->getCore(coreId).getB32V(src2Region.getRowIdx(), colIdx + j);
      int operand1 = *reinterpret_cast<unsigned*>(&operandVal1);
      int operand2 = *reinterpret_cast<unsigned*>(&operandVal2);
      
      int result = operand1 ^ operand2;
      device->getCore(coreId).setB32V(destRegion.getRowIdx(), colIdx + j, *reinterpret_cast<unsigned*>(&result));
    }
  }

  return true;
}

//! @brief  PIM CMD: gt v-layout
bool
pimCmdGTV::execute(pimDevice* device)
{ 
  std::printf("PIM-Info: GTV (obj id %d > %d -> %d)\n", m_src1, m_src2, m_dest);

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

  for (unsigned i = 0; i < objSrc1.getRegions().size(); ++i) {
    const pimRegion& src1Region = objSrc1.getRegions()[i];
    const pimRegion& src2Region = objSrc2.getRegions()[i];
    const pimRegion& destRegion = objDest.getRegions()[i];

    if (src1Region.getNumAllocRows() != src2Region.getNumAllocRows() || src1Region.getNumAllocRows() != destRegion.getNumAllocRows()) {
      std::printf("PIM-Error: Operands %d, %d and %d do not have equal bit length for v-layout\n", m_src1, m_src2, m_dest);
      return false;
    }

    PimCoreId coreId = src1Region.getCoreId();

    // perform the computation
    unsigned colIdx = src1Region.getColIdx();
    unsigned numAllocCols = src1Region.getNumAllocCols();
    for (unsigned j = 0; j < numAllocCols; ++j) {
      auto operandVal1 = device->getCore(coreId).getB32V(src1Region.getRowIdx(), colIdx + j);
      auto operandVal2 = device->getCore(coreId).getB32V(src2Region.getRowIdx(), colIdx + j);
      int operand1 = *reinterpret_cast<unsigned*>(&operandVal1);
      int operand2 = *reinterpret_cast<unsigned*>(&operandVal2);
      int result = operand1 > operand2 ? 1 : 0;
      device->getCore(coreId).setB32V(destRegion.getRowIdx(), colIdx + j, *reinterpret_cast<unsigned*>(&result));
    }
  }

  return true;
}

//! @brief  PIM CMD: lt v-layout
bool
pimCmdLTV::execute(pimDevice* device)
{ 
  std::printf("PIM-Info: LTV (obj id %d < %d -> %d)\n", m_src1, m_src2, m_dest);

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

  for (unsigned i = 0; i < objSrc1.getRegions().size(); ++i) {
    const pimRegion& src1Region = objSrc1.getRegions()[i];
    const pimRegion& src2Region = objSrc2.getRegions()[i];
    const pimRegion& destRegion = objDest.getRegions()[i];

    if (src1Region.getNumAllocRows() != src2Region.getNumAllocRows() || src1Region.getNumAllocRows() != destRegion.getNumAllocRows()) {
      std::printf("PIM-Error: Operands %d, %d and %d do not have equal bit length for v-layout\n", m_src1, m_src2, m_dest);
      return false;
    }

    PimCoreId coreId = src1Region.getCoreId();

    // perform the computation
    unsigned colIdx = src1Region.getColIdx();
    unsigned numAllocCols = src1Region.getNumAllocCols();
    for (unsigned j = 0; j < numAllocCols; ++j) {
      auto operandVal1 = device->getCore(coreId).getB32V(src1Region.getRowIdx(), colIdx + j);
      auto operandVal2 = device->getCore(coreId).getB32V(src2Region.getRowIdx(), colIdx + j);
      int operand1 = *reinterpret_cast<unsigned*>(&operandVal1);
      int operand2 = *reinterpret_cast<unsigned*>(&operandVal2);
      int result = operand1 < operand2 ? 1 : 0;
      device->getCore(coreId).setB32V(destRegion.getRowIdx(), colIdx + j, *reinterpret_cast<unsigned*>(&result));
    }
  }

  return true;
}

//! @brief  PIM CMD: eq v-layout
bool
pimCmdEQV::execute(pimDevice* device)
{ 
  std::printf("PIM-Info: EQV (obj id %d == %d -> %d)\n", m_src1, m_src2, m_dest);

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

  for (unsigned i = 0; i < objSrc1.getRegions().size(); ++i) {
    const pimRegion& src1Region = objSrc1.getRegions()[i];
    const pimRegion& src2Region = objSrc2.getRegions()[i];
    const pimRegion& destRegion = objDest.getRegions()[i];

    if (src1Region.getNumAllocRows() != src2Region.getNumAllocRows() || src1Region.getNumAllocRows() != destRegion.getNumAllocRows()) {
      std::printf("PIM-Error: Operands %d, %d and %d do not have equal bit length for v-layout\n", m_src1, m_src2, m_dest);
      return false;
    }

    PimCoreId coreId = src1Region.getCoreId();

    // perform the computation
    unsigned colIdx = src1Region.getColIdx();
    unsigned numAllocCols = src1Region.getNumAllocCols();
    for (unsigned j = 0; j < numAllocCols; ++j) {
      auto operandVal1 = device->getCore(coreId).getB32V(src1Region.getRowIdx(), colIdx + j);
      auto operandVal2 = device->getCore(coreId).getB32V(src2Region.getRowIdx(), colIdx + j);
      int operand1 = *reinterpret_cast<unsigned*>(&operandVal1);
      int operand2 = *reinterpret_cast<unsigned*>(&operandVal2);
      int result = (operand1 == operand2) ? 1 : 0;
      device->getCore(coreId).setB32V(destRegion.getRowIdx(), colIdx + j, *reinterpret_cast<unsigned*>(&result));
    }
  }

  return true;
}

//! @brief  PIM CMD: min v-layout
bool
pimCmdMinV::execute(pimDevice* device)
{ 
  std::printf("PIM-Info: MinV (obj id min(%d, %d) -> %d)\n", m_src1, m_src2, m_dest);

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

  for (unsigned i = 0; i < objSrc1.getRegions().size(); ++i) {
    const pimRegion& src1Region = objSrc1.getRegions()[i];
    const pimRegion& src2Region = objSrc2.getRegions()[i];
    const pimRegion& destRegion = objDest.getRegions()[i];

    if (src1Region.getNumAllocRows() != src2Region.getNumAllocRows() || src1Region.getNumAllocRows() != destRegion.getNumAllocRows()) {
      std::printf("PIM-Error: Operands %d, %d and %d do not have equal bit length for v-layout\n", m_src1, m_src2, m_dest);
      return false;
    }

    PimCoreId coreId = src1Region.getCoreId();

    // perform the computation
    unsigned colIdx = src1Region.getColIdx();
    unsigned numAllocCols = src1Region.getNumAllocCols();
    for (unsigned j = 0; j < numAllocCols; ++j) {
      auto operandVal1 = device->getCore(coreId).getB32V(src1Region.getRowIdx(), colIdx + j);
      auto operandVal2 = device->getCore(coreId).getB32V(src2Region.getRowIdx(), colIdx + j);
      int operand1 = *reinterpret_cast<unsigned*>(&operandVal1);
      int operand2 = *reinterpret_cast<unsigned*>(&operandVal2);
      int result = (operand1 < operand2) ? operand1 :operand2;
      device->getCore(coreId).setB32V(destRegion.getRowIdx(), colIdx + j, *reinterpret_cast<unsigned*>(&result));
    }
  }

  return true;
}

//! @brief  PIM CMD: max v-layout
bool
pimCmdMaxV::execute(pimDevice* device)
{ 
  std::printf("PIM-Info: MaxV (obj id min(%d, %d) -> %d)\n", m_src1, m_src2, m_dest);

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
  
  for (unsigned i = 0; i < objSrc1.getRegions().size(); ++i) {
    const pimRegion& src1Region = objSrc1.getRegions()[i];
    const pimRegion& src2Region = objSrc2.getRegions()[i];
    const pimRegion& destRegion = objDest.getRegions()[i];

    if (src1Region.getNumAllocRows() != src2Region.getNumAllocRows() || src1Region.getNumAllocRows() != destRegion.getNumAllocRows()) {
      std::printf("PIM-Error: Operands %d, %d and %d do not have equal bit length for v-layout\n", m_src1, m_src2, m_dest);
      return false;
    }

    PimCoreId coreId = src1Region.getCoreId();

    // perform the computation
    unsigned colIdx = src1Region.getColIdx();
    unsigned numAllocCols = src1Region.getNumAllocCols();
    for (unsigned j = 0; j < numAllocCols; ++j) {
      auto operandVal1 = device->getCore(coreId).getB32V(src1Region.getRowIdx(), colIdx + j);
      auto operandVal2 = device->getCore(coreId).getB32V(src2Region.getRowIdx(), colIdx + j);
      int operand1 = *reinterpret_cast<unsigned*>(&operandVal1);
      int operand2 = *reinterpret_cast<unsigned*>(&operandVal2);
      int result = (operand1 > operand2) ? operand1 :operand2;
      device->getCore(coreId).setB32V(destRegion.getRowIdx(), colIdx + j, *reinterpret_cast<unsigned*>(&result));
    }
  }

  return true;
}

//! @brief  PIM CMD: popcount v-layout
bool
pimCmdPopCountV::execute(pimDevice* device)
{ 
  std::printf("PIM-Info: PopCountV (obj id popcount(%d) -> %d)\n", m_src, m_dest);

  pimResMgr* resMgr = device->getResMgr();
  if (!isVAligned(m_src, m_dest, resMgr)) {
    return false;
  }

  const pimObjInfo& objSrc = resMgr->getObjInfo(m_src);
  const pimObjInfo& objDest = resMgr->getObjInfo(m_dest);

  if (objSrc.getDataType() != objDest.getDataType()) {
    std::printf("PIM-Error: Cannot convert from %s to %s\n", objSrc.getDataTypeName().c_str(), objDest.getDataTypeName().c_str());
    return false;
  }
  
  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];
    const pimRegion& destRegion = objDest.getRegions()[i];

    if (srcRegion.getNumAllocRows() != destRegion.getNumAllocRows()) {
      std::printf("PIM-Error: Operands %d and %d do not have equal bit length for v-layout\n", m_src, m_dest);
      return false;
    }

    PimCoreId coreId = srcRegion.getCoreId();

    // perform the computation
    unsigned colIdx = srcRegion.getColIdx();
    unsigned numAllocCols = srcRegion.getNumAllocCols();
    for (unsigned j = 0; j < numAllocCols; ++j) {
      auto operandVal = device->getCore(coreId).getB32V(srcRegion.getRowIdx(), colIdx + j);
      int operand = *reinterpret_cast<unsigned*>(&operandVal);
      int result = 0;
      while (operand) {
        operand &= (operand - 1);
        result++;
      }
      device->getCore(coreId).setB32V(destRegion.getRowIdx(), colIdx + j, *reinterpret_cast<unsigned*>(&result));
    }
  }

  return true;
}

//! @brief  PIM CMD: rotate right v-layout
bool
pimCmdRotateRightV::execute(pimDevice* device)
{ 
  std::printf("PIM-Info: RotateRightV (obj id %d)\n", m_src);

  pimResMgr* resMgr = device->getResMgr();

  const pimObjInfo& objSrc = resMgr->getObjInfo(m_src);
  unsigned carry = 0;

  for (const auto& srcRegion : objSrc.getRegions()) {

    pimCore& core = device->getCore(srcRegion.getCoreId());

    // retrieve the values
    unsigned colIdx = srcRegion.getColIdx();
    unsigned numAllocCols = srcRegion.getNumAllocCols();
    unsigned rowIdx = srcRegion.getRowIdx();
    std::vector<unsigned> regionVector(numAllocCols);
    for (unsigned j = 0; j < numAllocCols; ++j) {
      regionVector[j] = core.getB32V(rowIdx, colIdx + j);
    }
    // Perform the rotation
    for (unsigned j = 0; j < numAllocCols ; ++j) {
        int temp = regionVector[j];
        regionVector[j] = carry;
        carry = temp;
    }
    for (unsigned j = 0; j < numAllocCols; ++j) {
      core.setB32V(srcRegion.getRowIdx(), colIdx + j, regionVector[j]);
    }
  }
  if (!objSrc.getRegions().empty()) {
    const pimRegion& srcRegion = objSrc.getRegions().front();
    device->getCore(srcRegion.getCoreId()).setB32V(srcRegion.getRowIdx(), srcRegion.getColIdx(), *reinterpret_cast<unsigned*>(&carry));
  }
  return true;
}

//! @brief  PIM CMD: rotate left v-layout
bool
pimCmdRotateLeftV::execute(pimDevice* device)
{ 
  std::printf("PIM-Info: RotateLeftV (obj id %d)\n", m_src);

  pimResMgr* resMgr = device->getResMgr();

  const pimObjInfo& objSrc = resMgr->getObjInfo(m_src);
  unsigned carry = 0;
  for (unsigned i = objSrc.getRegions().size(); i > 0; --i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i-1];
    pimCore& core = device->getCore(srcRegion.getCoreId());

    // retrieve the values
    unsigned colIdx = srcRegion.getColIdx();
    unsigned numAllocCols = srcRegion.getNumAllocCols();
    unsigned rowIdx = srcRegion.getRowIdx();
    std::vector<unsigned> regionVector(numAllocCols);
    for (unsigned j = 0; j < numAllocCols; ++j) {
      regionVector[j] = core.getB32V(rowIdx, colIdx + j);
    }
    //Perform the rotation
    for (int j = numAllocCols-1; j >= 0; --j) {
      unsigned temp = regionVector[j];
      regionVector[j] = carry;
      carry = temp;
    }
    for (unsigned j = 0; j < numAllocCols; ++j) {
      core.setB32V(srcRegion.getRowIdx(), colIdx + j, regionVector[j]);
    }
  }
  if (!objSrc.getRegions().empty()) {
    const pimRegion& srcRegion = objSrc.getRegions().back();
    device->getCore(srcRegion.getCoreId()).setB32V(srcRegion.getRowIdx(), srcRegion.getColIdx()+srcRegion.getNumAllocCols()-1, *reinterpret_cast<unsigned*>(&carry));
  }
  return true;
}

//! @brief  Pim CMD: BitSIMD-V: Read a row to SA
bool
pimCmdReadRowToSa::execute(pimDevice* device)
{
  std::printf("PIM-Info: BitSIMD-V ReadRowToSa (obj id %d ofst %u)\n", m_objId, m_ofst);
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
  return true;
}

//! @brief  Pim CMD: BitSIMD-V: Write SA to a row
bool
pimCmdWriteSaToRow::execute(pimDevice* device)
{
  std::printf("PIM-Info: BitSIMD-V WriteSaToRow (obj id %d ofst %u)\n", m_objId, m_ofst);
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
  return true;
}

//! @brief  Pim CMD: BitSIMD-V: Move row reg to reg
bool
pimCmdRRegMove::execute(pimDevice* device)
{
  std::printf("PIM-Info: BitSIMD-V RRegMove (obj-id %d src-reg %d dest-reg %d)\n", m_objId, m_src, m_dest);
  pimResMgr* resMgr = device->getResMgr();
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_objId);
  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];
    PimCoreId coreId = srcRegion.getCoreId();
    for (unsigned j = 0; j < device->getNumCols(); j++) {
      device->getCore(coreId).getRowReg(m_dest)[j] = device->getCore(coreId).getRowReg(m_src)[j];
    }
  }
  return true;
}

//! @brief  Pim CMD: BitSIMD-V: Set row reg to 0/1
bool
pimCmdRRegSet::execute(pimDevice* device)
{
  std::printf("PIM-Info: BitSIMD-V RRegSet (obj-id %d src-reg %d val %d)\n", m_objId, m_dest, m_val);
  pimResMgr* resMgr = device->getResMgr();
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_objId);
  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];
    PimCoreId coreId = srcRegion.getCoreId();
    for (unsigned j = 0; j < device->getNumCols(); j++) {
      device->getCore(coreId).getRowReg(m_dest)[j] = m_val;
    }
  }
  return true;
}

//! @brief  Pim CMD: BitSIMD-V: row reg dest = !src
bool
pimCmdRRegNot::execute(pimDevice* device)
{
  std::printf("PIM-Info: BitSIMD-V RRegNot (obj-id %d src-reg %d dest-reg %d)\n", m_objId, m_src, m_dest);
  pimResMgr* resMgr = device->getResMgr();
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_objId);
  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];
    PimCoreId coreId = srcRegion.getCoreId();
    for (unsigned j = 0; j < device->getNumCols(); j++) {
      bool src = device->getCore(coreId).getRowReg(m_src)[j];
      device->getCore(coreId).getRowReg(m_dest)[j] = !src;
    }
  }
  return true;
}

//! @brief  Pim CMD: BitSIMD-V: row reg dest = src1 & src2
bool
pimCmdRRegAnd::execute(pimDevice* device)
{
  std::printf("PIM-Info: BitSIMD-V RRegAnd (obj-id %d src1-reg %d src2-reg %d dest-reg %d)\n",
              m_objId, m_src1, m_src2, m_dest);
  pimResMgr* resMgr = device->getResMgr();
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_objId);
  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];
    PimCoreId coreId = srcRegion.getCoreId();
    for (unsigned j = 0; j < device->getNumCols(); j++) {
      bool src1 = device->getCore(coreId).getRowReg(m_src1)[j];
      bool src2 = device->getCore(coreId).getRowReg(m_src2)[j];
      device->getCore(coreId).getRowReg(m_dest)[j] = (src1 & src2);
    }
  }
  return true;
}

//! @brief  Pim CMD: BitSIMD-V: row reg dest = src1 | src2
bool
pimCmdRRegOr::execute(pimDevice* device)
{
  std::printf("PIM-Info: BitSIMD-V RRegOr (obj-id %d src1-reg %d src2-reg %d dest-reg %d)\n",
              m_objId, m_src1, m_src2, m_dest);
  pimResMgr* resMgr = device->getResMgr();
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_objId);
  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];
    PimCoreId coreId = srcRegion.getCoreId();
    for (unsigned j = 0; j < device->getNumCols(); j++) {
      bool src1 = device->getCore(coreId).getRowReg(m_src1)[j];
      bool src2 = device->getCore(coreId).getRowReg(m_src2)[j];
      device->getCore(coreId).getRowReg(m_dest)[j] = src1 | src2;
    }
  }
  return true;
}

//! @brief  Pim CMD: BitSIMD-V: row reg dest = src1 nand src2
bool
pimCmdRRegNand::execute(pimDevice* device)
{
  std::printf("PIM-Info: BitSIMD-V RRegNand (obj-id %d src1-reg %d src2-reg %d dest-reg %d)\n",
              m_objId, m_src1, m_src2, m_dest);
  pimResMgr* resMgr = device->getResMgr();
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_objId);
  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];
    PimCoreId coreId = srcRegion.getCoreId();
    for (unsigned j = 0; j < device->getNumCols(); j++) {
      bool src1 = device->getCore(coreId).getRowReg(m_src1)[j];
      bool src2 = device->getCore(coreId).getRowReg(m_src2)[j];
      device->getCore(coreId).getRowReg(m_dest)[j] = !(src1 & src2);
    }
  }
  return true;
}

//! @brief  Pim CMD: BitSIMD-V: row reg dest = src1 nor src2
bool
pimCmdRRegNor::execute(pimDevice* device)
{
  std::printf("PIM-Info: BitSIMD-V RRegNor (obj-id %d src1-reg %d src2-reg %d dest-reg %d)\n",
              m_objId, m_src1, m_src2, m_dest);
  pimResMgr* resMgr = device->getResMgr();
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_objId);
  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];
    PimCoreId coreId = srcRegion.getCoreId();
    for (unsigned j = 0; j < device->getNumCols(); j++) {
      bool src1 = device->getCore(coreId).getRowReg(m_src1)[j];
      bool src2 = device->getCore(coreId).getRowReg(m_src2)[j];
      device->getCore(coreId).getRowReg(m_dest)[j] = !(src1 | src2);
    }
  }
  return true;
}

//! @brief  Pim CMD: BitSIMD-V: row reg dest = src1 xor src2
bool
pimCmdRRegXor::execute(pimDevice* device)
{
  std::printf("PIM-Info: BitSIMD-V RRegXor (obj-id %d src1-reg %d src2-reg %d dest-reg %d)\n",
              m_objId, m_src1, m_src2, m_dest);
  pimResMgr* resMgr = device->getResMgr();
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_objId);
  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];
    PimCoreId coreId = srcRegion.getCoreId();
    for (unsigned j = 0; j < device->getNumCols(); j++) {
      bool src1 = device->getCore(coreId).getRowReg(m_src1)[j];
      bool src2 = device->getCore(coreId).getRowReg(m_src2)[j];
      device->getCore(coreId).getRowReg(m_dest)[j] = src1 ^ src2;
    }
  }
  return true;
}

//! @brief  Pim CMD: BitSIMD-V: row reg dest = src1 xnor src2
bool
pimCmdRRegXnor::execute(pimDevice* device)
{
  std::printf("PIM-Info: BitSIMD-V RRegXnor (obj-id %d src1-reg %d src2-reg %d dest-reg %d)\n",
              m_objId, m_src1, m_src2, m_dest);
  pimResMgr* resMgr = device->getResMgr();
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_objId);
  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];
    PimCoreId coreId = srcRegion.getCoreId();
    for (unsigned j = 0; j < device->getNumCols(); j++) {
      bool src1 = device->getCore(coreId).getRowReg(m_src1)[j];
      bool src2 = device->getCore(coreId).getRowReg(m_src2)[j];
      device->getCore(coreId).getRowReg(m_dest)[j] = !(src1 ^ src2);
    }
  }
  return true;
}

//! @brief  Pim CMD: BitSIMD-V: row reg dest = maj(src1, src2, src3)
bool
pimCmdRRegMaj::execute(pimDevice* device)
{
  std::printf("PIM-Info: BitSIMD-V RRegMaj (obj-id %d src1-reg %d src2-reg %d src3-reg %d dest-reg %d)\n",
              m_objId, m_src1, m_src2, m_src3, m_dest);
  pimResMgr* resMgr = device->getResMgr();
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_objId);
  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];
    PimCoreId coreId = srcRegion.getCoreId();
    for (unsigned j = 0; j < device->getNumCols(); j++) {
      bool src1 = device->getCore(coreId).getRowReg(m_src1)[j];
      bool src2 = device->getCore(coreId).getRowReg(m_src2)[j];
      bool src3 = device->getCore(coreId).getRowReg(m_src3)[j];
      device->getCore(coreId).getRowReg(m_dest)[j] = ((src1 & src2) || (src1 & src3) || (src2 & src3));
    }
  }
  return true;
}

//! @brief  Pim CMD: BitSIMD-V: row reg dest = cond ? src1 : src2;
bool
pimCmdRRegSel::execute(pimDevice* device)
{
  std::printf("PIM-Info: BitSIMD-V RRegSel (obj-id %d cond-reg %d src1-reg %d src2-reg %d dest-reg %d)\n",
              m_objId, m_cond, m_src1, m_src2, m_dest);
  pimResMgr* resMgr = device->getResMgr();
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_objId);
  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];
    PimCoreId coreId = srcRegion.getCoreId();
    for (unsigned j = 0; j < device->getNumCols(); j++) {
      bool cond = device->getCore(coreId).getRowReg(m_cond)[j];
      bool src1 = device->getCore(coreId).getRowReg(m_src1)[j];
      bool src2 = device->getCore(coreId).getRowReg(m_src2)[j];
      device->getCore(coreId).getRowReg(m_dest)[j] = (cond ? src1 : src2);
    }
  }
  return true;
}

//! @brief  Pim CMD: BitSIMD-V: row reg rotate right by one step
bool
pimCmdRRegRotateR::execute(pimDevice* device)
{
  std::printf("PIM-Info: BitSIMD-V RRegRotateR (obj-id %d src-reg %d)\n", m_objId, m_dest);
  pimResMgr* resMgr = device->getResMgr();
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_objId);
  // rotate
  bool prevVal = 0;
  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];
    PimCoreId coreId = srcRegion.getCoreId();
    for (unsigned j = 0; j < srcRegion.getNumAllocCols(); ++j) {
      unsigned colIdx = srcRegion.getColIdx() + j;
      bool tmp = device->getCore(coreId).getRowReg(m_dest)[colIdx];
      device->getCore(coreId).getRowReg(m_dest)[colIdx] = prevVal;
      prevVal = tmp;
    }
  }
  // write the last val to the first place
  const pimRegion& firstRegion = objSrc.getRegions().front();
  PimCoreId firstCoreId = firstRegion.getCoreId();
  unsigned firstColIdx = firstRegion.getColIdx();
  device->getCore(firstCoreId).getRowReg(m_dest)[firstColIdx] = prevVal;
  return true;
}

//! @brief  Pim CMD: BitSIMD-V: row reg rotate left by one step
bool
pimCmdRRegRotateL::execute(pimDevice* device)
{
  std::printf("PIM-Info: BitSIMD-V RRegRotateL (obj-id %d src-reg %d)\n", m_objId, m_dest);
  pimResMgr* resMgr = device->getResMgr();
  const pimObjInfo& objSrc = resMgr->getObjInfo(m_objId);
  // rotate
  bool prevVal = 0;
  for (unsigned i = objSrc.getRegions().size(); i > 0; --i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i - 1];
    PimCoreId coreId = srcRegion.getCoreId();
    for (unsigned j = srcRegion.getNumAllocCols(); j > 0; --j) {
      unsigned colIdx = srcRegion.getColIdx() + j - 1;
      bool tmp = device->getCore(coreId).getRowReg(m_dest)[colIdx];
      device->getCore(coreId).getRowReg(m_dest)[colIdx] = prevVal;
      prevVal = tmp;
    }
  }
  // write the first val to the last place
  const pimRegion& lastRegion = objSrc.getRegions().back();
  PimCoreId lastCoreId = lastRegion.getCoreId();
  unsigned lastColIdx = lastRegion.getColIdx() + lastRegion.getNumAllocCols() - 1;
  device->getCore(lastCoreId).getRowReg(m_dest)[lastColIdx] = prevVal;
  return true;
}

