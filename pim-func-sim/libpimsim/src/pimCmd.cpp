// File: pimCmd.cpp
// PIM Functional Simulator - PIM Commands
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "pimCmd.h"
#include "pimDevice.h"
#include "pimCore.h"
#include "pimResMgr.h"
#include <cstdio>


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

//! @brief  PIM OP: Add int32 v-layout
bool
pimCmdAddInt32V::execute(pimDevice* device)
{
  std::printf("PIM-Info: AddInt32V (obj id %d + %d -> %d)\n", m_src1, m_src2, m_dest);

  pimResMgr* resMgr = device->getResMgr();
  if (!isVAligned(m_src1, m_src2, resMgr) || !isVAligned(m_src1, m_src2, resMgr)) {
    return false;
  }

  const pimObjInfo& objSrc1 = resMgr->getObjInfo(m_src1);
  const pimObjInfo& objSrc2 = resMgr->getObjInfo(m_src2);
  const pimObjInfo& objDest = resMgr->getObjInfo(m_dest);

  for (unsigned i = 0; i < objSrc1.getRegions().size(); ++i) {
    const pimRegion& regSrc1 = objSrc1.getRegions()[i];
    const pimRegion& regSrc2 = objSrc2.getRegions()[i];
    const pimRegion& regDest = objDest.getRegions()[i];

    if (regSrc1.getNumAllocRows() != 32 || regSrc2.getNumAllocRows() != 32 || regDest.getNumAllocRows() != 32) {
      std::printf("PIM-Error: Operands %d, %d and %d are not all 32-bit v-layout\n", m_src1, m_src2, m_dest);
      return false;
    }

    PimCoreId coreId = regSrc1.getCoreId();

    // perform the computation
    unsigned colIdx = regSrc1.getColIdx();
    unsigned numAllocCols = regSrc1.getNumAllocCols();
    for (unsigned j = 0; j < numAllocCols; ++j) {
      int operand1 = 0;
      for (int k = 31; k >= 0; --k) {
        bool val = device->getCore(coreId).getBit(regSrc1.getRowIdx() + k, colIdx + j);
        operand1 = (operand1 << 1) | val;
      }
      int operand2 = 0;
      for (int k = 31; k >= 0; --k) {
        bool val = device->getCore(coreId).getBit(regSrc2.getRowIdx() + k, colIdx + j);
        operand2 = (operand2 << 1) | val;
      }
      int result = operand1 + operand2;
      for (int k = 0; k < 32; ++k) {
        bool val = result & 1;
        device->getCore(coreId).setBit(regDest.getRowIdx() + k, colIdx + j, val);
        result = result >> 1;
      }
    }
  }

  return true;
}

