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

//! @brief  PIM CMD: int32 add v-layout
bool
pimCmdInt32AddV::execute(pimDevice* device)
{
  std::printf("PIM-Info: Int32AddV (obj id %d + %d -> %d)\n", m_src1, m_src2, m_dest);

  pimResMgr* resMgr = device->getResMgr();
  if (!isVAligned(m_src1, m_src2, resMgr) || !isVAligned(m_src1, m_dest, resMgr)) {
    return false;
  }

  const pimObjInfo& objSrc1 = resMgr->getObjInfo(m_src1);
  const pimObjInfo& objSrc2 = resMgr->getObjInfo(m_src2);
  const pimObjInfo& objDest = resMgr->getObjInfo(m_dest);

  for (unsigned i = 0; i < objSrc1.getRegions().size(); ++i) {
    const pimRegion& src1Region = objSrc1.getRegions()[i];
    const pimRegion& src2Region = objSrc2.getRegions()[i];
    const pimRegion& destRegion = objDest.getRegions()[i];

    if (src1Region.getNumAllocRows() != 32 || src2Region.getNumAllocRows() != 32 || destRegion.getNumAllocRows() != 32) {
      std::printf("PIM-Error: Operands %d, %d and %d are not all 32-bit v-layout\n", m_src1, m_src2, m_dest);
      return false;
    }

    PimCoreId coreId = src1Region.getCoreId();

    // perform the computation
    unsigned colIdx = src1Region.getColIdx();
    unsigned numAllocCols = src1Region.getNumAllocCols();
    for (unsigned j = 0; j < numAllocCols; ++j) {
      int operand1 = static_cast<int>(device->getCore(coreId).getB32V(src1Region.getRowIdx(), colIdx + j));
      int operand2 = static_cast<int>(device->getCore(coreId).getB32V(src2Region.getRowIdx(), colIdx + j));
      int result = operand1 + operand2;
      device->getCore(coreId).setB32V(destRegion.getRowIdx(), colIdx + j, static_cast<unsigned>(result));
    }
  }

  return true;
}

//! @brief  PIM CMD: int32 abs v-layout
bool
pimCmdInt32AbsV::execute(pimDevice* device)
{
  std::printf("PIM-Info: Int32AbsV (obj id %d -> %d)\n", m_src, m_dest);

  pimResMgr* resMgr = device->getResMgr();
  if (!isVAligned(m_src, m_dest, resMgr)) {
    return false;
  }

  const pimObjInfo& objSrc = resMgr->getObjInfo(m_src);
  const pimObjInfo& objDest = resMgr->getObjInfo(m_dest);

  for (unsigned i = 0; i < objSrc.getRegions().size(); ++i) {
    const pimRegion& srcRegion = objSrc.getRegions()[i];
    const pimRegion& destRegion = objDest.getRegions()[i];

    if (srcRegion.getNumAllocRows() != 32 || destRegion.getNumAllocRows() != 32) {
      std::printf("PIM-Error: Operands %d and %d are not all 32-bit v-layout\n", m_src, m_dest);
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

//! @brief  PIM CMD: int32 mul v-layout
bool
pimCmdInt32MulV::execute(pimDevice* device)
{
  std::printf("PIM-Info: Int32MulV (obj id %d * %d -> %d)\n", m_src1, m_src2, m_dest);

  pimResMgr* resMgr = device->getResMgr();
  if (!isVAligned(m_src1, m_src2, resMgr) || !isVAligned(m_src1, m_dest, resMgr)) {
    return false;
  }

  const pimObjInfo& objSrc1 = resMgr->getObjInfo(m_src1);
  const pimObjInfo& objSrc2 = resMgr->getObjInfo(m_src2);
  const pimObjInfo& objDest = resMgr->getObjInfo(m_dest);

  for (unsigned i = 0; i < objSrc1.getRegions().size(); ++i) {
    const pimRegion& src1Region = objSrc1.getRegions()[i];
    const pimRegion& src2Region = objSrc2.getRegions()[i];
    const pimRegion& destRegion = objDest.getRegions()[i];

    if (src1Region.getNumAllocRows() != 32 || src2Region.getNumAllocRows() != 32 || destRegion.getNumAllocRows() != 32) {
      std::printf("PIM-Error: Operands %d, %d and %d are not all 32-bit v-layout\n", m_src1, m_src2, m_dest);
      return false;
    }

    PimCoreId coreId = src1Region.getCoreId();

    // perform the computation
    unsigned colIdx = src1Region.getColIdx();
    unsigned numAllocCols = src1Region.getNumAllocCols();
    for (unsigned j = 0; j < numAllocCols; ++j) {
      int operand1 = static_cast<int>(device->getCore(coreId).getB32V(src1Region.getRowIdx(), colIdx + j));
      int operand2 = static_cast<int>(device->getCore(coreId).getB32V(src2Region.getRowIdx(), colIdx + j));
      int result = operand1 * operand2;
      device->getCore(coreId).setB32V(destRegion.getRowIdx(), colIdx + j, static_cast<unsigned>(result));
    }
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

