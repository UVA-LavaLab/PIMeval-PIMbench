// File: pimSim.cpp
// PIM Functional Simulator - PIM Simulator Main Entry
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "pimSim.h"
#include "pimCmd.h"
#include <cstdio>
#include <memory>


// The pimSim singleton
pimSim* pimSim::s_instance = nullptr;


//! @brief  Get or create the pimSim singleton
pimSim*
pimSim::get()
{
  if (!s_instance) {
    s_instance = new pimSim();
  }
  return s_instance;
}

//! @brief  Create a PIM device
bool
pimSim::createDevice(PimDeviceEnum deviceType, unsigned numCores, unsigned numRows, unsigned numCols)
{
  if (m_device) {
    std::printf("PIM-Error: PIM device is already created\n");
    return false;
  }
  m_device = new pimDevice();
  m_device->init(deviceType, numCores, numRows, numCols);
  if (!m_device->isValid()) {
    delete m_device;
    std::printf("PIM-Error: Failed to create PIM device of type %d\n", static_cast<int>(deviceType));
    return false;
  }
  return true;
}

//! @brief  Create a PIM device from a config file
bool
pimSim::createDeviceFromConfig(PimDeviceEnum deviceType, const char* configFileName)
{
  if (m_device) {
    std::printf("PIM-Error: PIM Device is already created\n");
    return false;
  }
  m_device = new pimDevice();
  m_device->init(deviceType, configFileName);
  if (!m_device->isValid()) {
    delete m_device;
    std::printf("PIM-Error: Failed to create PIM device of type %d\n", static_cast<int>(deviceType));
    return false;
  }
  return true;
}

//! @brief  Delete PIM device
bool
pimSim::deleteDevice()
{
  if (!m_device) {
    std::printf("PIM-Error: No PIM device to delete\n");
    return false;
  }
  delete m_device;
  m_device = nullptr;
  return true;
}

//! @brief  Check if device is valid
bool
pimSim::isValidDevice(bool showMsg) const
{
  bool isValid = m_device && m_device->isValid();
  if (!isValid && showMsg) {
    std::printf("PIM-Error: Invalid PIM device\n");
  }
  return isValid;
}

//! @brief  Show PIM command stats
void
pimSim::showStats() const
{
  if (m_device && m_device->isValid()) {
    m_device->showStats();
  }
}

//! @brief  Allocate a PIM object
PimObjId
pimSim::pimAlloc(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElement, PimDataType dataType)
{
  if (!isValidDevice()) { return -1; }
  return m_device->pimAlloc(allocType, numElements, bitsPerElement, dataType);
}

//! @brief  Allocate a PIM object that is associated with a reference ojbect
PimObjId
pimSim::pimAllocAssociated(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElement, PimObjId ref, PimDataType dataType)
{
  if (!isValidDevice()) { return -1; }
  return m_device->pimAllocAssociated(allocType, numElements, bitsPerElement, ref, dataType);
}

// @brief  Free a PIM object
bool
pimSim::pimFree(PimObjId obj)
{
  if (!isValidDevice()) { return false; }
  return m_device->pimFree(obj);
}

// @brief  Copy data from main memory to PIM device
bool
pimSim::pimCopyMainToDevice(PimCopyEnum copyType, void* src, PimObjId dest)
{
  if (!isValidDevice()) { return false; }
  return m_device->pimCopyMainToDevice(copyType, src, dest);
}

// @brief  Copy data from PIM device to main memory
bool
pimSim::pimCopyDeviceToMain(PimCopyEnum copyType, PimObjId src, void* dest)
{
  if (!isValidDevice()) { return false; }
  return m_device->pimCopyDeviceToMain(copyType, src, dest);
}

// @brief  Load vector with a scalar value
bool
pimSim::pimBroadCast(PimCopyEnum copyType, PimObjId src, unsigned value)
{
  if (!isValidDevice()) { return false; }
  return m_device->pimBroadCast(copyType, src, value);
}

// @brief  PIM OP: add
bool
pimSim::pimAdd(PimObjId src1, PimObjId src2, PimObjId dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdAddV>(src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: sub
bool
pimSim::pimSub(PimObjId src1, PimObjId src2, PimObjId dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdSubV>(src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief PIM OP: div
bool
pimSim::pimDiv(PimObjId src1, PimObjId src2, PimObjId dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdDivV>(src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: abs v-layout
bool
pimSim::pimAbs(PimObjId src, PimObjId dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdAbsV>(src, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: mul
bool
pimSim::pimMul(PimObjId src1, PimObjId src2, PimObjId dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdMulV>(src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: and
bool
pimSim::pimAnd(PimObjId src1, PimObjId src2, PimObjId dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdAndV>(src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: or
bool
pimSim::pimOr(PimObjId src1, PimObjId src2, PimObjId dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdOrV>(src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: xor
bool
pimSim::pimXor(PimObjId src1, PimObjId src2, PimObjId dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdXorV>(src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: gt
bool
pimSim::pimGT(PimObjId src1, PimObjId src2, PimObjId dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdGTV>(src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: lt
bool
pimSim::pimLT(PimObjId src1, PimObjId src2, PimObjId dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdLTV>(src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: eq
bool
pimSim::pimEQ(PimObjId src1, PimObjId src2, PimObjId dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdEQV>(src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: min
bool
pimSim::pimMin(PimObjId src1, PimObjId src2, PimObjId dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdMinV>(src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: max
bool
pimSim::pimMax(PimObjId src1, PimObjId src2, PimObjId dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdMaxV>(src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: popcount
bool
pimSim::pimPopCount(PimObjId src, PimObjId dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdPopCountV>(src, dest);
  return m_device->executeCmd(std::move(cmd));
}

int
pimSim::pimRedSum(PimObjId src)
{
  if (!isValidDevice()) { return false; }
  int result = 0;
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRedSum>(src, result);
  m_device->executeCmd(std::move(cmd));
  return result;
}

int
pimSim::pimRedSumRanged(PimObjId src, unsigned idxBegin, unsigned idxEnd)
{
  if (!isValidDevice()) { return false; }
  int result = 0;
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRedSumRanged>(src, result, idxBegin, idxEnd);
  m_device->executeCmd(std::move(cmd));
  return result;
}

bool
pimSim::pimRotateR(PimObjId src)
{
  return false;
}

bool
pimSim::pimRotateL(PimObjId src)
{
  return false;
}

bool
pimSim::pimOpReadRowToSa(PimObjId objId, unsigned ofst)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdReadRowToSa>(objId, ofst);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpWriteSaToRow(PimObjId objId, unsigned ofst)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdWriteSaToRow>(objId, ofst);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpTRA(PimObjId src1, unsigned ofst1, PimObjId src2, unsigned ofst2, PimObjId src3, unsigned ofst3)
{
  return false;
}

bool
pimSim::pimOpMove(PimObjId objId, PimRowReg src, PimRowReg dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegMove>(objId, src, dest);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpSet(PimObjId objId, PimRowReg src, bool val)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegSet>(objId, src, val);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpNot(PimObjId objId, PimRowReg src, PimRowReg dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegNot>(objId, src, dest);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpAnd(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegAnd>(objId, src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpOr(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegOr>(objId, src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpNand(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegNand>(objId, src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpNor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegNor>(objId, src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpXor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegXor>(objId, src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpXnor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegXnor>(objId, src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpMaj(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg src3, PimRowReg dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegMaj>(objId, src1, src2, src3, dest);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpSel(PimObjId objId, PimRowReg cond, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegSel>(objId, cond, src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpRotateRH(PimObjId objId, PimRowReg src)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegRotateR>(objId, src);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpRotateLH(PimObjId objId, PimRowReg src)
{
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegRotateL>(objId, src);
  return m_device->executeCmd(std::move(cmd));
}

