// File: pimSim.cpp
// PIM Functional Simulator - PIM Simulator Main Entry
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "pimSim.h"
#include "pimCmd.h"
#include "pimParamsDram.h"
#include "pimParamsPerf.h"
#include "pimStats.h"
#include "pimUtils.h"
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

//! @brief  Destroy the pimSim singleton
void
pimSim::destroy()
{
  if (s_instance) {
    delete s_instance;
    s_instance = nullptr;
  }
}

//! @brief  pimSim ctor
pimSim::pimSim()
{
  m_paramsDram = new pimParamsDram();
  m_paramsPerf = new pimParamsPerf(m_paramsDram);
  m_statsMgr = new pimStatsMgr(m_paramsDram, m_paramsPerf);
}

//! @brief  pimSim dtor
pimSim::~pimSim()
{
  delete m_statsMgr;
  delete m_paramsDram;
  delete m_paramsPerf;
}

//! @brief  Create a PIM device
bool
pimSim::createDevice(PimDeviceEnum deviceType, unsigned numBanks, unsigned numSubarrayPerBank, unsigned numRows, unsigned numCols)
{
  pimPerfMon perfMon("createDevice");
  if (m_device != nullptr) {
    std::printf("PIM-Error: PIM device is already created\n");
    return false;
  }

  m_paramsPerf->setDevice(deviceType);
  std::printf("PIM-Info: Current Device = %s, Simulation Target = %s\n",
              pimUtils::pimDeviceEnumToStr(m_paramsPerf->getCurDevice()).c_str(),
              pimUtils::pimDeviceEnumToStr(m_paramsPerf->getSimTarget()).c_str());

  m_device = new pimDevice();
  m_device->init(deviceType, numBanks, numSubarrayPerBank, numRows, numCols);
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
  pimPerfMon perfMon("createDeviceFromConfig");
  if (m_device) {
    std::printf("PIM-Error: PIM Device is already created\n");
    return false;
  }

  m_paramsPerf->setDevice(deviceType);
  std::printf("PIM-Info: Current Device = %s, Simulation Target = %s\n",
              pimUtils::pimDeviceEnumToStr(m_paramsPerf->getCurDevice()).c_str(),
              pimUtils::pimDeviceEnumToStr(m_paramsPerf->getSimTarget()).c_str());

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
  m_paramsPerf->setDevice(PIM_DEVICE_NONE);
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

//! @brief  Get device type
PimDeviceEnum
pimSim::getDeviceType() const
{
  return m_paramsPerf->getCurDevice();
}

//! @brief  Get simulation target device
PimDeviceEnum
pimSim::getSimTarget() const
{
  return m_paramsPerf->getSimTarget();
}

//! @brief  Get number of PIM cores
unsigned
pimSim::getNumCores() const
{
  if (m_device && m_device->isValid()) {
    return m_device->getNumCores();
  }
  return 0;
}

//! @brief  Get number of rows per PIM core
unsigned
pimSim::getNumRows() const
{
  if (m_device && m_device->isValid()) {
    return m_device->getNumRows();
  }
  return 0;
}

//! @brief  Get number of columns per PIM core
unsigned
pimSim::getNumCols() const
{
  if (m_device && m_device->isValid()) {
    return m_device->getNumCols();
  }
  return 0;
}

//! @brief  Show PIM command stats
void
pimSim::showStats() const
{
  m_statsMgr->showStats();
}

//! @brief  Allocate a PIM object
PimObjId
pimSim::pimAlloc(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElement, PimDataType dataType)
{
  pimPerfMon perfMon("pimAlloc");
  if (!isValidDevice()) { return -1; }
  return m_device->pimAlloc(allocType, numElements, bitsPerElement, dataType);
}

//! @brief  Allocate a PIM object that is associated with a reference ojbect
PimObjId
pimSim::pimAllocAssociated(unsigned bitsPerElement, PimObjId ref, PimDataType dataType)
{
  pimPerfMon perfMon("pimAllocAssociated");
  if (!isValidDevice()) { return -1; }
  return m_device->pimAllocAssociated(bitsPerElement, ref, dataType);
}

// @brief  Free a PIM object
bool
pimSim::pimFree(PimObjId obj)
{
  pimPerfMon perfMon("pimFree");
  if (!isValidDevice()) { return false; }
  return m_device->pimFree(obj);
}

// @brief  Copy data from main memory to PIM device
bool
pimSim::pimCopyMainToDevice(void* src, PimObjId dest)
{
  pimPerfMon perfMon("pimCopyMainToDevice");
  if (!isValidDevice()) { return false; }
  return m_device->pimCopyMainToDevice(src, dest);
}

// @brief  Copy data from PIM device to main memory
bool
pimSim::pimCopyDeviceToMain(PimObjId src, void* dest)
{
  pimPerfMon perfMon("pimCopyDeviceToMain");
  if (!isValidDevice()) { return false; }
  return m_device->pimCopyDeviceToMain(src, dest);
}

// @brief  Copy data from main memory to PIM device with type
bool
pimSim::pimCopyMainToDeviceWithType(PimCopyEnum copyType, void* src, PimObjId dest)
{
  pimPerfMon perfMon("pimCopyMainToDevice");
  if (!isValidDevice()) { return false; }
  return m_device->pimCopyMainToDeviceWithType(copyType, src, dest);
}

// @brief  Copy data from PIM device to main memory with type
bool
pimSim::pimCopyDeviceToMainWithType(PimCopyEnum copyType, PimObjId src, void* dest)
{
  pimPerfMon perfMon("pimCopyDeviceToMain");
  if (!isValidDevice()) { return false; }
  return m_device->pimCopyDeviceToMainWithType(copyType, src, dest);
}

// @brief  Load vector with a scalar value
bool
pimSim::pimBroadcast(PimObjId dest, unsigned value)
{
  pimPerfMon perfMon("pimBroadcast");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdBroadcast>(PimCmdEnum::BROADCAST, dest, value);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: add
bool
pimSim::pimAdd(PimObjId src1, PimObjId src2, PimObjId dest)
{
  pimPerfMon perfMon("pimAdd");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc2>(PimCmdEnum::ADD, src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: sub
bool
pimSim::pimSub(PimObjId src1, PimObjId src2, PimObjId dest)
{
  pimPerfMon perfMon("pimSub");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc2>(PimCmdEnum::SUB, src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief PIM OP: div
bool
pimSim::pimDiv(PimObjId src1, PimObjId src2, PimObjId dest)
{
  pimPerfMon perfMon("pimDiv");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc2>(PimCmdEnum::DIV, src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: abs v-layout
bool
pimSim::pimAbs(PimObjId src, PimObjId dest)
{
  pimPerfMon perfMon("pimAbs");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc1>(PimCmdEnum::ABS, src, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: mul
bool
pimSim::pimMul(PimObjId src1, PimObjId src2, PimObjId dest)
{
  pimPerfMon perfMon("pimMul");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc2>(PimCmdEnum::MUL, src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: and
bool
pimSim::pimAnd(PimObjId src1, PimObjId src2, PimObjId dest)
{
  pimPerfMon perfMon("pimAnd");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc2>(PimCmdEnum::AND, src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: or
bool
pimSim::pimOr(PimObjId src1, PimObjId src2, PimObjId dest)
{
  pimPerfMon perfMon("pimOr");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc2>(PimCmdEnum::OR, src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: xor
bool
pimSim::pimXor(PimObjId src1, PimObjId src2, PimObjId dest)
{
  pimPerfMon perfMon("pimXor");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc2>(PimCmdEnum::XOR, src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: xnor
bool
pimSim::pimXnor(PimObjId src1, PimObjId src2, PimObjId dest)
{
  pimPerfMon perfMon("pimXnor");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc2>(PimCmdEnum::XNOR, src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: gt
bool
pimSim::pimGT(PimObjId src1, PimObjId src2, PimObjId dest)
{
  pimPerfMon perfMon("pimGT");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc2>(PimCmdEnum::GT, src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: lt
bool
pimSim::pimLT(PimObjId src1, PimObjId src2, PimObjId dest)
{
  pimPerfMon perfMon("pimLT");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc2>(PimCmdEnum::LT, src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: eq
bool
pimSim::pimEQ(PimObjId src1, PimObjId src2, PimObjId dest)
{
  pimPerfMon perfMon("pimEQ");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc2>(PimCmdEnum::EQ, src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: min
bool
pimSim::pimMin(PimObjId src1, PimObjId src2, PimObjId dest)
{
  pimPerfMon perfMon("pimMin");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc2>(PimCmdEnum::MIN, src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: max
bool
pimSim::pimMax(PimObjId src1, PimObjId src2, PimObjId dest)
{
  pimPerfMon perfMon("pimMax");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc2>(PimCmdEnum::MAX, src1, src2, dest);
  return m_device->executeCmd(std::move(cmd));
}

// @brief  PIM OP: popcount
bool
pimSim::pimPopCount(PimObjId src, PimObjId dest)
{
  pimPerfMon perfMon("pimPopCount");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc1>(PimCmdEnum::POPCOUNT, src, dest);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimRedSum(PimObjId src, int* sum)
{
  pimPerfMon perfMon("pimRedSum");
  if (!isValidDevice()) { return false; }
  *sum = 0;
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRedSum>(PimCmdEnum::REDSUM, src, sum);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimRedSumRanged(PimObjId src, unsigned idxBegin, unsigned idxEnd, int* sum)
{
  pimPerfMon perfMon("pimRedSumRanged");
  if (!isValidDevice()) { return false; }
  *sum = 0;
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRedSum>(PimCmdEnum::REDSUM_RANGE, src, sum, idxBegin, idxEnd);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimRotateR(PimObjId src)
{
  pimPerfMon perfMon("pimRotateR");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRotate>(PimCmdEnum::ROTATE_R, src);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimRotateL(PimObjId src)
{
  pimPerfMon perfMon("pimRotateL");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRotate>(PimCmdEnum::ROTATE_L, src);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimShiftR(PimObjId src)
{
  pimPerfMon perfMon("pimShiftR");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRotate>(PimCmdEnum::SHIFT_R, src);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimShiftL(PimObjId src)
{
  pimPerfMon perfMon("pimShiftL");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRotate>(PimCmdEnum::SHIFT_L, src);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpReadRowToSa(PimObjId objId, unsigned ofst)
{
  pimPerfMon perfMon("pimOpReadRowToSa");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdReadRowToSa>(PimCmdEnum::ROW_R, objId, ofst);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpWriteSaToRow(PimObjId objId, unsigned ofst)
{
  pimPerfMon perfMon("pimOpWriteSaToRow");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdWriteSaToRow>(PimCmdEnum::ROW_W, objId, ofst);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpTRA(PimObjId src1, unsigned ofst1, PimObjId src2, unsigned ofst2, PimObjId src3, unsigned ofst3)
{
  pimPerfMon perfMon("pimOpTRA");
  return false;
}

bool
pimSim::pimOpMove(PimObjId objId, PimRowReg src, PimRowReg dest)
{
  pimPerfMon perfMon("pimOpMove");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegOp>(PimCmdEnum::RREG_MOV, objId, dest, src);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpSet(PimObjId objId, PimRowReg dest, bool val)
{
  pimPerfMon perfMon("pimOpSet");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegOp>(PimCmdEnum::RREG_SET, objId, dest, val);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpNot(PimObjId objId, PimRowReg src, PimRowReg dest)
{
  pimPerfMon perfMon("pimOpNot");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegOp>(PimCmdEnum::RREG_NOT, objId, dest, src);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpAnd(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  pimPerfMon perfMon("pimOpAnd");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegOp>(PimCmdEnum::RREG_AND, objId, dest, src1, src2);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpOr(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  pimPerfMon perfMon("pimOpOr");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegOp>(PimCmdEnum::RREG_OR, objId, dest, src1, src2);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpNand(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  pimPerfMon perfMon("pimOpNand");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegOp>(PimCmdEnum::RREG_NAND, objId, dest, src1, src2);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpNor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  pimPerfMon perfMon("pimOpNor");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegOp>(PimCmdEnum::RREG_NOR, objId, dest, src1, src2);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpXor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  pimPerfMon perfMon("pimOpXor");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegOp>(PimCmdEnum::RREG_XOR, objId, dest, src1, src2);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpXnor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  pimPerfMon perfMon("pimOpXnor");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegOp>(PimCmdEnum::RREG_XNOR, objId, dest, src1, src2);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpMaj(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg src3, PimRowReg dest)
{
  pimPerfMon perfMon("pimOpMaj");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegOp>(PimCmdEnum::RREG_MAJ, objId, dest, src1, src2, src3);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpSel(PimObjId objId, PimRowReg cond, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  pimPerfMon perfMon("pimOpSel");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegOp>(PimCmdEnum::RREG_SEL, objId, dest, cond, src1, src2);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpRotateRH(PimObjId objId, PimRowReg src)
{
  pimPerfMon perfMon("pimOpRotateRH");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegRotate>(PimCmdEnum::RREG_ROTATE_R, objId, src);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpRotateLH(PimObjId objId, PimRowReg src)
{
  pimPerfMon perfMon("pimOpRotateLH");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRRegRotate>(PimCmdEnum::RREG_ROTATE_L, objId, src);
  return m_device->executeCmd(std::move(cmd));
}

