// File: pimSim.cpp
// PIMeval Simulator - PIM Simulator Main Entry
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimSim.h"
#include "pimCmd.h"
#include "pimParamsDram.h"
#include "pimPerfEnergyModels.h"
#include "pimStats.h"
#include "pimUtils.h"
#include <cstdio>
#include <memory>
#include <algorithm>
#include <cctype>
#include <locale>
#include <stdexcept>
#include <sstream>
#include <filesystem>
#include <string>

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
}

//! @brief  pimSim dtor
pimSim::~pimSim()
{
  uninit();
}

//! @brief  Initialize pimSim member classes from the config file
bool
pimSim::init(const std::string& simConfigFileConetnt)
{
  if (!m_initCalled) {
    if (!simConfigFileConetnt.empty()) {
      bool success = parseConfigFromFile(simConfigFileConetnt);
      if (!success) {
        return false;
      }

      if (m_memConfigFileName.empty()) {
        std::string memConfigFileFullPath = m_configFilesPath + m_memConfigFileName;
        std::string fileContent;
        success = pimUtils::readFileContent(memConfigFileFullPath.c_str(), fileContent);
        if (!success) {
          return false;
        }
        m_paramsDram = new pimParamsDram(fileContent);
      } else {
        m_paramsDram = new pimParamsDram();
      }

      m_paramsPerf = new pimParamsPerf(m_paramsDram);
      m_statsMgr = new pimStatsMgr(m_paramsDram, m_paramsPerf);
      m_initCalled = true;
    } else {
      m_paramsDram = new pimParamsDram();
      m_paramsPerf = new pimParamsPerf(m_paramsDram);
      m_statsMgr = new pimStatsMgr(m_paramsDram, m_paramsPerf);
      m_initCalled = true;
    }
  }
  return true;
}

//! @brief  Uninitialize pimSim member claasses
void
pimSim::uninit()
{
  delete m_threadPool;
  delete m_statsMgr;
  delete m_paramsDram;
  delete m_paramsPerf;
}

//! @brief  Determine num threads and init thread pool
void
pimSim::initThreadPool(unsigned maxNumThreads)
{
  if (m_threadPool) {
    delete m_threadPool;
    m_threadPool = nullptr;
  }
  unsigned hwThreads = std::thread::hardware_concurrency();
  if (maxNumThreads == 0) {
    m_numThreads = hwThreads;
  } else {
    m_numThreads = std::min(maxNumThreads, hwThreads);
  }
  if (m_numThreads < 1) {
    m_numThreads = 1;
  }
  if (m_numThreads > 1) {
    m_threadPool = new pimUtils::threadPool(m_numThreads);
  }
}

//! @brief  Create a PIM device
bool
pimSim::createDevice(PimDeviceEnum deviceType, unsigned numRanks, unsigned numBankPerRank, unsigned numSubarrayPerBank, unsigned numRows, unsigned numCols)
{
  pimPerfMon perfMon("createDevice");
  if (m_device != nullptr) {
    std::printf("PIM-Error: PIM device is already created\n");
    return false;
  }
  bool success = init();
  if (!success) {
    std::printf("PIM-Error: Init failed\n");
    return false;
  }
  m_device = new pimDevice();
  m_device->init(deviceType, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
  if (!m_device->isValid()) {
    delete m_device;
    uninit();
    std::printf("PIM-Error: Failed to create PIM device of type %d\n", static_cast<int>(deviceType));
    return false;
  }
  unsigned maxNumThreads = 0; // use max hardware parallelism by default
  initThreadPool(maxNumThreads);
  return true;
}

//! @brief  Create a PIM device from a config file
bool
pimSim::createDeviceFromConfig(PimDeviceEnum deviceType, const char* configFileName)
{
  pimPerfMon perfMon("createDeviceFromConfig");
  bool success = false;
  std::string correctConfigFileName;
  if (!configFileName) {
    std::printf("PIM-Info: Null PIM device config file name. Read the config file name from environment variables %s and %s\n", pimUtils::envVarPimEvalConfigPath, pimUtils::envVarPimEvalConfigSim);

    // Read environment variable for the config file path
    std::string pimEvalConfigPath;
    if (!pimUtils::getEnvVar(pimUtils::envVarPimEvalConfigPath, pimEvalConfigPath)) {
      std::printf("PIM-Error: Could not read environment variable %s", pimUtils::envVarPimEvalConfigPath);
      return false;
    }

    // Read environment variable for the simulation config file name
    std::string pimEvalConfigSim;
    if (!pimUtils::getEnvVar(pimUtils::envVarPimEvalConfigSim, pimEvalConfigSim)) {
      std::printf("PIM-Error: Could not read environment variable %s", pimUtils::envVarPimEvalConfigSim);
      return false;
    }
    correctConfigFileName = pimEvalConfigPath + "/" + pimEvalConfigSim;
    std::printf("PIM-Info: Read config file from the environment variables is \"%s\".\n", correctConfigFileName.c_str());
  } else {
    correctConfigFileName = configFileName;
  }
  if (!std::filesystem::exists(correctConfigFileName)) {
    std::printf("PIM-Error: Config file not found.\n");
    return false;
  }

  std::string fileContent;
  success = pimUtils::readFileContent(correctConfigFileName.c_str(), fileContent);
  if (!success) {
    std::printf("PIM-Error: Failed to read config file %s\n", correctConfigFileName.c_str());
    return false;
  }

  m_configFilesPath = pimUtils::getDirectoryPath(correctConfigFileName);

  success = init(fileContent);
  if (!success) {
    std::printf("PIM-Error: Init failed\n");
    return false;
  }
  if (m_device) {
    std::printf("PIM-Error: PIM Device is already created\n");
    return false;
  }

  m_device = new pimDevice();
  m_device->init(deviceType, correctConfigFileName.c_str());
  if (!m_device->isValid()) {
    delete m_device;
    uninit();
    std::printf("PIM-Error: Failed to create PIM device of type %d\n", static_cast<int>(deviceType));
    return false;
  }
  unsigned maxNumThreads = m_numThreads;
  initThreadPool(maxNumThreads);
  return true;
}

bool
pimSim::getDeviceProperties(PimDeviceProperties* deviceProperties) {
  pimPerfMon perfMon("getDeviceProperties");
  if (!m_device) {
    std::printf("PIM-Error: No PIM device exists.\n");
    return false;
  }
  deviceProperties->deviceType = m_device->getDeviceType();
  deviceProperties->numRanks = m_device->getNumRanks();
  deviceProperties->numBankPerRank = m_device->getNumBankPerRank();
  deviceProperties->numSubarrayPerBank = m_device->getNumSubarrayPerBank();
  deviceProperties->numRowPerSubarray = m_device->getNumRowPerSubarray();
  deviceProperties->numColPerSubarray = m_device->getNumColPerSubarray();
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

//! @brief  Get device type
PimDeviceEnum
pimSim::getDeviceType() const
{
  if (m_device && m_device->isValid()) {
    return m_device->getDeviceType();
  }
  return PIM_DEVICE_NONE;
}

//! @brief  Get simulation target device
PimDeviceEnum
pimSim::getSimTarget() const
{
  if (m_device && m_device->isValid()) {
    return m_device->getSimTarget();
  }
  return PIM_DEVICE_NONE;
}

//! @brief  Get number of ranks
unsigned
pimSim::getNumRanks() const
{
  if (m_device && m_device->isValid()) {
    return m_device->getNumRanks();
  }
  return 0;
}

//! @brief  Get number of banks per rank
unsigned
pimSim::getNumBankPerRank() const
{
  if (m_device && m_device->isValid()) {
    return m_device->getNumBankPerRank();
  }
  return 0;
}

//! @brief  Get number of subarrays per bank
unsigned
pimSim::getNumSubarrayPerBank() const
{
  if (m_device && m_device->isValid()) {
    return m_device->getNumSubarrayPerBank();
  }
  return 0;
}

//! @brief  Get number of rows per subarray
unsigned
pimSim::getNumRowPerSubarray() const
{
  if (m_device && m_device->isValid()) {
    return m_device->getNumRowPerSubarray();
  }
  return 0;
}

//! @brief  Get number of cols per subarray
unsigned
pimSim::getNumColPerSubarray() const
{
  if (m_device && m_device->isValid()) {
    return m_device->getNumColPerSubarray();
  }
  return 0;
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

//! @brief  Reset PIM command stats
void
pimSim::resetStats() const
{
  m_statsMgr->resetStats();
}

//! @brief  Allocate a PIM object
PimObjId
pimSim::pimAlloc(PimAllocEnum allocType, uint64_t numElements, unsigned bitsPerElement, PimDataType dataType)
{
  pimPerfMon perfMon("pimAlloc");
  if (!isValidDevice()) { return -1; }
  return m_device->pimAlloc(allocType, numElements, bitsPerElement, dataType);
}

//! @brief  Allocate a PIM object that is associated with an existing ojbect
PimObjId
pimSim::pimAllocAssociated(unsigned bitsPerElement, PimObjId assocId, PimDataType dataType)
{
  pimPerfMon perfMon("pimAllocAssociated");
  if (!isValidDevice()) { return -1; }
  return m_device->pimAllocAssociated(bitsPerElement, assocId, dataType);
}

// @brief  Free a PIM object
bool
pimSim::pimFree(PimObjId obj)
{
  pimPerfMon perfMon("pimFree");
  if (!isValidDevice()) { return false; }
  return m_device->pimFree(obj);
}

//! @brief  Create an obj referencing to a range of an existing obj
PimObjId
pimSim::pimCreateRangedRef(PimObjId refId, uint64_t idxBegin, uint64_t idxEnd)
{
  pimPerfMon perfMon("pimCreateRangedRef");
  if (!isValidDevice()) { return -1; }
  return m_device->pimCreateRangedRef(refId, idxBegin, idxEnd);
}

//! @brief  Create an obj referencing to negation of an existing obj based on dual-contact memory cells
PimObjId
pimSim::pimCreateDualContactRef(PimObjId refId)
{
  pimPerfMon perfMon("pimCreateDualContactRef");
  if (!isValidDevice()) { return -1; }
  return m_device->pimCreateDualContactRef(refId);
}

// @brief  Copy data from main memory to PIM device within a range
bool
pimSim::pimCopyMainToDevice(void* src, PimObjId dest, uint64_t idxBegin, uint64_t idxEnd)
{
  pimPerfMon perfMon("pimCopyMainToDevice");
  if (!isValidDevice()) { return false; }
  return m_device->pimCopyMainToDevice(src, dest, idxBegin, idxEnd);
}

// @brief  Copy data from PIM device to main memory within a range
bool
pimSim::pimCopyDeviceToMain(PimObjId src, void* dest, uint64_t idxBegin, uint64_t idxEnd)
{
  pimPerfMon perfMon("pimCopyDeviceToMain");
  if (!isValidDevice()) { return false; }
  return m_device->pimCopyDeviceToMain(src, dest, idxBegin, idxEnd);
}

// @brief  Copy data from main memory to PIM device with type within a range
bool
pimSim::pimCopyMainToDeviceWithType(PimCopyEnum copyType, void* src, PimObjId dest, uint64_t idxBegin, uint64_t idxEnd)
{
  pimPerfMon perfMon("pimCopyMainToDevice");
  if (!isValidDevice()) { return false; }
  return m_device->pimCopyMainToDeviceWithType(copyType, src, dest, idxBegin, idxEnd);
}

// @brief  Copy data from PIM device to main memory with type within a range
bool
pimSim::pimCopyDeviceToMainWithType(PimCopyEnum copyType, PimObjId src, void* dest, uint64_t idxBegin, uint64_t idxEnd)
{
  pimPerfMon perfMon("pimCopyDeviceToMain");
  if (!isValidDevice()) { return false; }
  return m_device->pimCopyDeviceToMainWithType(copyType, src, dest, idxBegin, idxEnd);
}

// @brief  Copy data from PIM device to device within a range
bool
pimSim::pimCopyDeviceToDevice(PimObjId src, PimObjId dest, uint64_t idxBegin, uint64_t idxEnd)
{
  pimPerfMon perfMon("pimCopyDeviceToDevice");
  if (!isValidDevice()) { return false; }
  return m_device->pimCopyDeviceToDevice(src, dest, idxBegin, idxEnd);
}

// @brief  Load vector with a scalar value
template <typename T> bool
pimSim::pimBroadcast(PimObjId dest, T value)
{
  pimPerfMon perfMon("pimBroadcast");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdBroadcast<T>>(PimCmdEnum::BROADCAST, dest, value);
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

bool pimSim::pimAdd(PimObjId src, PimObjId dest, uint64_t scalerValue)
{
  pimPerfMon perfMon("pimAddScaler");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc1>(PimCmdEnum::ADD_SCALAR, src, dest, scalerValue);
  return m_device->executeCmd(std::move(cmd));
}

bool pimSim::pimSub(PimObjId src, PimObjId dest, uint64_t scalerValue)
{
  pimPerfMon perfMon("pimSubScaler");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc1>(PimCmdEnum::SUB_SCALAR, src, dest, scalerValue);
  return m_device->executeCmd(std::move(cmd));
}

bool pimSim::pimMul(PimObjId src, PimObjId dest, uint64_t scalerValue)
{
  pimPerfMon perfMon("pimMulScaler");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc1>(PimCmdEnum::MUL_SCALAR, src, dest, scalerValue);
  return m_device->executeCmd(std::move(cmd));
}

bool pimSim::pimDiv(PimObjId src, PimObjId dest, uint64_t scalerValue)
{
  pimPerfMon perfMon("pimDivScaler");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc1>(PimCmdEnum::DIV_SCALAR, src, dest, scalerValue);
  return m_device->executeCmd(std::move(cmd));
}

bool pimSim::pimAnd(PimObjId src, PimObjId dest, uint64_t scalerValue)
{
  pimPerfMon perfMon("pimAndScaler");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc1>(PimCmdEnum::AND_SCALAR, src, dest, scalerValue);
  return m_device->executeCmd(std::move(cmd));
}

bool pimSim::pimOr(PimObjId src, PimObjId dest, uint64_t scalerValue)
{
  pimPerfMon perfMon("pimOrScaler");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc1>(PimCmdEnum::OR_SCALAR, src, dest, scalerValue);
  return m_device->executeCmd(std::move(cmd));
}

bool pimSim::pimXor(PimObjId src, PimObjId dest, uint64_t scalerValue)
{
  pimPerfMon perfMon("pimXorScaler");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc1>(PimCmdEnum::XOR_SCALAR, src, dest, scalerValue);
  return m_device->executeCmd(std::move(cmd));
}

bool pimSim::pimXnor(PimObjId src, PimObjId dest, uint64_t scalerValue)
{
  pimPerfMon perfMon("pimXnorScaler");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc1>(PimCmdEnum::XNOR_SCALAR, src, dest, scalerValue);
  return m_device->executeCmd(std::move(cmd));
}

bool pimSim::pimGT(PimObjId src, PimObjId dest, uint64_t scalerValue)
{
  pimPerfMon perfMon("pimGTScaler");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc1>(PimCmdEnum::GT_SCALAR, src, dest, scalerValue);
  return m_device->executeCmd(std::move(cmd));
}

bool pimSim::pimLT(PimObjId src, PimObjId dest, uint64_t scalerValue)
{
  pimPerfMon perfMon("pimLTScaler");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc1>(PimCmdEnum::LT_SCALAR, src, dest, scalerValue);
  return m_device->executeCmd(std::move(cmd));
}

bool pimSim::pimEQ(PimObjId src, PimObjId dest, uint64_t scalerValue)
{
  pimPerfMon perfMon("pimEQScaler");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc1>(PimCmdEnum::EQ_SCALAR, src, dest, scalerValue);
  return m_device->executeCmd(std::move(cmd));
}

bool pimSim::pimMin(PimObjId src, PimObjId dest, uint64_t scalerValue)
{
  pimPerfMon perfMon("pimMinScaler");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc1>(PimCmdEnum::MIN_SCALAR, src, dest, scalerValue);
  return m_device->executeCmd(std::move(cmd));
}

bool pimSim::pimMax(PimObjId src, PimObjId dest, uint64_t scalerValue)
{
  pimPerfMon perfMon("pimMaxScaler");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc1>(PimCmdEnum::MAX_SCALAR, src, dest, scalerValue);
  return m_device->executeCmd(std::move(cmd));
}

bool pimSim::pimScaledAdd(PimObjId src1, PimObjId src2, PimObjId dest, uint64_t scalerValue) {
  pimPerfMon perfMon("pimScaledAdd");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc2>(PimCmdEnum::SCALED_ADD, src1, src2, dest, scalerValue);
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

template <typename T> bool
pimSim::pimRedSum(PimObjId src, T* sum)
{
  pimPerfMon perfMon("pimRedSum");
  if (!isValidDevice()) { return false; }
  *sum = 0;
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRedSum<T>>(PimCmdEnum::REDSUM, src, sum);
  return m_device->executeCmd(std::move(cmd));
}

template <typename T> bool
pimSim::pimRedSumRanged(PimObjId src, uint64_t idxBegin, uint64_t idxEnd, T* sum)
{
  pimPerfMon perfMon("pimRedSumRanged");
  if (!isValidDevice()) { return false; }
  *sum = 0;
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRedSum<T>>(PimCmdEnum::REDSUM_RANGE, src, sum, idxBegin, idxEnd);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimRotateElementsRight(PimObjId src)
{
  pimPerfMon perfMon("pimRotateElementsRight");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRotate>(PimCmdEnum::ROTATE_ELEM_R, src);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimRotateElementsLeft(PimObjId src)
{
  pimPerfMon perfMon("pimRotateElementsLeft");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRotate>(PimCmdEnum::ROTATE_ELEM_L, src);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimShiftElementsRight(PimObjId src)
{
  pimPerfMon perfMon("pimShiftElementsRight");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRotate>(PimCmdEnum::SHIFT_ELEM_R, src);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimShiftElementsLeft(PimObjId src)
{
  pimPerfMon perfMon("pimShiftElementsLeft");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdRotate>(PimCmdEnum::SHIFT_ELEM_L, src);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimShiftBitsRight(PimObjId src, PimObjId dest, unsigned shiftAmount)
{
  pimPerfMon perfMon("pimShiftBitsRight");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc1>(PimCmdEnum::SHIFT_BITS_R, src, dest, shiftAmount);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimShiftBitsLeft(PimObjId src, PimObjId dest, unsigned shiftAmount)
{
  pimPerfMon perfMon("pimShiftBitsLeft");
  if (!isValidDevice()) { return false; }
  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdFunc1>(PimCmdEnum::SHIFT_BITS_L, src, dest, shiftAmount);
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

bool
pimSim::pimOpAP(int numSrc, va_list args)
{
  pimPerfMon perfMon("pimOpAP");
  if (!isValidDevice()) { return false; }

  std::vector<std::pair<PimObjId, unsigned>> srcRows;
  for (int i = 0; i < numSrc; ++i) {
    PimObjId objId = va_arg(args, PimObjId);
    unsigned ofst = va_arg(args, unsigned);
    srcRows.emplace_back(objId, ofst);
  }

  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdAnalogAAP>(PimCmdEnum::ROW_AP, srcRows);
  return m_device->executeCmd(std::move(cmd));
}

bool
pimSim::pimOpAAP(int numSrc, int numDest, va_list args)
{
  pimPerfMon perfMon("pimOpAAP");
  if (!isValidDevice()) { return false; }

  std::vector<std::pair<PimObjId, unsigned>> srcRows;
  for (int i = 0; i < numSrc; ++i) {
    PimObjId objId = va_arg(args, PimObjId);
    int ofst = va_arg(args, unsigned);
    srcRows.emplace_back(objId, ofst);
  }
  std::vector<std::pair<PimObjId, unsigned>> destRows;
  for (int i = 0; i < numDest; ++i) {
    PimObjId objId = va_arg(args, PimObjId);
    int ofst = va_arg(args, unsigned);
    destRows.emplace_back(objId, ofst);
  }

  std::unique_ptr<pimCmd> cmd = std::make_unique<pimCmdAnalogAAP>(PimCmdEnum::ROW_AAP, srcRows, destRows);
  return m_device->executeCmd(std::move(cmd));
}

//! @breif parse config file to get memory config file path and maximum number of threads
bool
pimSim::parseConfigFromFile(const std::string& simConfigFileConetnt) {
  std::istringstream configStream(simConfigFileConetnt);
  std::string line;
  std::unordered_map<std::string, std::string> params;

  while (std::getline(configStream, line)) {
    line = pimUtils::removeAfterSemicolon(line);
    if (line.empty() || line[0] == '[') {
      continue;
    }
    size_t equalPos = line.find('=');
    if (equalPos != std::string::npos) {
      std::string key = line.substr(0, equalPos);
      std::string value = line.substr(equalPos + 1);
      params[pimUtils::trim(key)] = pimUtils::trim(value);
    }
  }
  try {
    bool success = false;
    std::string temp;
    temp = pimUtils::getOptionalParam(params, "max_num_threads", success);
    if (!success) {
      std::printf("PIM-Info: Maximum number of threads could not be located in PIMeval config file. Using maximum number of availale threads\n");
      m_numThreads = std::thread::hardware_concurrency();
    } else {
      m_numThreads = std::stoi(temp);
    }

    temp = pimUtils::getOptionalParam(params, "memory_config_file", success);
    if (!success) {
      std::printf("PIM-Info: PIM device params config file name could not be located in PIMeval config file. Using default values for memory config\n");
    } else {
      m_memConfigFileName = temp;
    }
  } catch (const std::invalid_argument& e) {
    std::string missing = e.what();
    std::string errorMessage("PIM-Error: Missing or invalid parameter: ");
    errorMessage += missing;
    errorMessage += "\n";
    std::printf("%s", errorMessage.c_str());
    return false;
  }
  return true;
}

// Explicit template instantiations
template bool pimSim::pimBroadcast<uint64_t>(PimObjId dest, uint64_t value);
template bool pimSim::pimBroadcast<int64_t>(PimObjId dest, int64_t value);

template bool pimSim::pimRedSum<uint64_t>(PimObjId src, uint64_t* sum);
template bool pimSim::pimRedSum<int64_t>(PimObjId src, int64_t* sum);

template bool pimSim::pimRedSumRanged<uint64_t>(PimObjId src, uint64_t idxBegin, uint64_t idxEnd, uint64_t* sum);
template bool pimSim::pimRedSumRanged<int64_t>(PimObjId src, uint64_t idxBegin, uint64_t idxEnd, int64_t* sum);

