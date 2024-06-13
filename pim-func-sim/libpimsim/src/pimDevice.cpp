// File: pimDevice.cpp
// PIM Functional Simulator - PIM Device
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "pimDevice.h"
#include "pimResMgr.h"
#include "pimSim.h"
#include "libpimsim.h"
#include <cstdio>
#include <deque>
#include <memory>
#include <cassert>


//! @brief  pimDevice ctor
pimDevice::pimDevice()
{
}

//! @brief  pimDevice dtor
pimDevice::~pimDevice()
{
  delete m_resMgr;
  m_resMgr = nullptr;
}

//! @brief  Adjust config for modeling different simulation target with same inputs
bool
pimDevice::adjustConfigForSimTarget(unsigned& numRanks, unsigned& numBankPerRank, unsigned& numSubarrayPerBank, unsigned& numRows, unsigned& numCols)
{
  std::printf("PIM-Info: Config: #ranks = %u, #bankPerRank = %u, #subarrayPerBank = %u, #rows = %u, $cols = %u\n",
              numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
  PimDeviceEnum simTarget = pimSim::get()->getParamsPerf()->getSimTarget();
  switch (simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_AP:
  case PIM_DEVICE_BITSIMD_H:
  case PIM_DEVICE_FULCRUM:
    std::printf("PIM-Info: Aggregate every two subarrays as a single core\n");
    if (numSubarrayPerBank % 2 != 0) {
      std::printf("PIM-Error: Please config even number of subarrays in each bank\n");
      return false;
    }
    numRows *= 2;
    numSubarrayPerBank /= 2;
    break;
  case PIM_DEVICE_BANK_LEVEL:
    std::printf("PIM-Info: Aggregate all subarrays within a bank as a single core\n");
    numRows *= numSubarrayPerBank;
    numSubarrayPerBank = 1;
    break;
  default:
    assert(0);
  }
  return true;
}

//! @brief  Init pim device, with config file
bool
pimDevice::init(PimDeviceEnum deviceType, unsigned numRanks, unsigned numBankPerRank, unsigned numSubarrayPerBank, unsigned numRows, unsigned numCols)
{
  assert(!m_isInit);
  assert(deviceType != PIM_DEVICE_NONE);

  m_deviceType = deviceType;

  // input params
  m_numRanks = numRanks;
  m_numBankPerRank = numBankPerRank;
  m_numSubarrayPerBank = numSubarrayPerBank;
  m_numRowPerSubarray = numRows;
  m_numColPerSubarray = numCols;

  if (adjustConfigForSimTarget(numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols)) {
    m_numCores = numRanks * numBankPerRank * numSubarrayPerBank;
    m_numRows = numRows;
    m_numCols = numCols;
  } else {
    return false;
  }

  m_isValid = (m_numRanks > 0 && m_numCores > 0 && m_numRows > 0 && m_numCols > 0);
  assert(m_numCols % 8 == 0);

  if (!m_isValid) {
    std::printf("PIM-Error: Incorrect device parameters: %u cores, %u rows, %u columns\n", m_numCores, m_numRows, m_numCols);
    return false;
  }

  m_resMgr = new pimResMgr(this);

  m_cores.resize(m_numCores, pimCore(m_numRows, m_numCols));

  std::printf("PIM-Info: Created PIM device with %u cores of %u rows and %u columns.\n", m_numCores, m_numRows, m_numCols);

  m_isInit = true;
  return m_isValid;
}

//! @brief  Init pim device, with config file
bool
pimDevice::init(PimDeviceEnum deviceType, const char* configFileName)
{
  assert(!m_isInit);
  assert(deviceType != PIM_DEVICE_NONE);
  if (!configFileName) {
    std::printf("PIM-Error: Null PIM device config file name\n");
    return false;
  }
  if (!std::filesystem::exists(configFileName)) {
    std::printf("PIM-Error: Config file not found.\n");
    return false;
  }

  m_deviceType = deviceType;

#ifdef DRAMSIM3_INTEG
  std::string configFile(configFileName);
  //TODO: DRAMSim3 requires an output directory but for our purpose we do not need it so sending empty string
  m_deviceMemory = new dramsim3::PIMCPU(configFile, "");
  m_deviceMemoryConfig = m_deviceMemory->getMemorySystem()->getConfig();
  u_int64_t rowsPerBank = m_deviceMemoryConfig->rows, columnPerRow = m_deviceMemoryConfig->columns * m_deviceMemoryConfig->device_width;

  // todo: adjust for sim target
  m_numRanks = 1;
  m_numCores = 16;
  m_numRows = rowsPerBank/m_numCores;
  m_numCols = columnPerRow;
#endif

  m_isValid = (m_numRanks > 0 && m_numCores > 0 && m_numRows > 0 && m_numCols > 0);
  assert(m_numCols % 8 == 0);

  if (!m_isValid) {
    std::printf("PIM-Error: Incorrect device parameters: %u cores, %u rows, %u columns\n", m_numCores, m_numRows, m_numCols);
    return false;
  }

  m_resMgr = new pimResMgr(this);

  m_cores.resize(m_numCores, pimCore(m_numRows, m_numCols));

  std::printf("PIM-Info: Created PIM device with %u cores of %u rows and %u columns.\n", m_numCores, m_numRows, m_numCols);

  m_isInit = true;
  return m_isValid;
}

//! @brief  Uninit pim device
void
pimDevice::uninit()
{
  m_cores.clear();
  delete m_resMgr;
  m_resMgr = nullptr;
  m_deviceType = PIM_DEVICE_NONE;
  m_numCores = 0;
  m_numRows = 0;
  m_numCols = 0;
  m_isValid = false;
  m_isInit = false;
}

//! @brief  Alloc a PIM object
PimObjId
pimDevice::pimAlloc(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElement, PimDataType dataType)
{
  if (allocType == PIM_ALLOC_AUTO) {
    if (pimSim::get()->getParamsPerf()->isVLayoutDevice()) {
      allocType = PIM_ALLOC_V;
    } else if (pimSim::get()->getParamsPerf()->isHLayoutDevice()) {
      allocType = PIM_ALLOC_H;
    } else {
      assert(0);
    }
  }
  return m_resMgr->pimAlloc(allocType, numElements, bitsPerElement, dataType);
}

//! @brief  Alloc a PIM object assiciated to a reference object
PimObjId
pimDevice::pimAllocAssociated(unsigned bitsPerElement, PimObjId ref, PimDataType dataType)
{
  return m_resMgr->pimAllocAssociated(bitsPerElement, ref, dataType);
}

//! @brief  Free a PIM object
bool
pimDevice::pimFree(PimObjId obj)
{
  return m_resMgr->pimFree(obj);
}

//! @brief  Copy data from host to PIM
bool
pimDevice::pimCopyMainToDevice(void* src, PimObjId dest)
{
  PimCopyEnum copyType = m_resMgr->isHLayoutObj(dest) ? PIM_COPY_H : PIM_COPY_V;
  return pimCopyMainToDeviceWithType(copyType, src, dest);
}

//! @brief  Copy data from PIM to host
bool
pimDevice::pimCopyDeviceToMain(PimObjId src, void* dest)
{
  PimCopyEnum copyType = m_resMgr->isHLayoutObj(src) ? PIM_COPY_H : PIM_COPY_V;
  return pimCopyDeviceToMainWithType(copyType, src, dest);
}

//! @brief  Copy data from host to PIM
bool
pimDevice::pimCopyMainToDeviceWithType(PimCopyEnum copyType, void* src, PimObjId dest)
{
  std::unique_ptr<pimCmd> cmd =
    std::make_unique<pimCmdCopy>(PimCmdEnum::COPY_H2D, copyType, src, dest);
  return executeCmd(std::move(cmd));
}

//! @brief  Copy data from PIM to host
bool
pimDevice::pimCopyDeviceToMainWithType(PimCopyEnum copyType, PimObjId src, void* dest)
{
  std::unique_ptr<pimCmd> cmd =
    std::make_unique<pimCmdCopy>(PimCmdEnum::COPY_D2H, copyType, src, dest);
  return executeCmd(std::move(cmd));
}

//! @brief  Copy data from PIM to PIM
bool
pimDevice::pimCopyDeviceToDevice(PimObjId src, PimObjId dest)
{
  PimCopyEnum copyType = PIM_COPY_V; // not used
  std::unique_ptr<pimCmd> cmd =
    std::make_unique<pimCmdCopy>(PimCmdEnum::COPY_D2D, copyType, src, dest);
  return executeCmd(std::move(cmd));
}

//! @brief  Execute a PIM command
bool
pimDevice::executeCmd(std::unique_ptr<pimCmd> cmd)
{
  cmd->setDevice(this);
  bool ok = cmd->execute();

  return ok;
}

