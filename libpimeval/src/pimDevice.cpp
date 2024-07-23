// File: pimDevice.cpp
// PIMeval Simulator - PIM Device
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimDevice.h"
#include "pimResMgr.h"
#include "pimSim.h"
#include "libpimeval.h"
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
  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_NAND:
  case PIM_DEVICE_BITSIMD_V_MAJ:
  case PIM_DEVICE_BITSIMD_V_AP:
  case PIM_DEVICE_DRISA_NOR:
  case PIM_DEVICE_DRISA_MIXED:
  case PIM_DEVICE_SIMDRAM:
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

//! @brief  Config device type and simulation target
void
pimDevice::configDevice(PimDeviceEnum curDevice, PimDeviceEnum simTarget)
{
  m_deviceType = curDevice;
  m_simTarget = curDevice;

  // determine simulation target for functional device
  if (curDevice == PIM_FUNCTIONAL) {
    // from 'make PIM_SIM_TARGET=...'
    #if defined(PIM_SIM_TARGET)
    if (simTarget == PIM_DEVICE_NONE) {
      simTarget = PIM_SIM_TARGET;
    }
    #endif
    // default sim target
    if (simTarget == PIM_DEVICE_NONE || simTarget == PIM_FUNCTIONAL) {
      simTarget = PIM_DEVICE_BITSIMD_V;
    }
    m_simTarget = simTarget;
  }
}

//! @brief  If a PIM device uses vertical data layout
bool
pimDevice::isVLayoutDevice() const
{
  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V: return true;
  case PIM_DEVICE_BITSIMD_V_NAND: return true;
  case PIM_DEVICE_BITSIMD_V_MAJ: return true;
  case PIM_DEVICE_BITSIMD_V_AP: return true;
  case PIM_DEVICE_DRISA_NOR: return true;
  case PIM_DEVICE_DRISA_MIXED: return true;
  case PIM_DEVICE_SIMDRAM: return true;
  case PIM_DEVICE_BITSIMD_H: return false;
  case PIM_DEVICE_FULCRUM: return false;
  case PIM_DEVICE_BANK_LEVEL: return false;
  case PIM_DEVICE_NONE:
  case PIM_FUNCTIONAL:
  default:
    assert(0);
  }
  return false;
}

//! @brief  If a PIM device uses horizontal data layout
bool
pimDevice::isHLayoutDevice() const
{
  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V: return false;
  case PIM_DEVICE_BITSIMD_V_NAND: return false;
  case PIM_DEVICE_BITSIMD_V_MAJ: return false;
  case PIM_DEVICE_BITSIMD_V_AP: return false;
  case PIM_DEVICE_DRISA_NOR: return false;
  case PIM_DEVICE_DRISA_MIXED: return false;
  case PIM_DEVICE_SIMDRAM: return false;
  case PIM_DEVICE_BITSIMD_H: return true;
  case PIM_DEVICE_FULCRUM: return true;
  case PIM_DEVICE_BANK_LEVEL: return true;
  case PIM_DEVICE_NONE:
  case PIM_FUNCTIONAL:
  default:
    assert(0);
  }
  return false;
}

//! @brief  If a PIM device uses hybrid data layout
bool
pimDevice::isHybridLayoutDevice() const
{
  return false;
}

//! @brief  Init pim device, with config file
bool
pimDevice::init(PimDeviceEnum deviceType, unsigned numRanks, unsigned numBankPerRank, unsigned numSubarrayPerBank, unsigned numRows, unsigned numCols)
{
  assert(!m_isInit);
  assert(deviceType != PIM_DEVICE_NONE);

  configDevice(deviceType);
  std::printf("PIM-Info: Current Device = %s, Simulation Target = %s\n",
              pimUtils::pimDeviceEnumToStr(m_deviceType).c_str(),
              pimUtils::pimDeviceEnumToStr(m_simTarget).c_str());

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

  unsigned maxNumThreads = 0; // use max hardware parallelism by default
  // TODO: read max num threads from config file
  pimSim::get()->initThreadPool(maxNumThreads);

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
pimDevice::pimAlloc(PimAllocEnum allocType, uint64_t numElements, unsigned bitsPerElement, PimDataType dataType)
{
  if (allocType == PIM_ALLOC_AUTO) {
    if (isVLayoutDevice()) {
      allocType = PIM_ALLOC_V;
    } else if (isHLayoutDevice()) {
      allocType = PIM_ALLOC_H;
    } else {
      assert(0);
    }
  }
  return m_resMgr->pimAlloc(allocType, numElements, bitsPerElement, dataType);
}

//! @brief  Alloc a PIM object assiciated to a reference object
PimObjId
pimDevice::pimAllocAssociated(unsigned bitsPerElement, PimObjId assocId, PimDataType dataType)
{
  return m_resMgr->pimAllocAssociated(bitsPerElement, assocId, dataType);
}

//! @brief  Free a PIM object
bool
pimDevice::pimFree(PimObjId obj)
{
  return m_resMgr->pimFree(obj);
}

//! @brief  Create an obj referencing to a range of an existing obj
PimObjId
pimDevice::pimCreateRangedRef(PimObjId refId, uint64_t idxBegin, uint64_t idxEnd)
{
  return m_resMgr->pimCreateRangedRef(refId, idxBegin, idxEnd);
}

//! @brief  Create an obj referencing to negation of an existing obj based on dual-contact memory cells
PimObjId
pimDevice::pimCreateDualContactRef(PimObjId refId)
{
  return m_resMgr->pimCreateDualContactRef(refId);
}

//! @brief  Copy data from host to PIM within a range
bool
pimDevice::pimCopyMainToDevice(void* src, PimObjId dest, uint64_t idxBegin, uint64_t idxEnd)
{
  PimCopyEnum copyType = m_resMgr->isHLayoutObj(dest) ? PIM_COPY_H : PIM_COPY_V;
  return pimCopyMainToDeviceWithType(copyType, src, dest, idxBegin, idxEnd);
}

//! @brief  Copy data from PIM to host within a range
bool
pimDevice::pimCopyDeviceToMain(PimObjId src, void* dest, uint64_t idxBegin, uint64_t idxEnd)
{
  PimCopyEnum copyType = m_resMgr->isHLayoutObj(src) ? PIM_COPY_H : PIM_COPY_V;
  return pimCopyDeviceToMainWithType(copyType, src, dest, idxBegin, idxEnd);
}

//! @brief  Copy data from host to PIM within a range
bool
pimDevice::pimCopyMainToDeviceWithType(PimCopyEnum copyType, void* src, PimObjId dest, uint64_t idxBegin, uint64_t idxEnd)
{
  std::unique_ptr<pimCmd> cmd =
    std::make_unique<pimCmdCopy>(PimCmdEnum::COPY_H2D, copyType, src, dest, idxBegin, idxEnd);
  return executeCmd(std::move(cmd));
}

//! @brief  Copy data from PIM to host within a range
bool
pimDevice::pimCopyDeviceToMainWithType(PimCopyEnum copyType, PimObjId src, void* dest, uint64_t idxBegin, uint64_t idxEnd)
{
  std::unique_ptr<pimCmd> cmd =
    std::make_unique<pimCmdCopy>(PimCmdEnum::COPY_D2H, copyType, src, dest, idxBegin, idxEnd);
  return executeCmd(std::move(cmd));
}

//! @brief  Copy data from PIM to PIM within a range
bool
pimDevice::pimCopyDeviceToDevice(PimObjId src, PimObjId dest, uint64_t idxBegin, uint64_t idxEnd)
{
  const pimObjInfo& obj = m_resMgr->getObjInfo(src);
  PimCopyEnum copyType = obj.isVLayout() ? PIM_COPY_V : PIM_COPY_H;
  std::unique_ptr<pimCmd> cmd =
    std::make_unique<pimCmdCopy>(PimCmdEnum::COPY_D2D, copyType, src, dest, idxBegin, idxEnd);
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

