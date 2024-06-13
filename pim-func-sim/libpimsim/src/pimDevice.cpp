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
  if (!m_resMgr->isValidObjId(dest)) {
    std::printf("PIM-Error: Invalid PIM object ID %d as copy destination\n", dest);
    return false;
  }

  const pimObjInfo& pimObj = m_resMgr->getObjInfo(dest);
  unsigned numElements = pimObj.getNumElements();
  unsigned bitsPerElement = pimObj.getBitsPerElement();


  #if defined(DEBUG)
  std::printf("PIM-Info: Copying %u elements of %u bits from host to PIM obj %d\n", numElements, bitsPerElement, dest);
  #endif

  pimSim::get()->getStatsMgr()->recordCopyMainToDevice((uint64_t)numElements * bitsPerElement);

  // read in all bits from src
  std::vector<bool> bits = readBitsFromHost(src, numElements, bitsPerElement);

  if (copyType == PIM_COPY_V) {
    // read bits from src and store vertically into dest
    // assume the number of rows allocated matches the src data type
    // also assume little endian
    // directly copy without row read/write for now
    size_t bitIdx = 0;
    for (const auto& region : pimObj.getRegions()) {
      PimCoreId coreId = region.getCoreId();
      unsigned rowIdx = region.getRowIdx();
      unsigned colIdx = region.getColIdx();
      unsigned numAllocRows = region.getNumAllocRows();
      unsigned numAllocCols = region.getNumAllocCols();
      for (size_t i = 0; i < (size_t)numAllocRows * numAllocCols; ++i) {
        bool val = bits[bitIdx++];
        unsigned row = rowIdx + i % numAllocRows;
        unsigned col = colIdx + i / numAllocRows;
        m_cores[coreId].setBit(row, col, val);
      }
      // m_cores[coreId].print();
    }
  } else if (copyType == PIM_COPY_H) {
    // read bits from src and store horizontally into dest
    size_t bitIdx = 0;
    for (const auto& region : pimObj.getRegions()) {
      PimCoreId coreId = region.getCoreId();
      unsigned rowIdx = region.getRowIdx();
      unsigned colIdx = region.getColIdx();
      unsigned numAllocRows = region.getNumAllocRows();
      unsigned numAllocCols = region.getNumAllocCols();
      for (size_t i = 0; i < (size_t)numAllocRows * numAllocCols; ++i) {
        bool val = bits[bitIdx++];
        unsigned row = rowIdx + i / numAllocCols;
        unsigned col = colIdx + i % numAllocCols;
        m_cores[coreId].setBit(row, col, val);
      }
      //m_cores[coreId].print();
    }
  } else {
    std::printf("PIM-Error: Unknown PIM copy type %d\n", static_cast<int>(copyType));
    return false;
  }
  return true;
}

//! @brief  Copy data from PIM to host
bool
pimDevice::pimCopyDeviceToMainWithType(PimCopyEnum copyType, PimObjId src, void* dest)
{
  if (!m_resMgr->isValidObjId(src)) {
    std::printf("PIM-Error: Invalid PIM object ID %d as copy source\n", src);
    return false;
  }

  const pimObjInfo& pimObj = m_resMgr->getObjInfo(src);
  unsigned numElements = pimObj.getNumElements();
  unsigned bitsPerElement = pimObj.getBitsPerElement();


  #if defined(DEBUG)
  std::printf("PIM-Info: Copying %u elements of %u bits from PIM obj %d to host\n", numElements, bitsPerElement, src);
  #endif
  
  pimSim::get()->getStatsMgr()->recordCopyDeviceToMain((uint64_t)numElements * bitsPerElement);

  // read in all bits from src
  std::vector<bool> bits;

  if (copyType == PIM_COPY_V) {
    for (const auto& region : pimObj.getRegions()) {
      PimCoreId coreId = region.getCoreId();
      unsigned rowIdx = region.getRowIdx();
      unsigned colIdx = region.getColIdx();
      unsigned numAllocRows = region.getNumAllocRows();
      unsigned numAllocCols = region.getNumAllocCols();
      for (unsigned c = 0; c < numAllocCols; ++c) {
        for (unsigned r = 0; r < numAllocRows; ++r) {
          unsigned row = rowIdx + r;
          unsigned col = colIdx + c;
          bool val = m_cores[coreId].getBit(row, col);
          bits.push_back(val);
        }
      }
    }
  } else if (copyType == PIM_COPY_H) {
     for (const auto& region : pimObj.getRegions()) {
      PimCoreId coreId = region.getCoreId();
      unsigned rowIdx = region.getRowIdx();
      unsigned colIdx = region.getColIdx();
      unsigned numAllocRows = region.getNumAllocRows();
      unsigned numAllocCols = region.getNumAllocCols();
      for (unsigned r = 0; r < numAllocRows; ++r) {
        for (unsigned c = 0; c < numAllocCols; ++c) {
          unsigned row = rowIdx + r;
          unsigned col = colIdx + c;
          bool val = m_cores[coreId].getBit(row, col);
          bits.push_back(val);
        }
      }
    }
  } else {
    std::printf("PIM-Error: Unknown PIM copy type %d\n", static_cast<int>(copyType));
    return false;
  }

  return writeBitsToHost(dest, bits);
}

//! @brief  Read bits from host
std::vector<bool>
pimDevice::readBitsFromHost(void* src, unsigned numElements, unsigned bitsPerElement)
{

  std::vector<bool> bits;
  unsigned char* bytePtr = static_cast<unsigned char*>(src);
 
  for (size_t i = 0; i < (size_t)numElements * bitsPerElement; i += 8) {
    unsigned byteIdx = i / 8;
    unsigned char byteVal = *(bytePtr + byteIdx);
    for (int j = 0; j < 8; ++j) {
      bits.push_back(byteVal & 1);
      byteVal = byteVal >> 1;
    }
  }

  return bits;
}

//! @brief  Write bits to host
bool
pimDevice::writeBitsToHost(void* dest, const std::vector<bool>& bits)
{
  unsigned char* bytePtr = static_cast<unsigned char*>(dest);
  unsigned byteIdx = 0;

  for (size_t i = 0; i < bits.size(); i += 8) {
    unsigned char byteVal = 0;
    for (int j = 7; j >= 0; --j) {
      byteVal = byteVal << 1;
      byteVal |= bits[i + j];
    }
    *(bytePtr + byteIdx) = byteVal;
    byteIdx++;
  }

  return true;
}

//! @brief  Execute a PIM command
bool
pimDevice::executeCmd(std::unique_ptr<pimCmd> cmd)
{
  cmd->setDevice(this);
  bool ok = cmd->execute();

  return ok;
}

