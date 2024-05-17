// File: pimDevice.cpp
// PIM Functional Simulator - PIM Device
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "pimDevice.h"
#include "pimResMgr.h"
#include "pimSim.h"
#include <cstdio>
#include <deque>
#include <memory>


//! @brief  pimDevice ctor
pimDevice::pimDevice()
  : m_deviceType(PIM_DEVICE_NONE),
    m_numCores(0),
    m_numRows(0),
    m_numCols(0),
    m_isValid(false),
    m_resMgr(nullptr)
{
}

//! @brief  pimDevice dtor
pimDevice::~pimDevice()
{
  delete m_resMgr;
  m_resMgr = nullptr;
}

//! @brief  Init pim device, with config file
bool
pimDevice::init(PimDeviceEnum deviceType, unsigned numCores, unsigned numRows, unsigned numCols)
{
  m_deviceType = deviceType;
  m_numCores = numCores;
  m_numRows = numRows;
  m_numCols = numCols;
  m_isValid = (numCores > 0 && numRows > 0 && numCols > 0);

  if (!m_isValid) {
    std::printf("PIM-Error: Incorrect device parameters: %u cores, %u rows, %u columns\n", numCores, numRows, numCols);
    return false;
  }

  m_resMgr = new pimResMgr(this);

  m_cores.resize(m_numCores, pimCore(m_numRows, m_numCols));

  std::printf("PIM-Info: Created PIM device with %u cores of %u rows and %u columns.\n", numCores, numRows, numCols);

  return m_isValid;
}

//! @brief  Init pim device, with config file
bool
pimDevice::init(PimDeviceEnum deviceType, const char* configFileName)
{
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
  m_numCores = 16;
  m_numRows = rowsPerBank/m_numCores;
  m_numCols = columnPerRow;
#endif

  m_isValid = (m_numCores > 0 && m_numRows > 0 && m_numCols > 0);

  if (!m_isValid) {
    std::printf("PIM-Error: Incorrect device parameters: %u cores, %u rows, %u columns\n", m_numCores, m_numRows, m_numCols);
    return false;
  }

  m_resMgr = new pimResMgr(this);

  m_cores.resize(m_numCores, pimCore(m_numRows, m_numCols));

  std::printf("PIM-Info: Created PIM device with %u cores of %u rows and %u columns.\n", m_numCores, m_numRows, m_numCols);

  return m_isValid;
}

//! @brief  Alloc a PIM object
PimObjId
pimDevice::pimAlloc(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElement, PimDataType dataType)
{
  return m_resMgr->pimAlloc(allocType, numElements, bitsPerElement, dataType);
}

//! @brief  Alloc a PIM object assiciated to a reference object
PimObjId
pimDevice::pimAllocAssociated(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElement, PimObjId ref, PimDataType dataType)
{
  return m_resMgr->pimAllocAssociated(allocType, numElements, bitsPerElement, ref, dataType);
}

//! @brief  Free a PIM object
bool
pimDevice::pimFree(PimObjId obj)
{
  return m_resMgr->pimFree(obj);
}

//! @brief  Copy data from host to PIM
bool
pimDevice::pimCopyMainToDevice(PimCopyEnum copyType, void* src, PimObjId dest)
{
  if (!m_resMgr->isValidObjId(dest)) {
    std::printf("PIM-Error: Invalid PIM object ID %d as copy destination\n", dest);
    return false;
  }

  const pimObjInfo& pimObj = m_resMgr->getObjInfo(dest);
  unsigned numElements = pimObj.getNumElements();
  unsigned bitsPerElement = pimObj.getBitsPerElement();

  std::printf("PIM-Info: Copying %u elements of %u bits from host to PIM obj %d\n", numElements, bitsPerElement, dest);

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

//! @brief  Load the vector with a scalar value
bool
pimDevice::pimBroadCast(PimCopyEnum copyType, PimObjId dest, unsigned value)
{
  if (!m_resMgr->isValidObjId(dest)) {
    std::printf("PIM-Error: Invalid PIM object ID %d as copy destination\n", dest);
    return false;
  }

  const pimObjInfo& pimObj = m_resMgr->getObjInfo(dest);
  unsigned numElements = pimObj.getNumElements();
  unsigned bitsPerElement = pimObj.getBitsPerElement();

  std::printf("PIM-Info: Loading %u elements of %u bits with %u to PIM obj %d\n", numElements, bitsPerElement, value, dest);
  std::vector <unsigned> srcVec(numElements, value);
  // read in all bits from src
  std::vector<bool> bits = readBitsFromHost((void*)srcVec.data(), numElements, bitsPerElement);

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
      //m_cores[coreId].print();
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
pimDevice::pimCopyDeviceToMain(PimCopyEnum copyType, PimObjId src, void* dest)
{
  if (!m_resMgr->isValidObjId(src)) {
    std::printf("PIM-Error: Invalid PIM object ID %d as copy source\n", src);
    return false;
  }

  const pimObjInfo& pimObj = m_resMgr->getObjInfo(src);
  unsigned numElements = pimObj.getNumElements();
  unsigned bitsPerElement = pimObj.getBitsPerElement();

  std::printf("PIM-Info: Copying %u elements of %u bits from PIM obj %d to host\n", numElements, bitsPerElement, src);

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
  pimSim::get()->getStatsMgr()->recordCmd(cmd->getName());

  bool ok = cmd->execute(this);

  return ok;
}

