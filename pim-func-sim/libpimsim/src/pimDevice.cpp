// File: pimDevice.cpp
// PIM Functional Simulator - PIM Device
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "pimDevice.h"
#include "pimResMgr.h"
#include <cstdio>
#include <deque>


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
  // TODO: check existence of the config file

  // TODO: read parameters from config file
  std::printf("PIM-NYI: Creating PIM device from config file is not implemented yet\n");
  m_deviceType = deviceType;
  m_numCores = 0;
  m_numRows = 0;
  m_numCols = 0;
  m_isValid = false;

  m_resMgr = new pimResMgr(this);

  return m_isValid;
}

//! @brief  Alloc a PIM object
PimObjId
pimDevice::pimAlloc(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElement)
{
  return m_resMgr->pimAlloc(allocType, numElements, bitsPerElement);
}

//! @brief  Alloc a PIM object assiciated to a reference object
PimObjId
pimDevice::pimAllocAssociated(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElement, PimObjId ref)
{
  return m_resMgr->pimAllocAssociated(allocType, numElements, bitsPerElement, ref);
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
  if (copyType == PIM_COPY_V) {
    
  } else if (copyType == PIM_COPY_H) {
    
  } else {
    std::printf("PIM-Error: Unknown PIM copy type %d\n", static_cast<int>(copyType));
    return false;
  }
  return false;
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

