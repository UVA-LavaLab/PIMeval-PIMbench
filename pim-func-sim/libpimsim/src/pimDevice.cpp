// File: pimDevice.cpp
// PIM Functional Simulator - PIM Device
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "pimDevice.h"
#include "pimResMgr.h"
#include <cstdio>


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
    std::printf("[PIM] Error: Incorrect device parameters: %d cores, %d rows, %d columns\n", numCores, numRows, numCols);
    return false;
  }

  m_resMgr = new pimResMgr(this);

  m_cores.resize(m_numCores, pimCore(m_numRows, m_numCols));

  return m_isValid;
}

//! @brief  Init pim device, with config file
bool
pimDevice::init(PimDeviceEnum deviceType, const char* configFileName)
{
  if (!configFileName) {
    std::printf("[PIM] Error: Null PIM device config file name\n");
    return false;
  }
  // TODO: check existence of the config file

  // TODO: read parameters from config file
  std::printf("[PIM] NYI: Creating PIM device from config file is not implemented yet\n");
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
pimDevice::pimAlloc(PimAllocEnum allocType, int numElements, int bitsPerElement)
{
  return m_resMgr->pimAlloc(allocType, numElements, bitsPerElement);
}

//! @brief  Alloc a PIM object assiciated to a reference object
PimObjId
pimDevice::pimAllocAssociated(PimAllocEnum allocType, int numElements, int bitsPerElement, PimObjId ref)
{
  return m_resMgr->pimAllocAssociated(allocType, numElements, bitsPerElement, ref);
}

//! @brief  Free a PIM object
bool
pimDevice::pimFree(PimObjId obj)
{
  return m_resMgr->pimFree(obj);
}

