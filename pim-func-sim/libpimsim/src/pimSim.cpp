// File: pimSim.cpp
// PIM Functional Simulator - PIM Simulator Main Entry
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "pimSim.h"
#include <iostream>


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
pimSim::createDevice(PimDeviceEnum deviceType, int numCores, int numRows, int numCols)
{
  if (m_device) {
    std::cout << "[PIM] Error: PIM device is already created" << std::endl;
    return false;
  }
  m_device = new pimDevice();
  m_device->init(deviceType, numCores, numRows, numCols);
  if (!m_device->isValid()) {
    delete m_device;
    std::cout << "[PIM] Error: Failed to create PIM device of type " << static_cast<int>(deviceType) << std::endl;
    return false;
  }
  return true;
}

//! @brief  Create a PIM device from a config file
bool
pimSim::createDeviceFromConfig(PimDeviceEnum deviceType, const char* configFileName)
{
  if (m_device) {
    std::cout << "[PIM] Error: PIM Device is already created" << std::endl;
    return false;
  }
  m_device = new pimDevice();
  m_device->init(deviceType, configFileName);
  if (!m_device->isValid()) {
    delete m_device;
    std::cout << "[PIM] Error: Failed to create PIM device of type " << static_cast<int>(deviceType) << std::endl;
    return false;
  }
  return true;
}

//! @brief  Delete PIM device
bool
pimSim::deleteDevice()
{
  if (!m_device) {
    std::cout << "[PIM] Error: No PIM device to delete" << std::endl;
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
    std::cout << "[PIM] Error: Invalid PIM device" << std::endl;
  }
  return isValid;
}

//! @brief  Allocate a PIM object
PimObjId
pimSim::pimAlloc(PimAllocEnum allocType, int numElements, int bitsPerElement)
{
  if (!isValidDevice()) { return -1; }
  return m_device->pimAlloc(allocType, numElements, bitsPerElement);
}

//! @brief  Allocate a PIM object that is associated with a reference ojbect
PimObjId
pimSim::pimAllocAssociated(PimAllocEnum allocType, int numElements, int bitsPerElement, PimObjId ref)
{
  if (!isValidDevice()) { return -1; }
  return m_device->pimAllocAssociated(allocType, numElements, bitsPerElement, ref);
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
  
  return true;
}

// @brief  Copy data from PIM device to main memory
bool
pimSim::pimCopyDeviceToMain(PimCopyEnum copyType, PimObjId src, void* dest)
{
  if (!isValidDevice()) { return false; }
  
  return true;
}

// @brief  PIM OP: add
bool
pimSim::pimAdd(PimObjId src1, PimObjId src2, PimObjId dest)
{
  
  return true;
}

