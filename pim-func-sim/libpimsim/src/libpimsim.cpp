// File: libpimsim.cpp
// PIM Functional Simulator Library Interface
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "libpimsim.h"
#include "pimSim.h"


//! @brief  Create a PIM device
PimStatus
pimCreateDevice(PimDeviceEnum deviceType, unsigned numCores, unsigned numRows, unsigned numCols)
{
  bool ok = pimSim::get()->createDevice(deviceType, numCores, numRows, numCols);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Create a PIM device from config file
PimStatus
pimCreateDeviceFromConfig(PimDeviceEnum deviceType, const char* configFileName)
{
  bool ok = pimSim::get()->createDeviceFromConfig(deviceType, configFileName);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Delete a PIM device
PimStatus
pimDeleteDevice()
{
  bool ok = pimSim::get()->deleteDevice();
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Allocate a PIM resource
PimObjId
pimAlloc(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElements)
{
  return pimSim::get()->pimAlloc(allocType, numElements, bitsPerElements);
}

//! @brief  Allocate a PIM resource, with an associated object as reference
PimObjId
pimAllocAssociated(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElements, PimObjId ref)
{
  return pimSim::get()->pimAllocAssociated(allocType, numElements, bitsPerElements, ref);
}

//! @brief  Free a PIM resource
PimStatus
pimFree(PimObjId obj)
{
  bool ok = pimSim::get()->pimFree(obj);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Copy data from main to PIM device
PimStatus
pimCopyHostToDevice(PimCopyEnum copyType, void* src, PimObjId dest)
{
  bool ok = pimSim::get()->pimCopyMainToDevice(copyType, src, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Copy data from PIM device to main
PimStatus
pimCopyDeviceToHost(PimCopyEnum copyType, PimObjId src, void* dest)
{
  bool ok = pimSim::get()->pimCopyDeviceToMain(copyType, src, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM vector add int32 v-layout
PimStatus
pimAddInt32V(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimAddInt32V(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

