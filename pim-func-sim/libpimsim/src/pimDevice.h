// File: pimDevice.h
// PIM Functional Simulator - PIM Device
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef LAVA_PIM_DEVICE_H
#define LAVA_PIM_DEVICE_H

#include "libpimsim.h"
#include "pimResMgr.h"


//! @class  pimDevice
//! @brief  PIM device
class pimDevice
{
public:
  pimDevice();
  ~pimDevice();

  bool init(PimDeviceEnum deviceType, int numCores, int numRows, int numCols);
  bool init(PimDeviceEnum deviceType, const char* configFileName);

  bool isValid() const { return m_isValid; }

  PimObjId pimAlloc(PimAllocEnum allocType, int numElements, int bitsPerElement);
  PimObjId pimAllocAssociated(PimAllocEnum allocType, int numElements, int bitsPerElement, PimObjId ref);
  bool pimFree(PimObjId obj);

private:
  PimDeviceEnum m_deviceType;
  int m_numCores;
  int m_numRows;
  int m_numCols;
  bool m_isValid;
  pimResMgr* m_resMgr;
};

#endif
