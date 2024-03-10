// File: pimDevice.h
// PIM Functional Simulator - PIM Device
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef LAVA_PIM_DEVICE_H
#define LAVA_PIM_DEVICE_H

#include "libpimsim.h"
#include "pimCore.h"

class pimResMgr;


//! @class  pimDevice
//! @brief  PIM device
class pimDevice
{
public:
  pimDevice();
  ~pimDevice();

  bool init(PimDeviceEnum deviceType, unsigned numCores, unsigned numRows, unsigned numCols);
  bool init(PimDeviceEnum deviceType, const char* configFileName);

  PimDeviceEnum getDeviceType() const { return m_deviceType; }
  unsigned getNumCores() const { return m_numCores; }
  unsigned getNumRows() const { return m_numRows; }
  unsigned getNumCols() const { return m_numCols; }
  bool isValid() const { return m_isValid; }

  PimObjId pimAlloc(PimAllocEnum allocType, int numElements, int bitsPerElement);
  PimObjId pimAllocAssociated(PimAllocEnum allocType, int numElements, int bitsPerElement, PimObjId ref);
  bool pimFree(PimObjId obj);

private:
  PimDeviceEnum m_deviceType;
  unsigned m_numCores;
  unsigned m_numRows;
  unsigned m_numCols;
  bool m_isValid;
  pimResMgr* m_resMgr;
  std::vector<pimCore> m_cores;
};

#endif

