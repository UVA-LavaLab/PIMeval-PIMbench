// File: pimSim.h
// PIM Functional Simulator - PIM Simulator Main Entry
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef LAVA_PIM_SIM_H
#define LAVA_PIM_SIM_H

#include "libpimsim.h"
#include "pimDevice.h"
#include <vector>


//! @class  pimSim
//! @brief  PIM simulator singleton class
class pimSim
{
public:
  static pimSim* get();

  // Device creation and deletion
  bool createDevice(PimDeviceEnum deviceType, unsigned numCores, unsigned numRows, unsigned numCols);
  bool createDeviceFromConfig(PimDeviceEnum deviceType, const char* configFileName);
  bool deleteDevice();
  bool isValidDevice(bool showMsg = true) const;

  // Resource allocation and deletion
  PimObjId pimAlloc(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElement);
  PimObjId pimAllocAssociated(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElement, PimObjId ref);
  bool pimFree(PimObjId obj);

  // Data transfer
  bool pimCopyMainToDevice(PimCopyEnum copyType, void* src, PimObjId dest);
  bool pimCopyDeviceToMain(PimCopyEnum copyType, PimObjId src, void* dest);

  // Computation
  bool pimAddInt32V(PimObjId src1, PimObjId src2, PimObjId dest);

private:
  pimSim() {}
  ~pimSim() {}
  pimSim(const pimSim&) = delete;
  pimSim operator=(const pimSim&) = delete;

  static pimSim* s_instance;

  // support one device for now
  pimDevice* m_device;
};

#endif

