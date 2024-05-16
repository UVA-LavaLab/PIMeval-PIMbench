// File: pimDevice.h
// PIM Functional Simulator - PIM Device
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef LAVA_PIM_DEVICE_H
#define LAVA_PIM_DEVICE_H

#include "libpimsim.h"
#include "pimCore.h"
#include "pimCmd.h"
#ifdef DRAMSIM3_INTEG
#include "cpu.h"
#endif
#include <memory>
#include <filesystem>

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

  PimObjId pimAlloc(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElement, PimDataType dataType);
  PimObjId pimAllocAssociated(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElement, PimObjId ref, PimDataType dataType);
  bool pimFree(PimObjId obj);

  bool pimCopyMainToDevice(PimCopyEnum copyType, void* src, PimObjId dest);
  bool pimCopyDeviceToMain(PimCopyEnum copyType, PimObjId src, void* dest);
  bool pimBroadCast(PimCopyEnum copyType, PimObjId dest, unsigned value);

  pimResMgr* getResMgr() { return m_resMgr; }
  pimCore& getCore(PimCoreId coreId) { return m_cores[coreId]; }
  bool executeCmd(std::unique_ptr<pimCmd> cmd);

private:
  std::vector<bool> readBitsFromHost(void* src, unsigned numElements, unsigned bitsPerElement);
  bool writeBitsToHost(void* dest, const std::vector<bool>& bits);

  PimDeviceEnum m_deviceType;
  unsigned m_numCores;
  unsigned m_numRows;
  unsigned m_numCols;
  bool m_isValid;
  pimResMgr* m_resMgr;
  std::vector<pimCore> m_cores;

#ifdef DRAMSIM3_INTEG
  dramsim3::PIMCPU* m_hostMemory;
  dramsim3::PIMCPU* m_deviceMemory;
  dramsim3::Config* m_deviceMemoryConfig;
#endif
};

#endif
