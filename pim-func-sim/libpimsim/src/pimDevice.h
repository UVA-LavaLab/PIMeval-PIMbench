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

  bool init(PimDeviceEnum deviceType, unsigned numRanks, unsigned numBankPerRank, unsigned numSubarrayPerBank, unsigned numRows, unsigned numCols);
  bool init(PimDeviceEnum deviceType, const char* configFileName);
  void uninit();

  PimDeviceEnum getDeviceType() const { return m_deviceType; }
  unsigned getNumRanks() const { return m_numRanks; }
  unsigned getNumBankPerRank() const { return m_numBankPerRank; }
  unsigned getNumSubarrayPerBank() const { return m_numSubarrayPerBank; }
  unsigned getNumRowPerSubarray() const { return m_numRowPerSubarray; }
  unsigned getNumColPerSubarray() const { return m_numColPerSubarray; }
  unsigned getNumCores() const { return m_numCores; }
  unsigned getNumRows() const { return m_numRows; }
  unsigned getNumCols() const { return m_numCols; }
  bool isValid() const { return m_isValid; }

  PimObjId pimAlloc(PimAllocEnum allocType, uint64_t numElements, unsigned bitsPerElement, PimDataType dataType);
  PimObjId pimAllocAssociated(unsigned bitsPerElement, PimObjId assocId, PimDataType dataType);
  bool pimFree(PimObjId obj);
  PimObjId pimCreateRangedRef(PimObjId refId, uint64_t idxBegin, uint64_t idxEnd);
  PimObjId pimCreateDualContactRef(PimObjId refId);

  bool pimCopyMainToDevice(void* src, PimObjId dest);
  bool pimCopyDeviceToMain(PimObjId src, void* dest);
  bool pimCopyMainToDeviceWithType(PimCopyEnum copyType, void* src, PimObjId dest);
  bool pimCopyDeviceToMainWithType(PimCopyEnum copyType, PimObjId src, void* dest);
  bool pimCopyDeviceToDevice(PimObjId src, PimObjId dest);

  pimResMgr* getResMgr() { return m_resMgr; }
  pimCore& getCore(PimCoreId coreId) { return m_cores[coreId]; }
  bool executeCmd(std::unique_ptr<pimCmd> cmd);

private:
  bool adjustConfigForSimTarget(unsigned& numRanks, unsigned& numBankPerRank, unsigned& numSubarrayPerBank, unsigned& numRows, unsigned& numCols);

  PimDeviceEnum m_deviceType = PIM_DEVICE_NONE;
  unsigned m_numRanks = 0;
  unsigned m_numBankPerRank = 0;
  unsigned m_numSubarrayPerBank = 0;
  unsigned m_numRowPerSubarray = 0;
  unsigned m_numColPerSubarray = 0;
  unsigned m_numCores = 0;
  unsigned m_numRows = 0;
  unsigned m_numCols = 0;
  bool m_isValid = false;
  bool m_isInit = false;
  pimResMgr* m_resMgr = nullptr;
  std::vector<pimCore> m_cores;

#ifdef DRAMSIM3_INTEG
  dramsim3::PIMCPU* m_hostMemory = nullptr;
  dramsim3::PIMCPU* m_deviceMemory = nullptr;
  dramsim3::Config* m_deviceMemoryConfig = nullptr;
#endif
};

#endif
