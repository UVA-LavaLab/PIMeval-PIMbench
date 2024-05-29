// File: pimParamsPerf.h
// PIM Functional Simulator - Performance parameters
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef LAVA_PIM_PARAMS_PERF_H
#define LAVA_PIM_PARAMS_PERF_H

#include "libpimsim.h"
#include "pimParamsDram.h"
#include "pimCmd.h"


//! @class  pimParamsPerf
//! @brief  PIM performance parameters base class
class pimParamsPerf
{
public:
  pimParamsPerf(pimParamsDram* paramsDram);
  ~pimParamsPerf() {}

  void setDevice(PimDeviceEnum deviceType);

  PimDeviceEnum getCurDevice() const { return m_curDevice; }
  PimDeviceEnum getSimTarget() const { return m_simTarget; }

  bool isVLayoutDevice() const;
  bool isHLayoutDevice() const;
  bool isHybridLayoutDevice() const;

  double getMsRuntimeForBytesTransfer(uint64_t numBytes) const;
  double getMsRuntimeForFunc1(PimCmdEnum cmdType) const;
  double getMsRuntimeForFunc2(PimCmdEnum cmdType) const;
  double getMsRuntimeForRedSum(PimCmdEnum cmdType, unsigned numElements) const;
  double getMsRuntimeForBroadcast(PimCmdEnum cmdType, bool isVLayout, unsigned bitsPerElement, unsigned maxElementsPerRegion) const;
  double getMsRuntimeForRotate(PimCmdEnum cmdType, unsigned bitsPerElement, unsigned numRegions) const;

private:
  const pimParamsDram* m_paramsDram; 
  PimDeviceEnum m_curDevice;
  PimDeviceEnum m_simTarget;
  double m_tR; // Row read latency in ms
  double m_tW; // Row write latency in ms
  double m_tL; // Logic operation / tCCD in ms
};

#endif

