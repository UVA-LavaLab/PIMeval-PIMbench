// File: pimParamsPerf.h
// PIM Functional Simulator - Performance parameters
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef LAVA_PIM_PARAMS_PERF_H
#define LAVA_PIM_PARAMS_PERF_H

#include "libpimeval.h"
#include "pimParamsDram.h"
#include "pimCmd.h"
#include "pimResMgr.h"
#include <unordered_map>
#include <tuple>


//! @class  pimParamsPerf
//! @brief  PIM performance parameters base class
class pimParamsPerf
{
public:
  pimParamsPerf(pimParamsDram* paramsDram);
  ~pimParamsPerf() {}

  double getMsRuntimeForBytesTransfer(uint64_t numBytes) const;
  double getMsRuntimeForFunc1(PimCmdEnum cmdType, const pimObjInfo& obj) const;
  double getMsRuntimeForFunc2(PimCmdEnum cmdType, const pimObjInfo& obj) const;
  double getMsRuntimeForRedSum(PimCmdEnum cmdType, const pimObjInfo& obj, unsigned numPass) const;
  double getMsRuntimeForBroadcast(PimCmdEnum cmdType, const pimObjInfo& obj) const;
  double getMsRuntimeForRotate(PimCmdEnum cmdType, const pimObjInfo& obj) const;

private:
  double getMsRuntimeBitSerial(PimDeviceEnum deviceType, PimCmdEnum cmdType, PimDataType dataType, unsigned bitsPerElement, unsigned numPass) const;

  const pimParamsDram* m_paramsDram;
  double m_tR; // Row read latency in ms
  double m_tW; // Row write latency in ms
  double m_tL; // Logic operation / tCCD in ms
  double m_fulcrumAluLatency = 0.00000609; // 6.09ns
  unsigned m_flucrumAluBitWidth = 32;
  double m_blimpCoreLatency = 0.000005; // 200 MHz. Reference: BLIMP paper
  unsigned m_blimpCoreBitWidth = 64; 

  static const std::unordered_map<PimDeviceEnum, std::unordered_map<PimDataType,
      std::unordered_map<PimCmdEnum, std::tuple<unsigned, unsigned, unsigned>>>> s_bitsimdPerfTable;
};

#endif

