// File: pimParamsPerf.h
// PIMeval Simulator - Performance parameters
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef LAVA_PIM_PARAMS_PERF_H
#define LAVA_PIM_PARAMS_PERF_H

#include "libpimeval.h"
#include "pimParamsDram.h"
#include "pimCmd.h"
#include "pimResMgr.h"


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
  const double m_nano_to_milli = 1000000.0;
  double m_tR; // Row read latency in ms
  double m_tW; // Row write latency in ms
  double m_tL; // Logic operation for bitserial / tCCD in ms
  double m_tGDL; // Fetch data from local row buffer to global row buffer
  int m_GDLWidth; // Number of bits that can be fetched from local to global row buffer.
  double m_fulcrumAluLatency = 0.00000609; // 6.09ns
  unsigned m_flucrumAluBitWidth = 32;
  double m_blimpCoreLatency = 0.000005; // 200 MHz. Reference: BLIMP paper
  unsigned m_blimpCoreBitWidth = 64; 
};

#endif

