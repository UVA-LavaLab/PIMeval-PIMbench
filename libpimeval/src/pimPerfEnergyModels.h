// File: pimPerfEnergyModels.h
// PIMeval Simulator - Performance Energy Models
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef LAVA_PIM_PERF_ENERGY_MODELS_H
#define LAVA_PIM_PERF_ENERGY_MODELS_H

#include "libpimeval.h"
#include "pimParamsDram.h"
#include "pimCmd.h"
#include "pimResMgr.h"


namespace pimNS {
  class perfEnergy
  {
    public:
      perfEnergy() : m_msRuntime(0.0), m_mjEnergy(0.0) {}
      perfEnergy(double msRuntime, double mjEnergy) : m_msRuntime(msRuntime), m_mjEnergy(mjEnergy) {}

      double m_msRuntime;
      double m_mjEnergy;
  };
}


//! @class  pimPerfEnergyBase
//! @brief  PIM performance energy model base class
class pimPerfEnergyBase
{
public:
  pimPerfEnergyBase(pimParamsDram* paramsDram);
  ~pimPerfEnergyBase() {}

  pimNS::perfEnergy getPerfEnergyForBytesTransfer(PimCmdEnum cmdType, uint64_t numBytes) const;
  pimNS::perfEnergy getPerfEnergyForFunc1(PimCmdEnum cmdType, const pimObjInfo& obj) const;
  pimNS::perfEnergy getPerfEnergyForFunc2(PimCmdEnum cmdType, const pimObjInfo& obj) const;
  pimNS::perfEnergy getPerfEnergyForRedSum(PimCmdEnum cmdType, const pimObjInfo& obj, unsigned numPass) const;
  pimNS::perfEnergy getPerfEnergyForBroadcast(PimCmdEnum cmdType, const pimObjInfo& obj) const;
  pimNS::perfEnergy getPerfEnergyForRotate(PimCmdEnum cmdType, const pimObjInfo& obj) const;

private:
  pimNS::perfEnergy getPerfEnergyBitSerial(PimDeviceEnum deviceType, PimCmdEnum cmdType, PimDataType dataType, unsigned bitsPerElement, unsigned numPass, const pimObjInfo& obj) const;

  const pimParamsDram* m_paramsDram;
  const double m_nano_to_milli = 1000000.0;
  const double m_pico_to_milli = 1000000000.0;
  double m_tR; // Row read latency in ms
  double m_tW; // Row write latency in ms
  double m_tL; // Logic operation for bitserial / tCCD in ms
  double m_tGDL; // Fetch data from local row buffer to global row buffer
  int m_GDLWidth; // Number of bits that can be fetched from local to global row buffer.
  int m_numChipsPerRank; // Number of chips per rank
  double m_fulcrumAluLatency = 0.00000609; // 6.09ns

  unsigned m_flucrumAluBitWidth = 32;
  double m_blimpCoreLatency = 0.000005; // ms; 200 MHz. Reference: BLIMP paper
  unsigned m_blimpCoreBitWidth = 64;

  double m_eAP; // Row read(ACT) energy in mJ microjoule
  double m_eL; // Logic energy in mJ microjoule
  double m_eR; // Read data from PIM
  double m_eW; // Write data to PIM
  double m_pBCore; // background power for each core in W
  double m_pBChip; // background power for each core in W
  double m_eGDL = 0.0000102; // CAS energy in mJ

  // Popcount logc Params from DRAM-CAM paper
  double m_pclNsDelay = 0.76; // 64-bit popcount logic ns delay, using LUT no pipeline design
  double m_pclUwPower = 0.03; // 64-bit popcount logic uW power, using LUT no pipeline design

  // Following values are taken from fulcrum paper.
  double m_fulcrumALUArithmeticEnergy = 0.0000000004992329586; // mJ
  double m_fulcrumALULogicalEnergy = 0.0000000001467846411; // mJ
  double m_fulcrumShiftEnergy = 0.0000000075; // mJ

  // Following values are taken from fulcrum paper as BLIMP paper does not model energy
  double m_blimpArithmeticEnergy = 0.0000000004992329586; // mJ
  double m_blimpLogicalEnergy = 0.0000000001467846411; // mJ
};

#endif

