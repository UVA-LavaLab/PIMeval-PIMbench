// File: pimPerfEnergyBase.h
// PIMeval Simulator - Performance Energy Models
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef LAVA_PIM_PERF_ENERGY_BASE_H
#define LAVA_PIM_PERF_ENERGY_BASE_H

#include "libpimeval.h"                // for PimDeviceEnum, PimDataType
#include "pimParamsDram.h"             // for pimParamsDram
#include "pimCmd.h"                    // for PimCmdEnum
#include "pimResMgr.h"                 // for pimObjInfo


namespace pimeval {
  class perfEnergy
  {
    public:
      perfEnergy() : m_msRuntime(0.0), m_mjEnergy(0.0) {}
      perfEnergy(double msRuntime, double mjEnergy) : m_msRuntime(msRuntime), m_mjEnergy(mjEnergy) {}

      double m_msRuntime;
      double m_mjEnergy;
  };
}

//! @class  pimPerfEnergyFactory
//! @brief  PIM performance energy model factory
class pimPerfEnergyBase;
class pimPerfEnergyFactory
{
public:
  static pimPerfEnergyBase* createPerfEnergyModel(PimDeviceEnum simTarget, pimParamsDram* paramsDram);
};

//! @class  pimPerfEnergyBase
//! @brief  PIM performance energy model base class
class pimPerfEnergyBase
{
public:
  pimPerfEnergyBase(pimParamsDram* paramsDram);
  virtual ~pimPerfEnergyBase() {}

  virtual pimeval::perfEnergy getPerfEnergyForBytesTransfer(PimCmdEnum cmdType, uint64_t numBytes) const;
  virtual pimeval::perfEnergy getPerfEnergyForFunc1(PimCmdEnum cmdType, const pimObjInfo& obj) const;
  virtual pimeval::perfEnergy getPerfEnergyForFunc2(PimCmdEnum cmdType, const pimObjInfo& obj) const;
  virtual pimeval::perfEnergy getPerfEnergyForRedSum(PimCmdEnum cmdType, const pimObjInfo& obj, unsigned numPass) const;
  virtual pimeval::perfEnergy getPerfEnergyForBroadcast(PimCmdEnum cmdType, const pimObjInfo& obj) const;
  virtual pimeval::perfEnergy getPerfEnergyForRotate(PimCmdEnum cmdType, const pimObjInfo& obj) const;

protected:
  const pimParamsDram* m_paramsDram;
  const double m_nano_to_milli = 1000000.0;
  const double m_pico_to_milli = 1000000000.0;
  double m_tR; // Row read latency in ms
  double m_tW; // Row write latency in ms
  double m_tL; // Logic operation for bitserial / tCCD in ms
  double m_tGDL; // Fetch data from local row buffer to global row buffer
  int m_GDLWidth; // Number of bits that can be fetched from local to global row buffer.
  int m_numChipsPerRank; // Number of chips per rank

  double m_eAP; // Row read(ACT) energy in mJ microjoule
  double m_eL; // Logic energy in mJ microjoule
  double m_eR; // Read data from PIM
  double m_eW; // Write data to PIM
  double m_pBCore; // background power for each core in W
  double m_pBChip; // background power for each core in W
  double m_eGDL = 0.0000102; // CAS energy in mJ
};

#endif

