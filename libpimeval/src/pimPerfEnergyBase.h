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
#include <cstdint>
#include <memory>                      // for std::unique_ptr


namespace pimeval {
  class perfEnergy
  {
    public:
      perfEnergy() : m_msRuntime(0.0), m_mjEnergy(0.0), m_msRead(0.0), m_msWrite(0.0), m_msCompute(0.0), m_totalOp(0) {}
      perfEnergy(double msRuntime, double mjEnergy, double msRead, double msWrite, double msCompute, uint64_t totalOp) : m_msRuntime(msRuntime), m_mjEnergy(mjEnergy), m_msRead(msRead), m_msWrite(msWrite), m_msCompute(msCompute), m_totalOp(totalOp)  {}

      double m_msRuntime;
      double m_mjEnergy;
      double m_msRead;
      double m_msWrite;
      double m_msCompute;
      uint64_t m_totalOp;
  };
}

//! @class  pimPerfEnergyModelParams
//! @brief  Parameters for creating perf energy models
class pimPerfEnergyModelParams
{
public:
  pimPerfEnergyModelParams(PimDeviceEnum simTarget, unsigned numRanks, const pimParamsDram& paramsDram)
    : m_simTarget(simTarget), m_numRanks(numRanks), m_paramsDram(paramsDram) {}
  PimDeviceEnum getSimTarget() const { return m_simTarget; }
  unsigned getNumRanks() const { return m_numRanks; }
  const pimParamsDram& getParamsDram() const { return m_paramsDram; }
private:
  PimDeviceEnum m_simTarget;
  unsigned m_numRanks;
  const pimParamsDram& m_paramsDram;
};

//! @class  pimPerfEnergyFactory
//! @brief  PIM performance energy model factory
class pimPerfEnergyBase;
class pimPerfEnergyFactory
{
public:
  // pimDevice is not fully constructed at this point. Do not call pimSim::get() in pimPerfEnergyBase ctor.
  static std::unique_ptr<pimPerfEnergyBase> createPerfEnergyModel(const pimPerfEnergyModelParams& params);
};

//! @class  pimPerfEnergyBase
//! @brief  PIM performance energy model base class
class pimPerfEnergyBase
{
public:
  pimPerfEnergyBase(const pimPerfEnergyModelParams& params);
  virtual ~pimPerfEnergyBase() {}

  virtual pimeval::perfEnergy getPerfEnergyForBytesTransfer(PimCmdEnum cmdType, uint64_t numBytes) const;
  virtual pimeval::perfEnergy getPerfEnergyForFunc1(PimCmdEnum cmdType, const pimObjInfo& objSrc, const pimObjInfo& objDest) const;
  virtual pimeval::perfEnergy getPerfEnergyForFunc2(PimCmdEnum cmdType, const pimObjInfo& objSrc1, const pimObjInfo& objSrc2, const pimObjInfo& objDest) const;
  virtual pimeval::perfEnergy getPerfEnergyForReduction(PimCmdEnum cmdType, const pimObjInfo& obj, unsigned numPass) const;
  virtual pimeval::perfEnergy getPerfEnergyForBroadcast(PimCmdEnum cmdType, const pimObjInfo& obj) const;
  virtual pimeval::perfEnergy getPerfEnergyForRotate(PimCmdEnum cmdType, const pimObjInfo& obj) const;
  virtual pimeval::perfEnergy getPerfEnergyForLut(PimCmdEnum cmdType, const pimObjInfo& objSrc, const pimObjInfo& objDest) const;

protected:
  PimDeviceEnum m_simTarget;
  unsigned m_numRanks;
  const pimParamsDram& m_paramsDram;

  const double m_nano_to_milli = 1000000.0;
  const double m_pico_to_milli = 1000000000.0;
  double m_tR; // Row read latency in ms
  double m_tW; // Row write latency in ms
  double m_tL; // Logic operation for bitserial / tCCD in ms
  double m_tGDL; // Fetch data from local row buffer to global row buffer
  int m_GDLWidth; // Number of bits that can be fetched from local to global row buffer.
  int m_numChipsPerRank; // Number of chips per rank
  double m_typicalRankBW; // typical rank data transfer bandwidth in GB/s

  double m_eAP; // Row read(ACT) energy in mJ microjoule
  double m_eL; // Logic energy in mJ microjoule
  double m_eR; // local row buffer to global row buffer
  double m_eW; // global row buffer to local row buffer
  double m_pBCore; // background power for each core in W
  double m_pBChip; // background power for each core in W
};

#endif

