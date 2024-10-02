// File: pimStats.h
// PIMeval Simulator - Stats
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef LAVA_PIM_STATS_H
#define LAVA_PIM_STATS_H

#include "pimParamsDram.h"
#include "pimPerfEnergyBase.h"
#include "libpimeval.h"
#include <cstdint>
#include <string>
#include <map>
#include <chrono>

//! @class  pimPerfMon
//! @brief  PIM performance monitor
class pimPerfMon
{
public:
  pimPerfMon(const std::string& tag);
  ~pimPerfMon();

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> m_startTime;
  std::string m_tag;
};


//! @class  pimStats
//! @brief  PIM stats manager
class pimStatsMgr
{
public:
  pimStatsMgr() {}
  ~pimStatsMgr() {}

  void showStats() const;
  void resetStats();
  
  void recordCmd(const std::string& cmdName, pimeval::perfEnergy mPerfEnergy) {
    auto& item = m_cmdPerf[cmdName];
    item.first++;
    item.second.m_msRuntime += mPerfEnergy.m_msRuntime;
    item.second.m_mjEnergy += mPerfEnergy.m_mjEnergy;
  }

  void recordMsElapsed(const std::string& tag, double elapsed) {
    auto& item = m_msElapsed[tag];
    item.first++;
    item.second += elapsed;
  }

  void recordCopyMainToDevice(uint64_t numBits, pimeval::perfEnergy mPerfEnergy) {
    m_bitsCopiedMainToDevice += numBits;
    m_elapsedTimeCopiedMainToDevice += mPerfEnergy.m_msRuntime;
    m_mJCopiedMainToDevice += mPerfEnergy.m_mjEnergy;
  }

  void recordCopyDeviceToMain(uint64_t numBits, pimeval::perfEnergy mPerfEnergy) {
    m_bitsCopiedDeviceToMain += numBits; 
    m_elapsedTimeCopiedDeviceToMain += mPerfEnergy.m_msRuntime;
    m_mJCopiedDeviceToMain += mPerfEnergy.m_mjEnergy;
  }
  
  void recordCopyDeviceToDevice(uint64_t numBits, pimeval::perfEnergy mPerfEnergy) {
    m_bitsCopiedDeviceToDevice += numBits;
    m_elapsedTimeCopiedDeviceToDevice += mPerfEnergy.m_msRuntime;
    m_mJCopiedDeviceToDevice += mPerfEnergy.m_mjEnergy;
  }

private:
  void showApiStats() const;
  void showDeviceParams() const;
  void showCopyStats() const;
  void showCmdStats() const;

  std::map<std::string, std::pair<int, pimeval::perfEnergy>> m_cmdPerf;
  std::map<std::string, std::pair<int, double>> m_msElapsed;

  uint64_t m_bitsCopiedMainToDevice = 0;
  uint64_t m_bitsCopiedDeviceToMain = 0;
  uint64_t m_bitsCopiedDeviceToDevice = 0;
  double m_elapsedTimeCopiedMainToDevice = 0.0;
  double m_elapsedTimeCopiedDeviceToMain = 0.0;
  double m_elapsedTimeCopiedDeviceToDevice = 0.0;
  double m_mJCopiedMainToDevice = 0.0;
  double m_mJCopiedDeviceToMain = 0.0;
  double m_mJCopiedDeviceToDevice = 0.0;
};

#endif

