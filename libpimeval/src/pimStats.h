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

  void startKernelTimer();
  void endKernelTimer();

  void showStats() const;
  void resetStats();

  void recordCmd(const std::string& cmdName, pimeval::perfEnergy mPerfEnergy);
  void recordCopyMainToDevice(uint64_t numBits, pimeval::perfEnergy mPerfEnergy);
  void recordCopyDeviceToMain(uint64_t numBits, pimeval::perfEnergy mPerfEnergy);
  void recordCopyDeviceToDevice(uint64_t numBits, pimeval::perfEnergy mPerfEnergy);

private:
  friend class pimPerfMon;
  void pimApiScopeStart();
  void pimApiScopeEnd(const std::string& tag, double elapsed);

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

  bool m_isKernelTimerOn = false;
  double m_curApiMsEstRuntime = 0.0;
  double m_kernelMsElapsedSim = 0.0;
  double m_kernelMsEstRuntime = 0.0;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_kernelStart{};
};

#endif

