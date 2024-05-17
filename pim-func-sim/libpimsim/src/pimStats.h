// File: pimStats.h
// PIM Functional Simulator - Stats
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef LAVA_PIM_STATS_H
#define LAVA_PIM_STATS_H

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

  void recordCmd(const std::string& cmdName) {
    m_cmdCnt[cmdName]++;
  }

  void recordMsElapsed(const std::string& tag, double elapsed) {
    auto& item = m_msElapsed[tag];
    item.first++;
    item.second += elapsed;
    m_msTotalElapsed += elapsed;
  }
  double getMsTotalElapsed() const { return m_msTotalElapsed; }

  void resetStats();

private:
  std::map<std::string, int> m_cmdCnt;
  std::map<std::string, std::pair<int, double>> m_msElapsed;
  double m_msTotalElapsed = 0.0;
};

#endif

