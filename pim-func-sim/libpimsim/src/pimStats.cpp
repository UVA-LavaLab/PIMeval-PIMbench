// File: pimStats.cpp
// PIM Functional Simulator - Stats
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "pimStats.h"
#include "pimSim.h"


//! @brief  Show PIM stats
void
pimStatsMgr::showStats() const
{
  std::printf("----------------------------------------\n");

  std::printf("PIM API Stats:\n");
  std::printf(" %30s : %10s %12s\n", "PIM-API", "CNT", "Elapsed(ms)");
  int totCalls = 0;
  for (const auto& it : m_msElapsed) {
    std::printf(" %30s : %10d %12f\n", it.first.c_str(), it.second.first, it.second.second);
    totCalls += it.second.first;
  }
  std::printf(" %30s : %10d %12f\n", "TOTAL", totCalls, m_msTotalElapsed);

  std::printf("PIM Command Stats:\n");
  std::printf(" %30s : %s\n", "PIM-CMD", "CNT");
  for (const auto& it : m_cmdCnt) {
    std::printf(" %30s : %d\n", it.first.c_str(), it.second);
  }

  std::printf("----------------------------------------\n");
}


//! @brief  Reset PIM stats
void
pimStatsMgr::resetStats()
{
  m_cmdCnt.clear();
  m_msElapsed.clear();
  m_msTotalElapsed = 0.0;
}

//! @brief pimPerfMon ctor
pimPerfMon::pimPerfMon(const std::string& tag)
{
  m_startTime = std::chrono::high_resolution_clock::now();
  m_tag = tag;
}

//! @brief pimPerfMon dtor
pimPerfMon::~pimPerfMon()
{
  auto now = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double, std::milli>(now - m_startTime).count();
  pimSim::get()->getStatsMgr()->recordMsElapsed(m_tag, elapsed);
}

