// File: pimStats.cpp
// PIM Functional Simulator - Stats
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "pimStats.h"
#include "pimSim.h"
#include "pimUtils.h"
#include <algorithm>


//! @brief  Show PIM stats
void
pimStatsMgr::showStats() const
{
  std::printf("----------------------------------------\n");
  showApiStats();
  showDeviceParams();
  showCopyStats();
  showCmdStats();
  std::printf("----------------------------------------\n");
}

//! @brief  Show API stats
void
pimStatsMgr::showApiStats() const
{
  std::printf("Simulator API Stats:\n");
  std::printf(" %30s : %10s %14s\n", "PIM-API", "CNT", "Elapsed(ms)");
  int totCalls = 0;
  int totCallsDevice = 0;
  int totCallsAlloc = 0;
  int totCallsCopy = 0;
  int totCallsCompute = 0;
  double msTotalElapsed = 0.0;
  double msTotalElapsedDevice = 0.0;
  double msTotalElapsedAlloc = 0.0;
  double msTotalElapsedCopy = 0.0;
  double msTotalElapsedCompute = 0.0;
  for (const auto& it : m_msElapsed) {
    std::printf(" %30s : %10d %14f\n", it.first.c_str(), it.second.first, it.second.second);
    totCalls += it.second.first;
    msTotalElapsed += it.second.second;
    if (it.first.find("createDevice") == 0) {
      totCallsDevice += it.second.first;
      msTotalElapsedDevice += it.second.second;
    } else if (it.first.find("pimAlloc") == 0 || it.first.find("pimFree") == 0) {
      totCallsAlloc += it.second.first;
      msTotalElapsedAlloc += it.second.second;
    } else if (it.first.find("pimCopy") == 0) {
      totCallsCopy += it.second.first;
      msTotalElapsedCopy += it.second.second;
    } else {
      totCallsCompute += it.second.first;
      msTotalElapsedCompute += it.second.second;
    }
  }
  std::printf(" %30s : %10d %14f\n", "TOTAL ---------", totCalls, msTotalElapsed);
  std::printf(" %30s : %10d %14f\n", "TOTAL (Device )", totCallsDevice, msTotalElapsedDevice);
  std::printf(" %30s : %10d %14f\n", "TOTAL ( Alloc )", totCallsAlloc, msTotalElapsedAlloc);
  std::printf(" %30s : %10d %14f\n", "TOTAL (  Copy )", totCallsCopy, msTotalElapsedCopy);
  std::printf(" %30s : %10d %14f\n", "TOTAL (Compute)", totCallsCompute, msTotalElapsedCompute);
}

//! @brief  Show PIM device params
void
pimStatsMgr::showDeviceParams() const
{
  std::printf("PIM Params:\n");
  std::printf(" %30s : %s\n", "PIM Device Type Enum",
              pimUtils::pimDeviceEnumToStr(pimSim::get()->getDeviceType()).c_str());
  std::printf(" %30s : %s\n", "PIM Simulation Target",
              pimUtils::pimDeviceEnumToStr(pimSim::get()->getSimTarget()).c_str());
  std::printf(" %30s : %u, %u, %u, %u, %u\n", "Rank, Bank, Subarray, Row, Col",
              pimSim::get()->getNumRanks(),
              pimSim::get()->getNumBankPerRank(),
              pimSim::get()->getNumSubarrayPerBank(),
              pimSim::get()->getNumRowPerSubarray(),
              pimSim::get()->getNumColPerSubarray());
  std::printf(" %30s : %u\n", "Number of PIM Cores", pimSim::get()->getNumCores());
  std::printf(" %30s : %u\n", "Number of Rows per Core", pimSim::get()->getNumRows());
  std::printf(" %30s : %u\n", "Number of Cols per Core", pimSim::get()->getNumCols());
  std::printf(" %30s : %f GB/s\n", "Typical Rank BW", m_paramsDram->getTypicalRankBW());
  std::printf(" %30s : %f\n", "Row Read (ns)", m_paramsDram->getNsRowRead());
  std::printf(" %30s : %f\n", "Row Write (ns)", m_paramsDram->getNsRowWrite());
  std::printf(" %30s : %f\n", "tCCD (ns)", m_paramsDram->getNsTCCD());
  std::printf(" %30s : %f\n", "AAP (ns)", m_paramsDram->getNsAAP());
}

//! @brief  Show data copy stats
void
pimStatsMgr::showCopyStats() const
{
  std::printf("Data Copy Stats:\n");
  uint64_t bytesCopiedMainToDevice = m_bitsCopiedMainToDevice / 8;
  uint64_t bytesCopiedDeviceToMain = m_bitsCopiedDeviceToMain / 8;
  uint64_t bytesCopiedDeviceToDevice = m_bitsCopiedDeviceToDevice / 8;
  uint64_t totalBytes = bytesCopiedMainToDevice + bytesCopiedDeviceToMain;
  double totalMsRuntime = m_paramsPerf->getMsRuntimeForBytesTransfer(totalBytes);
  std::printf(" %30s : %lu bytes\n", "Host to Device", bytesCopiedMainToDevice);
  std::printf(" %30s : %lu bytes\n", "Device to Host", bytesCopiedDeviceToMain);
  std::printf(" %30s : %lu bytes %14f ms Estimated Runtime\n", "TOTAL ---------", totalBytes, totalMsRuntime);
  std::printf(" %30s : %lu bytes\n", "Device to Device", bytesCopiedDeviceToDevice);
}

//! @brief  Show PIM cmd and perf stats
void
pimStatsMgr::showCmdStats() const
{
  std::printf("PIM Command Stats:\n");
  std::printf(" %30s : %10s %14s\n", "PIM-CMD", "CNT", "EstimatedRuntime(ms)");
  int totalCmd = 0;
  double totalMsRuntime = 0.0;
  for (const auto& it : m_cmdPerf) {
    std::printf(" %30s : %10d %14f\n", it.first.c_str(), it.second.first, it.second.second);
    totalCmd += it.second.first;
    totalMsRuntime += it.second.second;
  }
  std::printf(" %30s : %10d %14f\n", "TOTAL ---------", totalCmd, totalMsRuntime);

  // analyze micro-ops
  int numR = 0;
  int numW = 0;
  int numL = 0;
  int numActivate = 0;
  int numPrecharge = 0;
  for (const auto& it : m_cmdPerf) {
    if (it.first == "row_r") {
      numR += it.second.first;
      numActivate += it.second.first;
      numPrecharge += it.second.first;
    } else if (it.first == "row_w") {
      numW += it.second.first;
      numActivate += it.second.first;
      numPrecharge += it.second.first;
    } else if (it.first.find("rreg.") == 0) {
      numL += it.second.first;
    }
  }
  if (numR > 0 || numW > 0 || numL > 0) {
    std::printf(" %30s : %d, %d, %d\n", "Num Read, Write, Logic", numR, numW, numL);
    std::printf(" %30s : %d, %d\n", "Num Activate, Precharge", numActivate, numPrecharge);
  }
}

//! @brief  Reset PIM stats
void
pimStatsMgr::resetStats()
{
  m_cmdPerf.clear();
  m_msElapsed.clear();
  m_bitsCopiedMainToDevice = 0;
  m_bitsCopiedDeviceToMain = 0;
  m_bitsCopiedDeviceToDevice = 0;
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

