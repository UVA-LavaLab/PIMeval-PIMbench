// File: pimStats.cpp
// PIMeval Simulator - Stats
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimStats.h"
#include "pimSim.h"
#include "pimUtils.h"
#include <chrono>            // for chrono
#include <cstdint>           // for uint64_t
#include <cstdio>            // for printf
#include <iostream>          // for cout
#include <iomanip>           // for setw, fixed, setprecision


//! @brief  Show PIM stats
void
pimStatsMgr::showStats() const
{
  std::printf("----------------------------------------\n");
  if (pimSim::get()->isDebug(pimSimConfig::DEBUG_API_CALLS)) {
    showApiStats();
  }
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
  const pimParamsDram& paramsDram = pimSim::get()->getParamsDram();
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
  std::printf(" %30s : %f GB/s\n", "Typical Rank BW", paramsDram.getTypicalRankBW());
  std::printf(" %30s : %f\n", "Row Read (ns)", paramsDram.getNsRowRead());
  std::printf(" %30s : %f\n", "Row Write (ns)", paramsDram.getNsRowWrite());
  std::printf(" %30s : %f\n", "tCCD (ns)", paramsDram.getNsTCCD_S());
  if (pimSim::get()->isDebug(pimSimConfig::DEBUG_PERF)) {
    std::printf(" %30s : %f\n", "AAP (ns)", paramsDram.getNsAAP());
  }
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
  double totalMsRuntime = m_elapsedTimeCopiedMainToDevice + m_elapsedTimeCopiedDeviceToMain + m_elapsedTimeCopiedDeviceToDevice;
  double totalMjEnergy = m_mJCopiedMainToDevice + m_mJCopiedDeviceToMain + m_mJCopiedDeviceToDevice;
  std::printf(" %45s : %llu bytes\n", "Host to Device", (unsigned long long)bytesCopiedMainToDevice);
  std::printf(" %45s : %llu bytes\n", "Device to Host", (unsigned long long)bytesCopiedDeviceToMain);
  std::printf(" %45s : %llu bytes\n", "Device to Device", (unsigned long long)bytesCopiedDeviceToDevice);
  std::printf(" %45s : %llu bytes %14.6f ms Estimated Runtime %14.6f mj Estimated Energy\n", "TOTAL ---------", (unsigned long long)totalBytes, totalMsRuntime, totalMjEnergy);
}

//! @brief  Show PIM cmd and perf stats
void
pimStatsMgr::showCmdStats() const
{
  std::printf("PIM Command Stats:\n");
  std::printf(" %44s : %10s %14s %14s %14s %7s %7s %7s\n", "PIM-CMD", "CNT", "Runtime(ms)", "Energy(mJ)", "GOPS/W", "%R", "%W", "%L");
  int totalCmd = 0;
  double totalMsRuntime = 0.0;
  double totalMjEnergy = 0.0;
  double totalMsRead = 0.0;
  double totalMsWrite = 0.0;
  double totalMsCompute = 0.0;
  uint64_t totalOp = 0;
  for (const auto& it : m_cmdPerf) {
    double cmdRuntime = it.second.second.m_msRuntime;
    double percentRead = cmdRuntime == 0.0 ? 0.0 : (it.second.second.m_msRead * 100 / cmdRuntime);
    double percentWrite = cmdRuntime == 0.0 ? 0.0 : (it.second.second.m_msWrite * 100 / cmdRuntime);
    double percentCompute = cmdRuntime == 0.0 ? 0.0 : (it.second.second.m_msCompute * 100 / cmdRuntime);
    double cmdEnergy = it.second.second.m_mjEnergy;
    double perfWatt = cmdEnergy == 0.0 ? 0.0 : (it.second.second.m_totalOp * 1.0 / cmdEnergy * 1e-6);
    std::printf(" %44s : %10d %14f %14f %14f %7.2f %7.2f %7.2f\n", it.first.c_str(), it.second.first, it.second.second.m_msRuntime, it.second.second.m_mjEnergy, perfWatt, percentRead, percentWrite, percentCompute);
    totalCmd += it.second.first;
    totalMsRuntime += it.second.second.m_msRuntime;
    totalMjEnergy += it.second.second.m_mjEnergy;
    totalMsRead += it.second.first * percentRead;
    totalMsWrite += it.second.first * percentWrite;
    totalMsCompute += it.second.first * percentCompute;
    totalOp += it.second.second.m_totalOp;
  }
  std::printf(" %44s : %10d %14f %14f %14f %7.2f %7.2f %7.2f\n", "TOTAL ---------", totalCmd, totalMsRuntime, totalMjEnergy, (totalOp * 1.0 / totalMjEnergy * 1e-6), (totalMsRead / totalCmd), (totalMsWrite / totalCmd), (totalMsCompute / totalCmd) );
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

//! @brief  Record estimated runtime and energy of a PIM command
void
pimStatsMgr::recordCmd(const std::string& cmdName, pimeval::perfEnergy mPerfEnergy)
{
  auto& item = m_cmdPerf[cmdName];
  item.first++;
  item.second.m_msRuntime += mPerfEnergy.m_msRuntime;
  m_curApiMsEstRuntime += mPerfEnergy.m_msRuntime;
  item.second.m_mjEnergy += mPerfEnergy.m_mjEnergy;
  item.second.m_msRead += mPerfEnergy.m_msRead;
  item.second.m_msWrite += mPerfEnergy.m_msWrite;
  item.second.m_msCompute += mPerfEnergy.m_msCompute;
  item.second.m_totalOp += mPerfEnergy.m_totalOp;
}

//! @brief  Record estimated runtime and energy of data copy
void
pimStatsMgr::recordCopyMainToDevice(uint64_t numBits, pimeval::perfEnergy mPerfEnergy)
{
  m_bitsCopiedMainToDevice += numBits;
  m_elapsedTimeCopiedMainToDevice += mPerfEnergy.m_msRuntime;
  m_curApiMsEstRuntime += mPerfEnergy.m_msRuntime;
  m_mJCopiedMainToDevice += mPerfEnergy.m_mjEnergy;
}

//! @brief  Record estimated runtime and energy of data copy
void
pimStatsMgr::recordCopyDeviceToMain(uint64_t numBits, pimeval::perfEnergy mPerfEnergy)
{
  m_bitsCopiedDeviceToMain += numBits;
  m_elapsedTimeCopiedDeviceToMain += mPerfEnergy.m_msRuntime;
  m_curApiMsEstRuntime += mPerfEnergy.m_msRuntime;
  m_mJCopiedDeviceToMain += mPerfEnergy.m_mjEnergy;
}

//! @brief  Record estimated runtime and energy of data copy
void
pimStatsMgr::recordCopyDeviceToDevice(uint64_t numBits, pimeval::perfEnergy mPerfEnergy)
{
  m_bitsCopiedDeviceToDevice += numBits;
  m_elapsedTimeCopiedDeviceToDevice += mPerfEnergy.m_msRuntime;
  m_curApiMsEstRuntime += mPerfEnergy.m_msRuntime;
  m_mJCopiedDeviceToDevice += mPerfEnergy.m_mjEnergy;
}

//! @brief  Preprocessing at the beginning of a PIM API scope
void
pimStatsMgr::pimApiScopeStart()
{
  // Restart for current PIM API call
  m_curApiMsEstRuntime = 0.0;
}

//! @brief  Postprocessing at the end of a PIM API scope
void
pimStatsMgr::pimApiScopeEnd(const std::string& tag, double elapsed)
{
  // Record API stats
  auto& item = m_msElapsed[tag];
  item.first++;
  item.second += elapsed;

  // Update kernel stats
  if (m_isKernelTimerOn) {
    m_kernelMsElapsedSim += elapsed;
    m_kernelMsEstRuntime += m_curApiMsEstRuntime;
  }
}

//! @brief  Start timer for a PIM kernel to measure CPU runtime and DRAM refresh
void
pimStatsMgr::startKernelTimer()
{
  if (m_isKernelTimerOn) {
    std::printf("PIM-Warning: Kernel timer has already started\n");
    return;
  }
  std::printf("PIM-Info: Start kernel timer.\n");
  m_isKernelTimerOn = true;
  m_kernelStart = std::chrono::high_resolution_clock::now();
}

//! @brief  End timer for a PIM kernel to measure CPU runtime and DRAM refresh
void
pimStatsMgr::endKernelTimer()
{
  if (!m_isKernelTimerOn) {
    std::printf("PIM-Warning: Kernel timer has not started\n");
    return;
  }
  auto now = std::chrono::high_resolution_clock::now();
  double kernelMsElapsedTotal = std::chrono::duration<double, std::milli>(now - m_kernelStart).count();
  double kernelMsElapsedCpu = kernelMsElapsedTotal - m_kernelMsElapsedSim;
  std::printf("PIM-Info: End kernel timer. Runtime = %14f ms, CPU = %14f ms, PIM = %14f ms\n",
      kernelMsElapsedCpu + m_kernelMsEstRuntime, kernelMsElapsedCpu, m_kernelMsEstRuntime);
  m_kernelStart = std::chrono::high_resolution_clock::time_point(); // reset
  m_isKernelTimerOn = false;
}

//! @brief pimPerfMon ctor
pimPerfMon::pimPerfMon(const std::string& tag)
{
  m_startTime = std::chrono::high_resolution_clock::now();
  m_tag = tag;
  // assumption: pimPerfMon is not nested
  if (pimSim::get()->getStatsMgr()) {
    pimSim::get()->getStatsMgr()->pimApiScopeStart();
  }
}

//! @brief pimPerfMon dtor
pimPerfMon::~pimPerfMon()
{
  auto now = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double, std::milli>(now - m_startTime).count();
  if (pimSim::get()->getStatsMgr()) {
    pimSim::get()->getStatsMgr()->pimApiScopeEnd(m_tag, elapsed);
  }
}

