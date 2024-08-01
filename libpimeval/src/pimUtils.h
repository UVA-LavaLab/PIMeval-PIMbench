// File: pimUtils.h
// PIMeval Simulator - Utilities
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef LAVA_PIM_UTILS_H
#define LAVA_PIM_UTILS_H

#include "libpimeval.h"
#include <string>
#include <queue>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <cctype>
#include <locale>
#include <unordered_map>

namespace pimUtils
{
  std::string pimStatusEnumToStr(PimStatus status);
  std::string pimDeviceEnumToStr(PimDeviceEnum deviceType);
  PimDeviceEnum strToPimDeviceEnum(const std::string& deviceTypeStr);
  std::string pimAllocEnumToStr(PimAllocEnum allocType);
  std::string pimCopyEnumToStr(PimCopyEnum copyType);
  std::string pimDataTypeEnumToStr(PimDataType dataType);
  std::string& ltrim(std::string& s);
  std::string& rtrim(std::string& s);
  std::string& trim(std::string& s);
  bool readFileContent(const char* fileName, std::string& fileContent);
  std::string getParam(const std::unordered_map<std::string, std::string>& params, const std::string& key);
  std::string getOptionalParam(const std::unordered_map<std::string, std::string>& params, const std::string& key, bool& returnStatus);
  std::string removeAfterSemicolon(const std::string &input);

  std::vector<bool> readBitsFromHost(void* src, uint64_t numElements, unsigned bitsPerElement);
  bool writeBitsToHost(void* dest, const std::vector<bool>& bits);
  std::string getDirectoryPath(const std::string& filePath);
  bool getEnvVar(const std::string &varName, std::string &varValue);

  const std::unordered_map<PimDeviceEnum, std::string> enumToStrMap = {
      {PIM_DEVICE_NONE, "PIM_DEVICE_NONE"},
      {PIM_FUNCTIONAL, "PIM_FUNCTIONAL"},
      {PIM_DEVICE_BITSIMD_V, "PIM_DEVICE_BITSIMD_V"},
      {PIM_DEVICE_BITSIMD_V_NAND, "PIM_DEVICE_BITSIMD_V_NAND"},
      {PIM_DEVICE_BITSIMD_V_MAJ, "PIM_DEVICE_BITSIMD_V_MAJ"},
      {PIM_DEVICE_BITSIMD_V_AP, "PIM_DEVICE_BITSIMD_V_AP"},
      {PIM_DEVICE_DRISA_NOR, "PIM_DEVICE_DRISA_NOR"},
      {PIM_DEVICE_DRISA_MIXED, "PIM_DEVICE_DRISA_MIXED"},
      {PIM_DEVICE_SIMDRAM, "PIM_DEVICE_SIMDRAM"},
      {PIM_DEVICE_BITSIMD_H, "PIM_DEVICE_BITSIMD_H"},
      {PIM_DEVICE_FULCRUM, "PIM_DEVICE_FULCRUM"},
      {PIM_DEVICE_BANK_LEVEL, "PIM_DEVICE_BANK_LEVEL"}
  };

  const std::unordered_map<std::string, PimDeviceEnum> strToEnumMap = {
      {"PIM_DEVICE_NONE", PIM_DEVICE_NONE},
      {"PIM_FUNCTIONAL", PIM_FUNCTIONAL},
      {"PIM_DEVICE_BITSIMD_V", PIM_DEVICE_BITSIMD_V},
      {"PIM_DEVICE_BITSIMD_V_NAND", PIM_DEVICE_BITSIMD_V_NAND},
      {"PIM_DEVICE_BITSIMD_V_MAJ", PIM_DEVICE_BITSIMD_V_MAJ},
      {"PIM_DEVICE_BITSIMD_V_AP", PIM_DEVICE_BITSIMD_V_AP},
      {"PIM_DEVICE_DRISA_NOR", PIM_DEVICE_DRISA_NOR},
      {"PIM_DEVICE_DRISA_MIXED", PIM_DEVICE_DRISA_MIXED},
      {"PIM_DEVICE_SIMDRAM", PIM_DEVICE_SIMDRAM},
      {"PIM_DEVICE_BITSIMD_H", PIM_DEVICE_BITSIMD_H},
      {"PIM_DEVICE_FULCRUM", PIM_DEVICE_FULCRUM},
      {"PIM_DEVICE_BANK_LEVEL", PIM_DEVICE_BANK_LEVEL}
  };

  static constexpr const char* envVarPimEvalTarget = "PIMEVAL_TARGET";
  static constexpr const char* envVarPimEvalConfigPath = "PIMEVAL_CONFIG_PATH";
  static constexpr const char* envVarPimEvalConfigSim = "PIMEVAL_CONFIG_SIM";

  //! @class  threadWorker
  //! @brief  Thread worker base class
  class threadWorker {
  public:
    threadWorker() {}
    virtual ~threadWorker() {}
    virtual void execute() = 0;
  };

  //! @class  threadPool
  //! @brief  Thread pool that runs multiple workers in threads
  class threadPool {
  public:
    threadPool(size_t numThreads);
    ~threadPool();
    void doWork(const std::vector<pimUtils::threadWorker*>& workers);
  private:
    void workerThread();

    std::vector<std::thread> m_threads;
    std::queue<threadWorker*> m_workers;
    std::mutex m_mutex;
    std::condition_variable m_cond;
    bool m_terminate;
    std::atomic<size_t> m_workersRemaining;
  };

}

#endif

